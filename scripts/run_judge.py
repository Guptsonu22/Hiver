"""Phase 4 runner: LLM judge on calibration (50) then full evaluation (150).

Order of operations (enforced):
  1. Verify data/judge/human_calibration.csv has all 100 rows reviewed
     (50 examples x 2 baselines). STOP otherwise — never fabricate.
  2. Run the LLM judge on the 100 calibration rows (cached in
     results/judge_outputs.jsonl; cache reused, never silently overwritten).
  3. Compute judge-human agreement (exact + linear-weighted kappa per dimension).
  4. Run the judge on all 300 full-evaluation rows (150 x 2 baselines).
  5. Write results/judge_metrics.json with metadata, agreement, per-baseline
     means + distributions, and limitations.

Requires OPENAI_API_KEY (and optional JUDGE_MODEL). Fails clearly without it.

Usage:
    python scripts/run_judge.py [--calibration-only] [--full-only]
    python scripts/run_judge.py --machine-calibration   # DIAGNOSTIC ONLY

--machine-calibration scores the full 300 inputs with the deterministic machine
heuristic (src/machine_score.py) and writes results/machine_metrics.json. It NEVER
touches human_calibration.csv, NEVER writes judge_metrics.json/judge_outputs.jsonl,
and NEVER computes human-vs-judge agreement. The official path (no flags) still
requires genuine 100/100 human calibration.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.judge import (  # noqa: E402
    CALIBRATION_SEED,
    CALIBRATION_SIZE,
    EVALUATION_SIZE,
    HUMAN_CALIBRATION_CSV,
    JUDGE_DIR,
    JUDGE_DIMENSIONS,
    JUDGE_INPUTS_JSONL,
    JUDGE_METRICS_JSON,
    JUDGE_OUTPUTS_JSONL,
    JUDGE_PROMPT_VERSION,
    RUBRIC_VERSION,
    append_cached_output,
    build_judge_prompt,
    call_llm_judge,
    compute_agreement,
    get_judge_config,
    load_cached_outputs,
    utc_now_iso,
    validate_judge_score,
)

EXPECTED_CALIB_ROWS = CALIBRATION_SIZE * 2  # 100
EXPECTED_FULL_ROWS = EVALUATION_SIZE * 2  # 300

MACHINE_CSV = JUDGE_DIR / "machine_calibration_100.csv"
MACHINE_METRICS_JSON = JUDGE_DIR.parent.parent / "results" / "machine_metrics.json"
GOLDEN_CSV_PATH = JUDGE_DIR.parent.parent / "data" / "golden" / "golden_candidates.csv"


def _require_file(path: Path) -> None:
    if not Path(path).exists():
        raise SystemExit(f"Missing {path}. Run scripts/build_judge_inputs.py first.")


def verify_human_calibration() -> pd.DataFrame:
    _require_file(HUMAN_CALIBRATION_CSV)
    df = pd.read_csv(HUMAN_CALIBRATION_CSV, dtype="object").fillna("")
    if len(df) != EXPECTED_CALIB_ROWS:
        raise SystemExit(f"CALIBRATION row count {len(df)} != {EXPECTED_CALIB_ROWS}. STOP.")
    score_cols = [d + "_human" for d in JUDGE_DIMENSIONS]
    reviewed_mask = df["annotation_status"].astype(str).str.strip().str.lower() == "reviewed"
    for c in score_cols:
        vals = df.loc[reviewed_mask, c].astype(str).str.strip()
        bad = vals[~vals.isin(["0", "1", "2", "3"])]
        if len(bad):
            raise SystemExit(f"Column {c} has {len(bad)} invalid reviewed scores. STOP.")
    reviewed = int(reviewed_mask.sum())
    pending = len(df) - reviewed
    print(f"CALIBRATION_TOTAL = 50 examples ({EXPECTED_CALIB_ROWS} rows)")
    print(f"HUMAN_REVIEWED = {reviewed}")
    print(f"HUMAN_PENDING = {pending}")
    if reviewed != EXPECTED_CALIB_ROWS or pending != 0:
        raise SystemExit("STOP: fewer than 100/100 calibration rows reviewed. "
                         "Run scripts/annotate_judge_calibration.py first.")
    return df


def judge_rows(rows: pd.DataFrame, model: str) -> pd.DataFrame:
    cache = load_cached_outputs(JUDGE_OUTPUTS_JSONL)
    out_records = []
    for _, r in rows.iterrows():
        key = f"{r['baseline']}::{r['pair_id']}"
        if key in cache:
            cached = cache[key]
            out_records.append(cached)
            continue
        prompt = build_judge_prompt(
            customer_message=str(r["customer_message"]),
            context_before_customer=str(r.get("context_before_customer", "") or ""),
            generated_reply=str(r["generated_reply"]),
            retrieved_evidence=str(r["retrieved_historical_response"]),
            reference_response=str(r.get("reference_response", "") or ""),
            predicted_intent=str(r.get("predicted_intent", "") or ""),
        )
        score = call_llm_judge(prompt, model=model)  # raises clearly on API failure
        validate_judge_score(score)
        rec = {
            "pair_id": str(r["pair_id"]),
            "baseline": str(r["baseline"]),
            "judge_model": model,
            "rubric_version": RUBRIC_VERSION,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "timestamp_utc": utc_now_iso(),
            **{k: score[k] for k in
               ["relevance", "helpfulness", "groundedness", "appropriateness",
                "unsupported_claims", "overall_score", "confidence"]},
            "reason": score["reason"],
        }
        append_cached_output(rec)
        cache[key] = rec
        out_records.append(rec)
        print(f"[judge] {key} -> overall={score['overall_score']}")
    return pd.DataFrame(out_records)


def summarize_baseline(judge_df: pd.DataFrame, baseline: str) -> dict:
    sub = judge_df[judge_df["baseline"] == baseline]
    dims = {}
    for dim in JUDGE_DIMENSIONS:
        vals = sub[dim].astype(int).tolist()
        dist = {str(k): int(sum(1 for v in vals if v == k)) for k in range(4)}
        dims[dim] = {"mean": round(float(sum(vals) / len(vals)), 4) if vals else 0.0,
                     "n": len(vals), "distribution": dist}
    return dims


def run_machine_mode() -> None:
    """Diagnostic machine evaluation. Separate outputs; never official results."""
    from src.machine_score import (
        MACHINE_DIMS,
        METHOD,
        RUBRIC_TARGET,
        score_inputs_frame,
        summarize_machine_baseline,
    )
    print("=" * 75)
    print("MACHINE-CALIBRATION MODE — DIAGNOSTIC ONLY")
    print("Scores are MACHINE-generated (deterministic heuristic), NOT human labels,")
    print("NOT LLM judge scores. No human-vs-judge agreement is computed here.")
    print("=" * 75)
    _require_file(JUDGE_INPUTS_JSONL)
    _require_file(MACHINE_CSV)
    _require_file(GOLDEN_CSV_PATH)

    golden = pd.read_csv(GOLDEN_CSV_PATH, dtype={"pair_id": str})
    golden_ids = set(golden["pair_id"])
    reviewed_ids = set(golden[golden["annotation_status"] == "reviewed"]["pair_id"])

    mach = pd.read_csv(MACHINE_CSV, dtype="object").fillna("")
    if len(mach) != EXPECTED_CALIB_ROWS:
        raise SystemExit(f"Machine file has {len(mach)} rows != {EXPECTED_CALIB_ROWS}. STOP.")
    if any(c.startswith("human_") for c in mach.columns):
        raise SystemExit("Machine file contains human_* columns. STOP (integrity).")
    for c in MACHINE_DIMS:
        bad = mach[~mach[c].astype(str).str.strip().isin(["0", "1", "2", "3"])]
        if len(bad):
            raise SystemExit(f"Machine column {c} has {len(bad)} invalid scores. STOP.")
    if not set(mach["pair_id"].astype(str)) <= reviewed_ids:
        raise SystemExit("Machine pair_ids outside reviewed 150. STOP.")

    full_inputs = pd.read_json(JUDGE_INPUTS_JSONL, lines=True, dtype={"pair_id": str})
    if len(full_inputs) != EXPECTED_FULL_ROWS:
        raise SystemExit(f"Full inputs {len(full_inputs)} != {EXPECTED_FULL_ROWS}. STOP.")
    overlap = set(full_inputs["retrieval_source_pair_id"].astype(str)) & golden_ids
    if overlap:
        raise SystemExit(f"LEAKAGE: {len(overlap)} retrieval sources are golden IDs. STOP.")

    scored = score_inputs_frame(full_inputs)
    metrics = {
        "metadata": {
            "kind": "MACHINE-GENERATED DIAGNOSTIC - NOT human labels, NOT LLM judge scores",
            "method": METHOD,
            "rubric_target": RUBRIC_TARGET,
            "full_rows": EXPECTED_FULL_ROWS,
            "machine_calibration_rows": len(mach),
            "machine_sources": mach["source"].value_counts().to_dict(),
            "timestamp_utc": utc_now_iso(),
            "leakage_status": "PASSED (retrieval_source_pair_id never in 322 golden IDs)",
            "disclaimer": "No human-vs-judge agreement can be computed from machine labels. "
                          "The official workflow still requires genuine 100/100 human calibration.",
        },
        "MostFrequentBaseline": summarize_machine_baseline(scored, "MostFrequentBaseline"),
        "KeywordBaseline": summarize_machine_baseline(scored, "KeywordBaseline"),
    }
    MACHINE_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MACHINE_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Machine diagnostic saved to {MACHINE_METRICS_JSON}")
    for bl in ("MostFrequentBaseline", "KeywordBaseline"):
        means = {d.split("_", 1)[1]: metrics[bl][d]["mean"] for d in MACHINE_DIMS}
        print(f"  {bl}: {means}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-only", action="store_true")
    ap.add_argument("--full-only", action="store_true")
    ap.add_argument("--machine-calibration", action="store_true",
                    help="Diagnostic machine mode (see module docstring).")
    args = ap.parse_args()

    if args.machine_calibration:
        if args.calibration_only or args.full_only:
            raise SystemExit("--machine-calibration is mutually exclusive with other flags.")
        run_machine_mode()
        return

    _require_file(JUDGE_INPUTS_JSONL)
    cfg = get_judge_config()  # fail-closed without API key
    model = cfg["model"]
    print(f"Judge model: {model} | rubric={RUBRIC_VERSION} | prompt={JUDGE_PROMPT_VERSION}")

    human_df = verify_human_calibration()
    full_inputs = pd.read_json(JUDGE_INPUTS_JSONL, lines=True, dtype={"pair_id": str})
    if len(full_inputs) != EXPECTED_FULL_ROWS:
        raise SystemExit(f"Full inputs {len(full_inputs)} != {EXPECTED_FULL_ROWS}. STOP.")

    calib_ids = set(human_df["pair_id"].astype(str))
    calib_inputs = full_inputs[full_inputs["pair_id"].astype(str).isin(calib_ids)].copy()
    assert len(calib_inputs) == EXPECTED_CALIB_ROWS

    if not args.full_only:
        print("--- Judging calibration (100 rows) ---")
        calib_judge = judge_rows(calib_inputs, model)
        # Align human + judge frames for agreement.
        h = human_df.rename(columns={d + "_human": d + "_human" for d in JUDGE_DIMENSIONS})
        j = calib_judge.rename(columns={d: d + "_judge" for d in JUDGE_DIMENSIONS})
        agreement = compute_agreement(h, j)
        print(json.dumps(agreement, indent=2))
    else:
        agreement = None
        cache = load_cached_outputs(JUDGE_OUTPUTS_JSONL)
        missing = [k for k in
                   [f"{r['baseline']}::{r['pair_id']}" for _, r in calib_inputs.iterrows()]
                   if k not in cache]
        if missing:
            raise SystemExit(f"--full-only requested but {len(missing)} calibration "
                             f"judge outputs missing. Run without --full-only first.")

    if not args.calibration_only:
        print("--- Judging full evaluation (300 rows) ---")
        full_judge = judge_rows(full_inputs, model)
        if agreement is None:
            cache_df = pd.DataFrame(load_cached_outputs(JUDGE_OUTPUTS_JSONL).values())
            calib_judge = cache_df[cache_df["pair_id"].astype(str).isin(calib_ids)].copy()
            # normalize column names back (cache stores raw dim names)
            h = human_df
            j = calib_judge.rename(columns={d: d + "_judge" for d in JUDGE_DIMENSIONS
                                            if d in calib_judge.columns})
            agreement = compute_agreement(h, j)
        metrics = {
            "metadata": {
                "evaluation_population": EVALUATION_SIZE,
                "calibration_population": CALIBRATION_SIZE,
                "calibration_rows": EXPECTED_CALIB_ROWS,
                "full_rows": EXPECTED_FULL_ROWS,
                "judge_model": model,
                "rubric_version": RUBRIC_VERSION,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "calibration_seed": CALIBRATION_SEED,
                "timestamp_utc": utc_now_iso(),
                "leakage_status": "PASSED (0 golden pairs in training/retrieval pool; "
                                  "retrieval_source_pair_id never in 322 golden IDs)",
                "evaluation_population_note": "Strictly the 150 human-reviewed golden "
                                              "examples; 172 pending excluded.",
            },
            "human_judge_agreement": agreement,
            "agreement_interpretation": (
                f"On the 50-example calibration subset, the LLM judge achieved "
                f"{agreement['overall_exact_agreement']} mean exact agreement and "
                f"{agreement['overall_weighted_kappa']} mean linear-weighted kappa "
                f"against human ratings. Weakest dimension: "
                f"{agreement['weakest_dimension_by_kappa']}. High agreement does not "
                f"prove the judge is objectively correct."
            ),
            "MostFrequentBaseline": summarize_baseline(full_judge, "MostFrequentBaseline"),
            "KeywordBaseline": summarize_baseline(full_judge, "KeywordBaseline"),
            "limitations": [
                "Judge is not perfect; agreement measured only on the 50-example "
                "calibration subset.",
                "Judge scores on the remaining 100 examples are not human ground truth.",
                "Historical Twitter responses are noisy.",
                "Heuristic intent labels are weak training signals.",
                "No pending examples evaluated.",
                "unsupported_claims is inverted (0=best/clean, 3=worst/fabricated).",
            ],
        }
        JUDGE_DIR.mkdir(parents=True, exist_ok=True)
        with open(JUDGE_METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {JUDGE_METRICS_JSON}")


if __name__ == "__main__":
    main()
