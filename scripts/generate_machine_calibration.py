"""Generate the MACHINE (non-human) calibration artifact.

Reads the 100 calibration rows (50 examples x 2 baselines) from the leakage-free
judge inputs and scores each row. For every row it FIRST tries the real LLM judge
(existing rubric logic); on ANY failure (missing key, HTTP 401, malformed output)
it falls back to the explicitly named deterministic heuristic
(``deterministic_heuristic_v1`` from src/machine_score.py). Per-row ``source``
records which path produced the scores.

HARD GUARANTEES:
  - NEVER writes to data/judge/human_calibration.csv (byte-identical before/after).
  - Output columns use the ``machine_*`` prefix; there are NO human_* score
    columns and NO human_confirmed/human-edited markers anywhere in the output.
  - Output: data/judge/machine_calibration_100.csv (100 rows).
  - Machine scores are NEVER human labels; NEVER feed them to human-vs-judge
    agreement. The official workflow (scripts/run_judge.py without flags) still
    requires genuine 100/100 human calibration.

Usage:
    python scripts/generate_machine_calibration.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.judge import (  # noqa: E402
    CALIBRATION_SEED,
    EVALUATION_SIZE,
    HUMAN_CALIBRATION_CSV,
    JUDGE_DIR,
    build_judge_prompt,
    call_llm_judge,
    get_judge_config,
    utc_now_iso,
    validate_judge_score,
)
from src.machine_score import METHOD, SOURCE_LOCAL, heuristic_reply_scores  # noqa: E402

GOLDEN_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"
CALIBRATION_CSV = JUDGE_DIR / "calibration_sample.csv"
JUDGE_INPUTS_JSONL = JUDGE_DIR / "judge_inputs_full150.jsonl"
MACHINE_CSV = JUDGE_DIR / "machine_calibration_100.csv"
MACHINE_LLM_CACHE = JUDGE_DIR / "machine_llm_cache.jsonl"

JUDGE_TO_MACHINE = {
    "relevance": "machine_relevance",
    "helpfulness": "machine_helpfulness",
    "groundedness": "machine_groundedness",
    "appropriateness": "machine_appropriateness",
    "unsupported_claims": "machine_unsupported_claims",
    "overall_score": "machine_overall",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_machine_llm_cache() -> dict:
    cache = {}
    if MACHINE_LLM_CACHE.exists():
        for line in open(MACHINE_LLM_CACHE, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cache[f"{obj['baseline']}::{obj['pair_id']}"] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def _try_llm_scores(row, cache: dict):
    """Attempt the real LLM judge. Returns (scores_dict, source) or raises."""
    key = f"{row['baseline']}::{row['pair_id']}"
    if key in cache:
        hit = cache[key]
        return hit["scores"], hit["source"]
    cfg = get_judge_config()  # raises without key
    prompt = build_judge_prompt(
        customer_message=str(row["customer_message"]),
        context_before_customer=str(row.get("context_before_customer", "") or ""),
        generated_reply=str(row["generated_reply"]),
        retrieved_evidence=str(row["retrieved_historical_response"]),
        reference_response=str(row.get("reference_response", "") or ""),
        predicted_intent=str(row.get("predicted_intent", "") or ""),
    )
    score = validate_judge_score(call_llm_judge(prompt, model=cfg["model"]))
    mapped = {JUDGE_TO_MACHINE[k]: int(score[k]) for k in JUDGE_TO_MACHINE}
    source = f"llm:{cfg['model']} (real API output; MACHINE-generated, NOT human)"
    MACHINE_LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MACHINE_LLM_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"baseline": str(row["baseline"]), "pair_id": str(row["pair_id"]),
                            "model": cfg["model"], "scores": mapped, "source": source,
                            "reason": score.get("reason", "")[:300],
                            "timestamp_utc": utc_now_iso()}, ensure_ascii=False) + "\n")
    return mapped, source


def main() -> None:
    human_before = _sha256(HUMAN_CALIBRATION_CSV)

    golden = pd.read_csv(GOLDEN_CSV, dtype={"pair_id": str})
    golden_ids = set(golden["pair_id"])
    reviewed_ids = set(golden[golden["annotation_status"] == "reviewed"]["pair_id"])
    sample = pd.read_csv(CALIBRATION_CSV, dtype={"pair_id": str})
    assert len(sample) == 50, f"Expected 50 calibration examples, got {len(sample)}"
    sample_ids = set(sample["pair_id"])
    assert sample_ids <= reviewed_ids, "Calibration sample must be subset of reviewed 150."

    full = pd.read_json(JUDGE_INPUTS_JSONL, lines=True, dtype={"pair_id": str})
    assert len(full) == 300, f"Expected 300 judge inputs, got {len(full)}"
    calib = full[full["pair_id"].isin(sample_ids)].copy()
    assert len(calib) == 100, f"Expected 100 calibration rows, got {len(calib)}"
    overlap = set(calib["retrieval_source_pair_id"].astype(str)) & golden_ids
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} retrieval sources are golden IDs!"

    llm_cache = _load_machine_llm_cache()
    rows, n_llm, n_heur = [], 0, 0
    for _, r in calib.sort_values(["pair_id", "baseline"]).iterrows():
        try:
            scores, source = _try_llm_scores(r, llm_cache)
            reason = llm_cache.get(f"{r['baseline']}::{r['pair_id']}", {}).get("reason", "")
            n_llm += 1
        except Exception as e:
            print(f"  [row {r['pair_id']}/{r['baseline']}: LLM unavailable ({e}) -> heuristic]")
            h = heuristic_reply_scores(str(r["customer_message"]), str(r["generated_reply"]),
                                       str(r["retrieved_historical_response"]))
            scores = {k: h[k] for k in JUDGE_TO_MACHINE.values()}
            reason = h["machine_reason"]
            source = SOURCE_LOCAL
            n_heur += 1
        rows.append({
            "pair_id": str(r["pair_id"]), "baseline": str(r["baseline"]),
            "predicted_intent": str(r["predicted_intent"]),
            "retrieval_source_pair_id": str(r["retrieval_source_pair_id"]),
            **scores, "machine_reason": reason, "method": METHOD,
            "source": source, "generated_utc": utc_now_iso(),
        })

    out = pd.DataFrame(rows)
    for c in ["machine_relevance", "machine_helpfulness", "machine_groundedness",
              "machine_appropriateness", "machine_unsupported_claims", "machine_overall"]:
        assert out[c].between(0, 3).all(), f"Column {c} out of range!"
    assert not any(c.startswith("human_") for c in out.columns), "Human columns forbidden!"
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(MACHINE_CSV, index=False, encoding="utf-8")

    human_after = _sha256(HUMAN_CALIBRATION_CSV)
    assert human_before == human_after, "HUMAN CALIBRATION FILE WAS MODIFIED - ABORT!"

    print(f"MACHINE calibration: {len(out)} rows ({n_llm} real-LLM, {n_heur} heuristic)")
    print(f"Wrote {MACHINE_CSV}")
    print("human_calibration.csv byte-identical: TRUE (untouched)")
    print("NOTE: these are MACHINE scores, NOT human labels. No agreement computed.")


if __name__ == "__main__":
    main()
