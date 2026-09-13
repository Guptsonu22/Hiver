"""Phase 4 setup: build leakage-free judge inputs for the 150 reviewed examples.

Generates (deterministically, no LLM calls, no human labels fabricated):
  data/judge/judge_inputs_full150.jsonl  — 300 rows (150 reviewed x 2 baselines)
      with generated_reply + retrieval_source_pair_id + retrieved evidence.
  data/judge/calibration_sample.csv      — deterministic 50-example subset (seed 42).
  data/judge/human_calibration.csv       — annotation template (all 'pending',
      human score columns EMPTY). A human must fill these via
      scripts/annotate_judge_calibration.py. Nothing is marked reviewed here.

Leakage guarantees:
  - Training/retrieval pool excludes all 322 golden pair IDs.
  - Every retrieval_source_pair_id is asserted NOT in golden IDs.
  - Only reviewed (150) examples used; pending (172) never included.

Usage:
    python scripts/build_judge_inputs.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import KeywordBaseline, MostFrequentBaseline  # noqa: E402
from src.evaluate import prepare_leakage_free_split, load_data  # noqa: E402
from src.judge import (  # noqa: E402
    CALIBRATION_CSV,
    CALIBRATION_SEED,
    CALIBRATION_SIZE,
    HUMAN_CALIBRATION_CSV,
    JUDGE_DIR,
    JUDGE_INPUTS_JSONL,
    build_evaluation_frame,
    sample_calibration_set,
)

HUMAN_SCORE_COLS = [
    "relevance",
    "helpfulness",
    "groundedness",
    "appropriateness",
    "unsupported_claims",
    "overall_score",
]


def main() -> None:
    golden, pairs = load_data()
    train_df, test_df, pending_df, verification = prepare_leakage_free_split(golden, pairs)
    golden_ids = set(golden["pair_id"].astype(str))
    assert len(test_df) == 150, f"Expected 150 reviewed, got {len(test_df)}"

    mf = MostFrequentBaseline(training_pool=train_df, golden_pair_ids=golden_ids)
    kw = KeywordBaseline(training_pool=train_df, golden_pair_ids=golden_ids)

    mf_frame = build_evaluation_frame(test_df, train_df, golden_ids, "MostFrequentBaseline", mf)
    kw_frame = build_evaluation_frame(test_df, train_df, golden_ids, "KeywordBaseline", kw)
    full = pd.concat([mf_frame, kw_frame], ignore_index=True)
    assert len(full) == 300, f"Expected 300 rows (150x2), got {len(full)}"
    overlap = set(full["retrieval_source_pair_id"].astype(str)) & golden_ids
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} retrieval sources in golden set!"

    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    # Refuse to silently overwrite an existing inputs file with different content.
    full_sorted = full.sort_values(["baseline", "pair_id"]).reset_index(drop=True)
    full_sorted.to_json(JUDGE_INPUTS_JSONL, orient="records", lines=True, force_ascii=False)
    print(f"[build] wrote {JUDGE_INPUTS_JSONL} ({len(full_sorted)} rows)")

    # Deterministic 50-example calibration subset from the 150 reviewed.
    calib = sample_calibration_set(test_df, n=CALIBRATION_SIZE, seed=CALIBRATION_SEED)
    assert len(calib) == 50
    assert set(calib["pair_id"].astype(str)) <= set(test_df["pair_id"].astype(str))
    assert (calib["annotation_status"] == "reviewed").all()
    calib[["pair_id", "human_primary_intent", "annotation_status"]].to_csv(CALIBRATION_CSV, index=False)
    print(f"[build] wrote {CALIBRATION_CSV} (50 rows, seed={CALIBRATION_SEED})")

    # Human annotation template: one row per (calibration example x baseline) = 100 rows.
    # Scores left EMPTY; annotation_status='pending'. Never auto-mark reviewed.
    calib_ids = set(calib["pair_id"].astype(str))
    template = full_sorted[full_sorted["pair_id"].astype(str).isin(calib_ids)].copy()
    assert len(template) == 100, f"Expected 100 calibration rows (50x2), got {len(template)}"
    for col in HUMAN_SCORE_COLS:
        template[col + "_human"] = ""
    template["human_note"] = ""
    template["annotation_status"] = "pending"
    keep = (
        ["pair_id", "baseline", "customer_message", "context_before_customer",
         "human_primary_intent", "predicted_intent", "generated_reply",
         "retrieval_source_pair_id", "retrieved_historical_response",
         "reference_response"]
        + [c + "_human" for c in HUMAN_SCORE_COLS]
        + ["human_note", "annotation_status"]
    )
    template = template[keep].sort_values(["pair_id", "baseline"]).reset_index(drop=True)
    if HUMAN_CALIBRATION_CSV.exists():
        print(f"[build] {HUMAN_CALIBRATION_CSV} already exists — NOT overwriting "
              f"(resume with scripts/annotate_judge_calibration.py).")
    else:
        template.to_csv(HUMAN_CALIBRATION_CSV, index=False, encoding="utf-8")
        print(f"[build] wrote {HUMAN_CALIBRATION_CSV} (100 rows pending, scores EMPTY)")

    print("CALIBRATION_TOTAL = 50 examples x 2 baselines = 100 rows")
    print("HUMAN_REVIEWED = 0 (awaiting real human annotation)")
    print("HUMAN_PENDING = 100")
    print(f"Verification: overlap_after={verification['overlap_after']}, "
          f"training={verification['training_count']}, test={verification['test_count']}")


if __name__ == "__main__":
    main()
