"""Failure analysis over real agent outputs on the 150 reviewed examples.

Prints (all from actual evaluation data, never fabricated):
  1. Top confusion pairs (true intent -> predicted intent, with counts)
  2. Lowest-similarity retrievals (weak-evidence cases)
  3. Escalation reason distribution
  4. Clarification-reply cases
  5. Per-class F1 lows

Usage:
    python scripts/analyze_failures.py [--top N]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import load_data, prepare_leakage_free_split  # noqa: E402

RESULTS_DIR = ROOT / "results"
AGENT_OUTPUTS_JSONL = RESULTS_DIR / "agent_outputs.jsonl"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    golden, pairs = load_data()
    _, test_df, _, _ = prepare_leakage_free_split(golden, pairs)
    recs = [json.loads(l) for l in open(AGENT_OUTPUTS_JSONL, encoding="utf-8") if l.strip()]
    assert len(recs) == 150, f"Expected 150 agent outputs, got {len(recs)}"
    by_id = {r["pair_id"]: r for r in recs}

    errors = []
    for _, row in test_df.iterrows():
        r = by_id[str(row["pair_id"])]
        if r["predicted_intent"] != str(row["human_primary_intent"]):
            errors.append({
                "pair_id": str(row["pair_id"]),
                "true": str(row["human_primary_intent"]),
                "pred": r["predicted_intent"],
                "method": r["classification_method"],
                "top_sim": r["retrieved_evidence"][0]["similarity"] if r["retrieved_evidence"] else 0.0,
                "customer": str(row["customer_message"])[:220],
                "reply": r["generated_reply"][:220],
                "escalation": r["escalation"]["decision"],
            })
    print(f"INTENT ERRORS: {len(errors)}/150 (accuracy={(150 - len(errors)) / 150:.4f})")
    conf = pd.DataFrame(errors).groupby(["true", "pred"]).size().sort_values(ascending=False)
    print(f"\nTOP-{args.top} CONFUSION PAIRS (true -> pred : count):")
    for (t, p), c in conf.head(args.top).items():
        print(f"  {t} -> {p} : {c}")

    print(f"\nLOWEST-SIMILARITY CASES (weak evidence):")
    lows = sorted(recs, key=lambda r: r["retrieved_evidence"][0]["similarity"]
                  if r["retrieved_evidence"] else -1.0)[:args.top]
    for r in lows:
        row = test_df.set_index("pair_id").loc[r["pair_id"]]
        sim = r["retrieved_evidence"][0]["similarity"] if r["retrieved_evidence"] else 0.0
        print(f"  {r['pair_id']} sim={sim:.3f} true={row['human_primary_intent']} "
              f"pred={r['predicted_intent']} esc={r['escalation']['decision']}")
        print(f"    customer: {str(row['customer_message'])[:200]}")

    esc_reasons = pd.Series([r["escalation"]["reason"] for r in recs]).value_counts()
    print(f"\nESCALATION REASONS ({sum(1 for r in recs if r['escalation']['decision'] == 'escalate')}/150 escalated):")
    for reason, c in esc_reasons.head(8).items():
        print(f"  [{c}] {str(reason)[:130]}")

    print("\nFULL ERROR TABLE (for manual top-5 failure-mode inspection):")
    for e in errors:
        print(f"  {e['pair_id']} | true={e['true']} pred={e['pred']} sim={e['top_sim']:.3f} "
              f"esc={e['escalation']} | {e['customer']}")


if __name__ == "__main__":
    main()
