"""Evaluate the AI Support Agent on the 150 reviewed golden examples.

Computes (deterministic, no API needed):
  - intent accuracy / macro F1 / per-class F1 (same metric as baselines)
  - retrieval stats: success rate, mean top-1 similarity, empty-evidence rate,
    mean evidence count, DM-redirect evidence rate
  - escalation rate + decision distribution
  - compares MostFrequentBaseline / KeywordBaseline / SupportAgent

Optional --judge: LLM-judge agent replies for reply-quality comparison.
Requires HUMAN_REVIEWED=100/100 calibration AND OPENAI_API_KEY; otherwise
STOPs with a clear message (never fabricates).

Outputs:
  results/agent_outputs.jsonl  (one record per example, cached, resumable)
  results/agent_metrics.json   (non-judge metrics; judge section only with --judge)

Usage:
    python scripts/evaluate_agent.py
    python scripts/evaluate_agent.py --judge
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import RETRIEVAL_TOP_K, SupportAgent  # noqa: E402
from src.evaluate import compute_intent_metrics, load_data, prepare_leakage_free_split  # noqa: E402
from src.judge import EVALUATION_SIZE  # noqa: E402

RESULTS_DIR = ROOT / "results"
AGENT_OUTPUTS_JSONL = RESULTS_DIR / "agent_outputs.jsonl"
AGENT_METRICS_JSON = RESULTS_DIR / "agent_metrics.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true",
                    help="Also LLM-judge agent replies (needs 100/100 calibration + API key).")
    args = ap.parse_args()

    golden, pairs = load_data()
    train_df, test_df, _, v = prepare_leakage_free_split(golden, pairs)
    assert len(test_df) == EVALUATION_SIZE == 150
    golden_ids = set(golden["pair_id"].astype(str))

    agent = SupportAgent(train_df, golden_pair_ids=golden_ids, top_k=RETRIEVAL_TOP_K)

    # Resume from cache.
    done: Dict[str, dict] = {}
    if AGENT_OUTPUTS_JSONL.exists():
        with open(AGENT_OUTPUTS_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done[obj["pair_id"]] = obj
    print(f"[agent] cached outputs: {len(done)}/150")

    y_true, y_pred = [], []
    with open(AGENT_OUTPUTS_JSONL, "a", encoding="utf-8") as f:
        for _, row in test_df.iterrows():
            pid = str(row["pair_id"])
            y_true.append(str(row["human_primary_intent"]))
            if pid in done:
                y_pred.append(done[pid]["predicted_intent"])
                continue
            ctx = "" if pd.isna(row["context_before_customer"]) else str(row["context_before_customer"])
            out = agent.handle(str(row["customer_message"]), ctx)
            rec = {"pair_id": pid, "human_primary_intent": str(row["human_primary_intent"]),
                   "reference_response": str(row["brand_response"]), **out}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[pid] = rec
            y_pred.append(out["predicted_intent"])
    assert len(done) == 150 and len(y_pred) == 150

    records = [done[str(pid)] for pid in test_df["pair_id"].astype(str)]
    intent_metrics = compute_intent_metrics(y_true, y_pred)
    top1 = [r["retrieved_evidence"][0]["similarity"] if r["retrieved_evidence"] else 0.0
            for r in records]
    empty_n = sum(1 for r in records if not r["retrieved_evidence"])
    esc_n = sum(1 for r in records if r["escalation"]["decision"] == "escalate")
    dm_hits = sum(1 for r in records
                  if any(e["has_dm_redirect"] for e in r["retrieved_evidence"]))
    methods = pd.Series([r["classification_method"] for r in records]).value_counts().to_dict()

    metrics = {
        "metadata": {
            "evaluation_population": 150,
            "training_retrieval_count": v["training_count"],
            "overlap_after_exclusion": v["overlap_after"],
            "top_k": RETRIEVAL_TOP_K,
            "response_mode": "extractive (top-1 historical response; no hallucination by construction)",
            "escalation_note": "Policy outputs only; no escalation accuracy (no human escalation labels).",
        },
        "intent_classification": {
            "accuracy": intent_metrics["accuracy"],
            "macro_f1": intent_metrics["macro_f1"],
            "per_class_f1": {k: vv["f1"] for k, vv in intent_metrics["per_class"].items()},
            "classification_methods": methods,
        },
        "retrieval": {
            "retrieval_success_rate": round((150 - empty_n) / 150, 4),
            "empty_evidence_rate": round(empty_n / 150, 4),
            "mean_top1_similarity": round(float(sum(top1) / 150), 4),
            "mean_evidence_count": round(float(sum(len(r["retrieved_evidence"]) for r in records) / 150), 4),
            "dm_redirect_evidence_rate": round(dm_hits / 150, 4),
        },
        "escalation": {
            "escalation_rate": round(esc_n / 150, 4),
            "auto_handle_rate": round((150 - esc_n) / 150, 4),
        },
        "reply_quality_judge": None,  # filled only by --judge after calibration+API
    }

    if args.judge:
        from src.judge import HUMAN_CALIBRATION_CSV
        h = pd.read_csv(HUMAN_CALIBRATION_CSV, dtype="object").fillna("")
        rev = int((h["annotation_status"].str.lower() == "reviewed").sum())
        if rev != 100:
            raise SystemExit(f"STOP: human calibration {rev}/100. Complete annotation first.")
        from src.judge import (JUDGE_PROMPT_VERSION, RUBRIC_VERSION, append_cached_output,
                               build_judge_prompt, call_llm_judge, get_judge_config,
                               load_cached_outputs, utc_now_iso, validate_judge_score)
        cfg = get_judge_config()
        cache = load_cached_outputs()
        scored = []
        for r in records:
            key = f"SupportAgent::{r['pair_id']}"
            if key in cache:
                scored.append(cache[key])
                continue
            ev0 = r["retrieved_evidence"][0] if r["retrieved_evidence"] else None
            prompt = build_judge_prompt(
                customer_message=str(test_df.set_index("pair_id").loc[r["pair_id"], "customer_message"]),
                context_before_customer="",
                generated_reply=r["generated_reply"],
                retrieved_evidence=ev0["brand_response"] if ev0 else "(no evidence)",
                predicted_intent=r["predicted_intent"])
            s = call_llm_judge(prompt, model=cfg["model"])
            validate_judge_score(s)
            rec = {"pair_id": r["pair_id"], "baseline": "SupportAgent",
                   "judge_model": cfg["model"], "rubric_version": RUBRIC_VERSION,
                   "prompt_version": JUDGE_PROMPT_VERSION, "timestamp_utc": utc_now_iso(), **s}
            append_cached_output(rec)
            scored.append(rec)
        import numpy as np
        dims = ["relevance", "helpfulness", "groundedness", "appropriateness",
                "unsupported_claims", "overall_score"]
        metrics["reply_quality_judge"] = {
            d: {"mean": round(float(np.mean([s[d] for s in scored])), 4),
                "distribution": {str(k): sum(1 for s in scored if s[d] == k) for k in range(4)}}
            for d in dims
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(AGENT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "intent_classification"}, indent=2))
    print(f"Intent accuracy={metrics['intent_classification']['accuracy']} "
          f"macroF1={metrics['intent_classification']['macro_f1']}")
    print(f"Results saved to {AGENT_METRICS_JSON}")


if __name__ == "__main__":
    main()
