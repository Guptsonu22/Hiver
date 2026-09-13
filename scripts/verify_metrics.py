"""Verify failure modes and metrics are consistent with actual result files."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load agent outputs
rows = []
with open(ROOT / "results" / "agent_outputs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print(f"Agent outputs rows: {len(rows)}")

# Verify each FM pair
checks = {
    "FM5 pair_883440_883439": "pair_883440_883439",
    "FM1 pair_110043_1822595": "pair_110043_1822595",
    "FM3 pair_199121_199122": "pair_199121_199122",
    "FM4 pair_662956_662955": "pair_662956_662955",
    "FM2 pair_785533_785532": "pair_785533_785532",
}
row_index = {r["pair_id"]: r for r in rows}
for label, pid in checks.items():
    if pid in row_index:
        r = row_index[pid]
        print(f"{label}: pred={r['predicted_intent']} esc={r['escalation']}")
    else:
        print(f"{label}: NOT FOUND IN OUTPUTS")

# Verify agent metrics
with open(ROOT / "results" / "agent_metrics.json") as f:
    m = json.load(f)
print("\n--- agent_metrics.json ---")
print(f"accuracy: {m['intent_classification']['accuracy']}")
print(f"macro_f1: {m['intent_classification']['macro_f1']}")
print(f"retrieval_success: {m['retrieval']['retrieval_success_rate']}")
print(f"mean_top1_sim: {m['retrieval']['mean_top1_similarity']}")
print(f"escalation_rate: {m['escalation']['escalation_rate']}")
print(f"reply_quality_judge: {m['reply_quality_judge']}")

# Verify baseline metrics
with open(ROOT / "results" / "baseline_metrics.json") as f:
    b = json.load(f)
print("\n--- baseline_metrics.json ---")
print(f"MostFreq accuracy: {b['baselines']['MostFrequentBaseline']['intent_classification']['accuracy']}")
print(f"Keyword accuracy: {b['baselines']['KeywordBaseline']['intent_classification']['accuracy']}")
print(f"Keyword macro_f1: {b['baselines']['KeywordBaseline']['intent_classification']['macro_f1']}")
print(f"Leakage check: {b['dataset']['leakage_assertion_result']}")
print(f"Pool size: {b['dataset']['training_retrieval_count']}")
print(f"Golden reviewed: {b['dataset']['golden_reviewed']}")
