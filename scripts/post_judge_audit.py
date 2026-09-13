"""Full post-judge artifact audit."""
import csv, json
from pathlib import Path

ROOT = Path(".")

# 1. Human calibration status
cal = list(csv.DictReader(open("data/judge/human_calibration.csv", encoding="utf-8")))
reviewed = [r for r in cal if r.get("annotation_status") == "reviewed"]
pending   = [r for r in cal if r.get("annotation_status") == "pending"]
print(f"human_calibration.csv: {len(cal)} total | reviewed={len(reviewed)} | pending={len(pending)}")

# 2. Judge inputs
ji = [json.loads(l) for l in open("data/judge/judge_inputs_full150.jsonl", encoding="utf-8") if l.strip()]
print(f"judge_inputs_full150.jsonl: {len(ji)} rows ready for judging")

# 3. Results inventory
print("\nresults/ inventory:")
for f in sorted(Path("results").iterdir()):
    print(f"  {f.name}: {f.stat().st_size:,} bytes")

# 4. Check for judge_outputs.jsonl and judge_metrics.json
jo_path = ROOT / "results" / "judge_outputs.jsonl"
jm_path = ROOT / "results" / "judge_metrics.json"
print(f"\njudge_outputs.jsonl exists: {jo_path.exists()}")
print(f"judge_metrics.json exists:  {jm_path.exists()}")

# 5. Agent metrics
m = json.load(open("results/agent_metrics.json"))
print(f"\nagent_metrics.json:")
print(f"  evaluation_population:  {m['metadata']['evaluation_population']}")
print(f"  overlap_after_exclusion: {m['metadata']['overlap_after_exclusion']}")
print(f"  intent accuracy:        {m['intent_classification']['accuracy']}")
print(f"  intent macro_f1:        {m['intent_classification']['macro_f1']}")
print(f"  retrieval_success_rate: {m['retrieval']['retrieval_success_rate']}")
print(f"  mean_top1_similarity:   {m['retrieval']['mean_top1_similarity']}")
print(f"  escalation_rate:        {m['escalation']['escalation_rate']}")
print(f"  reply_quality_judge:    {m['reply_quality_judge']}")

# 6. Baseline metrics
b = json.load(open("results/baseline_metrics.json"))
print(f"\nbaseline_metrics.json:")
mf = b["baselines"]["MostFrequentBaseline"]["intent_classification"]
kw = b["baselines"]["KeywordBaseline"]["intent_classification"]
print(f"  MostFrequent accuracy={mf['accuracy']}  macro_f1={mf['macro_f1']}")
print(f"  Keyword      accuracy={kw['accuracy']}  macro_f1={kw['macro_f1']}")
print(f"  leakage_assertion: {b['dataset']['leakage_assertion_result']}")
print(f"  training_pool:     {b['dataset']['training_retrieval_count']}")
print(f"  golden_reviewed:   {b['dataset']['golden_reviewed']}")

# 7. Machine metrics (dev only -- document that it exists but is NOT human labels)
mm = json.load(open("results/machine_metrics.json"))
print(f"\nmachine_metrics.json (NOT human labels, NOT judge output):")
print(f"  rows:   {mm.get('metadata', {}).get('rows_evaluated', '?')}")
print(f"  source: {mm.get('metadata', {}).get('label_source', '?')}")
print(f"  agreement_computed: {mm.get('agreement') is not None}")

# 8. Verify agent_outputs.jsonl
rows = [json.loads(l) for l in open("results/agent_outputs.jsonl", encoding="utf-8") if l.strip()]
print(f"\nagent_outputs.jsonl: {len(rows)} rows")
escalated = sum(1 for r in rows if r["escalation"]["decision"] == "escalate")
print(f"  escalated={escalated}  auto_handle={len(rows)-escalated}  rate={escalated/len(rows):.4f}")

print("\nALL ARTIFACT CHECKS COMPLETE.")
