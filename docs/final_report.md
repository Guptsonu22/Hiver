# Hiver SDE Intern — Final Report Content (SpotifyCares Support Intelligence)

Every number below is an actual verified output. Judge-dependent cells are marked
PENDING (calibration 0/100, no API key, zero API calls made) — never filled with
estimates. Full evidence: `docs/judge_rubric.md`, `docs/failure_modes.md`,
`docs/decision_log.md` (D1–D25).

## 1. Problem framing
Twitter support (@SpotifyCares, 43,092 customer→brand pairs) is high-volume,
repetitive, and safety-sensitive (account takeovers, double-billing). Goal: a
leakage-safe agent that classifies intent, retrieves historical cases, drafts
grounded replies, and escalates transparently — evaluated on human ground truth
with measured judge–human agreement.

## 2. What "good" means
Intent accuracy/macro-F1 on 150 human-reviewed examples; reply quality on relevance,
helpfulness, groundedness, appropriateness, unsupported-claims (LLM judge PLUS human
agreement — never length/overlap); escalation that never silently auto-handles
security/financial cases; zero golden leakage; every claim reproducible from the repo.

## 3. Dataset
2.8M raw tweets → 91,889-tweet SpotifyCares ecosystem → 28,280 conversation trees →
43,092 customer→brand pairs → **leakage-free pool 42,770** (322/322 golden IDs excluded,
overlap_after = 0). English-majority (87.74%); 25.97% DM redirects; 8,624 heuristic
hard cases. Deterministic seeds (42) throughout.

## 4. Taxonomy
10 intents (8 technical + `unclear_insufficient_context` + `other_miscellaneous`),
keyword patterns + business-priority tie-breaks (`src/intents/taxonomy.py`). Training
labels are **weak/heuristic** (D11) — the supervision ceiling for any learner (see §12 FM3).

## 5. Golden evaluation set
322 candidates; **150 human-reviewed = the ONLY ground truth**; 172 pending excluded
from training AND evaluation. Frozen; untouched by Phase 4 (322/150/172 re-verified).

## 6. Baselines (150 reviewed, verified Phase 3)
MostFrequent (`other_miscellaneous`): acc **0.0467**, macro F1 **0.0089**.
Keyword taxonomy + per-intent modal reply: acc **0.5800**, macro F1 **0.5914**.

## 7. Agent architecture (`src/agent.py`, unchanged)
TF-IDF kNN classifier (k=5, similarity-weighted, keyword fallback) → TF-IDF top-3
retrieval → extractive top-1 historical reply (hallucination impossible by
construction; LLM path opt-in, fail-closed) → rule escalation (security/financial,
unclear, top-sim < 0.15, ambiguity margin < 0.05) → structured output with per-call
leakage assertion. Retrieval vectorized via `lexsort` (D23).

## 8. Evaluation methodology
Intent: accuracy/macro-F1/per-class F1 (same function for all systems). Reply quality:
LLM judge (rubric v1, 0–3, `unsupported_claims` inverted) over leakage-free generated
replies + retrieved evidence, cached in `results/judge_outputs.jsonl`. Agreement: exact
rate + linear-weighted kappa on the 50-example calibration subset ONLY; judge scores on
the other 100 are NOT ground truth. Escalation: rate only (no human escalation labels
exist — accuracy would be circular).

## 9. LLM judge
Rubric v1 + `judge-prompt-v1`, temperature 0, strict JSON schema, env-configured model
(default `gpt-4o-mini`), fail-closed without `OPENAI_API_KEY`, stdlib `urllib` (no SDK
dependency). Status: **PENDING** — 0 calls made. Resume: finish calibration (§10), set a
fresh key, run `scripts/run_judge.py`, then `scripts/evaluate_agent.py --judge`.

## 10. Human agreement
Calibration: deterministic 50 examples × 2 baselines = 100 rows (seed 42, ⊂ reviewed
150, ∩ pending = ∅). Workflow: `scripts/annotate_judge_calibration.py` (resume-safe,
`--status` validation, optional `--suggest` showing labeled suggestions confirmed
per-field, tracked in `ai_suggestion_shown`). Status: **PENDING — 0/100 reviewed.**
Agreement (exact + kappa per dimension, weakest-dimension finding, "agreement ≠
correctness") will be computed from real labels only.

**Machine diagnostic (NOT human calibration, NOT a substitute):** because genuine
calibration stands at 0/100 and the API returned 401, a separately-tracked machine
artifact exists — `data/judge/machine_calibration_100.csv` (100/100 machine rows, all
`deterministic_heuristic_v1`; `human_calibration.csv` verified byte-identical) and
`results/machine_metrics.json` (full-300 heuristic means; both baselines overall ~2.0).
No agreement was or will be computed from machine labels. Details:
`docs/machine_calibration.md`. Limitation disclosed: the assignment's human-label
requirement is UNSATISFIED; all reply-quality and agreement cells below remain PENDING.

## 11. Results (real; reply-quality judge cells PENDING)

| Metric | MostFreq | Keyword | SupportAgent |
|:---|:---:|:---:|:---:|
| Intent Accuracy | 0.0467 | 0.5800 | **0.4867** |
| Intent Macro F1 | 0.0089 | 0.5914 | **0.5086** |
| Reply Relevance | PENDING | PENDING | PENDING |
| Reply Helpfulness | PENDING | PENDING | PENDING |
| Reply Groundedness | PENDING | PENDING | PENDING |
| Reply Appropriateness | PENDING | PENDING | PENDING |
| Unsupported Claims | PENDING | PENDING | PENDING |
| Overall Reply Score | PENDING | PENDING | PENDING |
| Retrieval success / empty rate | — | — | 1.00 / 0.00 |
| Mean top-1 similarity | — | — | 0.5967 |
| Escalation rate | 0.00 | 0.153 | **0.38** |

## 12. Top 5 failure modes (real; `docs/failure_modes.md`)
FM1 follow-ups confidently wrong at sim 0.87–0.97 ("are u serious" → offline,
auto-handled); FM2 short texts collapse to unclear (catalog→unclear ×9; 38/150
escalations); FM3 weak-label noise (sim = 1.000 yet wrong — supervision ceiling);
FM4 non-English → misc, auto-handled; FM5 **security misses** (`pair_883440_883439`,
"someone is in my account" → `other_miscellaneous` → auto_handle; re-verified —
highest severity).

## 13. What is misleading about my headline number?
"Keyword accuracy 0.58" flatters: n = 150 is small and class-imbalanced; weak training
labels cap kNN learning; Twitter text is noisy/short/multilingual; reply quality is
unmeasured until the judge runs (length ≠ usefulness); n = 50 agreement will have wide
CIs (plus possible suggestion-anchoring, audited via `ai_suggestion_shown`); escalation
*rate* is not correctness; ID-level (not semantic) leakage freedom misses near-duplicate
re-tweets; historical replies may be outdated/generic; FM5 proves headline metrics hide
safety-critical misses.

## 14. What was not built
No judge scores/agreement (blocked); no abstractive generation by default; no
multi-label intent; no ANN scale-out; no human escalation labels; no non-English
support; no resolution/outcome modeling (response type ≠ resolution, D6).

## 15. One-week next steps + decision log
1. Finish 100-row calibration + judge run (unblocks §9–§11). 2. Security-first override
+ red-team slice (FM5). 3. Follow-up detector with parent-intent inheritance (FM1).
4. Language gate → escalate non-English (FM4). 5. Duplicate label-cleaning + keyword
cross-check (FM3). Decisions: D1–D12 in `DECISIONS.md`, D13–D25 in
`docs/decision_log.md`. Reproduce: venv + `pip install -r requirements.txt` +
`pytest`; see README for full command map.
