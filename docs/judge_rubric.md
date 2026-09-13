# Phase 4 — LLM-as-Judge Reply-Quality Rubric (v1)

Reply length / non-empty rate / keyword overlap (Phase 3) cannot tell whether a
generated response is actually useful and grounded in historical SpotifyCares
behavior. Phase 4 adds semantic evaluation with an LLM judge plus measured
judge-human agreement.

## Evaluation data

- Population: EXACTLY the 150 reviewed golden examples. The 172 pending are
  never evaluated, never retrieved from, never sampled.
- Per example the judge sees: `customer_message`, `context_before_customer`,
  `generated_reply` (from the leakage-free baseline), `retrieved_historical_response`
  (the historical SpotifyCares response actually used, from the 42,770-pair
  leakage-free pool), `predicted_intent`, and the golden `reference_response`
  as context only.
- `retrieval_source_pair_id` is recorded per row and runtime-asserted to never
  belong to the 322 golden pair IDs.

## Rubric (bounded 0–3)

| Dimension | 0 | 1 | 2 | 3 |
|:---|:---|:---|:---|:---|
| Relevance | Off-topic/wrong issue | Partially on-topic | Acceptable, addresses issue | Strong, directly on-point |
| Helpfulness / Actionability | No usable guidance | Vague guidance | Acceptable next steps/explanation | Strong, concrete resolution path |
| Groundedness | Contradicts evidence | Mostly ungrounded | Acceptable, consistent with evidence | Strong, clearly supported |
| Appropriateness | Rude/unsafe/off-brand | Weak tone | Acceptable support tone | Strong, professional + empathetic |
| Unsupported claims (INVERTED) | — | — | — | — |
| ↳ | 0 = no meaningful unsupported claim (BEST) | 1 = minor | 2 = significant | 3 = severe/fabricated (WORST) |
| Overall | Poor/unacceptable | Weak | Acceptable | Strong |
| Confidence | Low | — | — | High |

Each dimension returns `{score, short reason}`; the judge also returns
`overall_score`, `confidence`, and one concise evidence-based `reason`.
No chain-of-thought is exposed.

## Human calibration

- Deterministic 50-example subset of the 150 reviewed set (seed 42),
  i.e. 100 annotation rows (50 × 2 baselines) in `data/judge/human_calibration.csv`.
- The human scores the GENERATED REPLY (not the golden reference) on the same
  0–3 scales, with the inverted unsupported_claims interpretation above.
- Workflow: `scripts/annotate_judge_calibration.py` (resume-safe, never
  overwrites completed rows; `--status` for validation counts). All 100 rows must be
  `reviewed` before agreement. Optional `--suggest` shows LLM scores ONLY as
  "AI SUGGESTION — NOT HUMAN LABEL", stored solely on explicit per-field human
  confirmation (tracked in `ai_suggestion_shown`; anchoring effect disclaimed).
- Agreement: exact agreement rate + linear-weighted Cohen's kappa per
  dimension (see `src/judge.py`). Report as "On the 50-example calibration
  subset, the LLM judge achieved X exact agreement and Y weighted kappa
  against human ratings." High agreement ≠ objective correctness.

## Full evaluation

- Only after calibration: judge runs on all 300 rows (150 × 2 baselines),
  cached in `results/judge_outputs.jsonl`, summarized in
  `results/judge_metrics.json` with per-baseline means + distributions.
- Calibration agreement (human+judge, n=50) is reported separately from full
  judge-only results (n=150). Judge scores on the 100 non-calibration examples
  are not human ground truth.

## Reproducibility

- `JUDGE_MODEL` env-configurable (default `gpt-4o-mini`); `OPENAI_API_KEY`
  required — code fails clearly without it and never fabricates scores.
- Recorded per run: model, `judge-prompt-v1`, rubric `v1`, seed 42,
  populations (150/50), timestamp, leakage status.
