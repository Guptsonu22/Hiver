# Decision Log — Phase 3 / Phase 4 / Agent (D13–D24)

> Format: Decision → Why → Alternative considered → Trade-off.
> Phase 2 decisions D1–D12 live in `DECISIONS.md`. Only decisions actually made
> in this project are listed; nothing is reconstructed or invented.

## D13 — Canonical `pair_id` string match as the leakage identifier
**Decision:** Use exact `pair_id` string equality (e.g. `pair_43459_43458`) for all
leakage checks, verified as 322/322 golden IDs present in `spotify_pairs.parquet`.
**Why:** Tweet IDs are unstable across re-downloads; the constructed `pair_id` is the
only stable join key between golden candidates and processed pairs.
**Alternative:** Fuzzy text matching on customer messages.
**Trade-off:** Exact matching misses near-duplicate wording (same complaint re-tweeted),
so leakage freedom is ID-level, not semantic-level — stated as a limitation.

## D14 — 150-reviewed-only evaluation; pending 172 excluded from train AND test
**Decision:** All 322 golden IDs excluded from the retrieval pool; metrics computed on
exactly the 150 `reviewed` rows; the 172 `pending` rows appear nowhere.
**Why:** Pending rows have no human labels — including them in training risks
evaluating on seen data later; including them in test is unevaluable.
**Alternative:** Train on pending (they are "unlabeled anyway").
**Trade-off:** Training pool shrinks 43,092 → 42,770 (−0.75%); evaluation n=150 is small
(explicitly disclosed in the report).

## D15 — Bounded 0–3 rubric with INVERTED `unsupported_claims`
**Decision:** Five dimensions on 0–3; the first four are higher-is-better while
`unsupported_claims` is lower-is-better (0=clean … 3=fabricated).
**Why:** Keeps every scale small and annotator-friendly while making the safety
dimension read naturally ("0 unsupported claims" = clean).
**Alternative:** Uniform higher-is-better ("groundedness" only) or 1–5 Likert.
**Trade-off:** Inverted polarity is a known annotator-confusion risk — mitigated by
labeling every UI prompt and column header with the direction.

## D16 — Deterministic 50-example calibration subset (seed 42)
**Decision:** `sample(n=50, random_state=42)` from the 150 reviewed; 100 annotation
rows (50 × 2 baselines); membership assertions (⊂ reviewed, ∩ pending = ∅).
**Why:** Full-150 human labeling is expensive; 50 gives a measurable agreement signal
while keeping the remaining 100 judge-only (honestly labeled as non-ground-truth).
**Alternative:** Human-label all 150, or a smaller n=20 slice.
**Trade-off:** Agreement CIs are wide at n=50; weakest-dimension findings are indicative,
not definitive.

## D17 — Fail-closed judge: no key → RuntimeError, never a heuristic stand-in
**Decision:** `get_judge_config()` raises an actionable error without `OPENAI_API_KEY`;
no fake/local "LLM judge" scores exist anywhere in the pipeline.
**Why:** A heuristic masquerading as an LLM judge would silently invalidate the entire
Phase 4 claim ("judge-human agreement").
**Alternative:** Local fallback scorer when the API is unreachable.
**Trade-off:** Evaluation halts without credentials (as it currently does) — by design.

## D18 — stdlib `urllib` for judge API calls (no new dependency)
**Decision:** Chat-completions calls use `urllib` + `response_format: json_object` +
temperature 0, with strict JSON schema validation on every response.
**Why:** Avoids adding an OpenAI SDK dependency to a project that otherwise needs only
pandas/sklearn; strict parsing turns malformed outputs into loud errors, not silent data.
**Alternative:** `openai` Python package with structured outputs.
**Trade-off:** Less ergonomic retries/streaming; acceptable for a 300-row batch job with caching.

## D19 — Exact agreement + linear-weighted Cohen's kappa (hand-rolled, no sklearn)
**Decision:** Report both per dimension; kappa uses linear disagreement weights for the
ordinal 0–3 scale, implemented in numpy (no sklearn dependency in `src/judge.py`).
**Why:** Exact agreement is interpretable; linear-weighted kappa credits close misses
(1 vs 2) over far misses (0 vs 3), which matters on ordinal scales.
**Alternative:** Quadratic kappa or Krippendorff's alpha.
**Trade-off:** Linear vs quadratic weighting can reorder "weakest dimension" findings —
weighting choice is documented alongside the numbers.

## D20 — Agent classifier: TF-IDF kNN over weak labels (+ keyword fallback), not keyword reuse
**Decision:** kNN (k=5, similarity-weighted vote, taxonomy-order tie-break) over the
42,770 pool; zero-similarity queries fall back to `classify_customer_intent`.
**Why:** Reusing the keyword classifier would make the "agent" a relabeled baseline with
identical intent accuracy (0.58) — no new information. kNN gives a genuinely different,
reproducible operating point (0.4867 / 0.5086).
**Alternative:** Fine-tuned classifier or LLM zero-shot labeling of 42k rows.
**Trade-off:** kNN inherits weak-label noise (failure mode FM3); accuracy is lower than
the keyword baseline — reported plainly instead of tuned away.

## D21 — Extractive top-1 reply as the default generator (no hallucination by construction)
**Decision:** Default `use_llm=False`: the reply IS the top-1 retrieved historical
SpotifyCares response; low-confidence queries get a fixed clarification template.
**Why:** An abstractive default would need judge scores to prove it doesn't hallucinate —
scores we cannot produce without calibration+API. Extractive grounding makes
`unsupported_claims ≈ 0` a structural property, not a claim.
**Alternative:** LLM-first generation with evidence grounding.
**Trade-off:** Replies can be stale/generic (e.g. answering a follow-up with a stranger's
response — FM1);_OPT-IN LLM path exists (`use_llm=True`, fail-closed) for later comparison.

## D22 — Escalation as policy outputs; no "escalation accuracy" metric
**Decision:** Transparent rules (security/financial → escalate; unclear → escalate;
top-sim < 0.15 → escalate; ambiguity margin < 0.05 → escalate); report only
escalation *rate* (0.38), never accuracy.
**Why:** No human escalation labels exist — any "accuracy" would be measured against
our own rules (circular).
**Alternative:** Treat policy agreement with heuristics as accuracy.
**Trade-off:** We cannot claim the 38% escalation rate is "correct", only that it is
auditable per-decision with reasons (failure mode FM5 shows where it under-fires).

## D23 — Vectorized `lexsort` retrieval after the 10-minute timeout
**Decision:** Replaced per-query Python `sorted()` over 42,770 docs with precomputed
pair-rank + `np.lexsort((-sims, pair_rank))`, preserving the exact (-sim, pair_id) order.
**Why:** The first full 150-example agent run timed out at 10 min (300 Python sorts);
lexsort cut per-query ordering to milliseconds with identical determinism.
**Alternative:** Approximate nearest neighbors (e.g. annoy/faiss).
**Trade-off:** Exact search retained (better for auditability); ANN deferred to scale work.

## D24 — Resume-safe annotator that never auto-marks reviewed
**Decision:** `human_calibration.csv` ships with 100 `pending` rows and empty scores;
the annotator saves after every row, skips completed rows, and only marks `reviewed`
when all six scores are human-entered (smoke-tested with a quit-without-saving run).
**Why:** The 0→100 human-labeling step is the highest-integrity-risk moment of Phase 4;
any auto-fill path (default values, Enter-to-confirm on empty) could silently fabricate labels.
**Alternative:** Pre-fill with model suggestions for the human to confirm.
**Trade-off:** Slower annotation (100 fully manual rows) — accepted; suggestions would
anchor the human and inflate agreement.

## D25 — Optional labeled LLM suggestions in the annotator (default OFF, confirmation-tracked)
**Decision:** Added an opt-in `--suggest` flag that shows per-row LLM scores ONLY as
"AI SUGGESTION — NOT HUMAN LABEL"; a suggestion becomes a label ONLY on explicit
per-field human confirm/override, tracked in a new `ai_suggestion_shown` column; added
`--status` for non-interactive validation counts. Default path remains fully manual.
**Why:** The assignment permits suggestions under strict labeling + explicit confirmation;
per-field confirmation with an audit column keeps the speedup without silent fabrication.
**Alternative:** No suggestions at all (pure D24), or pre-filled editable defaults.
**Trade-off:** Confirmed suggestions may anchor the human and inflate judge-human
agreement — disclosed in the report; the audit column lets agreement be recomputed on
suggestion-free rows only.
