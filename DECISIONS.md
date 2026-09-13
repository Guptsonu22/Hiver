# Decision Log — SpotifyCares Support Intelligence (D1–D13)

> **Format:** Decision → Why → Alternative → Trade-off.
> Thirteen non-obvious engineering decisions made during this project.
> All decisions are genuine — none reconstructed or invented.

---

## D1 — SpotifyCares selected over higher-volume brands

**Decision:** Build on SpotifyCares (43,092 pairs) rather than AmazonHelp (177k) or AppleSupport (106k).

**Why:** SpotifyCares responses contain *dense, repeatable, step-by-step technical procedures* (cache clearing, offline mode toggling, clean reinstall, audio driver resets) and *sharp escalation boundaries* (billing disputes and account takeovers require DM; playback troubleshooting does not). AmazonHelp predominantly issues logistics/tracking redirects; airline handles issue flight status notices. Neither gives a rich domain for retrieval-grounded reply generation or meaningful escalation policy.

**Alternative:** AmazonHelp (4× the volume).

**Trade-off:** Smaller absolute evaluation pool; all findings are SpotifyCares-specific and may not generalise to retail or travel support.

---

## D2 — Ten-intent taxonomy: 8 technical + `unclear` + `other`

**Decision:** Use exactly 10 mutually exclusive intents derived from TF-IDF/KMeans (k=8 domain clusters) plus two explicit catch-all categories: `unclear_insufficient_context` (16.55% of initial inquiries) and `other_miscellaneous` (34.31%). `unclear_insufficient_context` is a *first-class intent* with its own routing action — not a residual `else` branch.

**Why:** Fewer than 8 technical intents merge meaningfully different problem types (billing and plan management are distinct routing destinations). More than 10 creates classes with <3% support, making kNN unreliable. 16.55% of inquiries are genuinely under-specified; treating them as a named class enables explicit monitoring and a distinct action (prompt for device/OS) rather than silently mis-routing them.

**Alternative:** Emit all matching intents (multi-label) and route based on a priority order.

**Trade-off:** ~3,050 genuinely multi-intent messages receive only one primary label. The priority order (financial > security > feature-specific) encodes business harm severity.

---

## D3 — `pair_id` string match as the sole leakage gate

**Decision:** All leakage checks use exact `pair_id` string equality (e.g. `pair_43459_43458`). All 322 golden IDs are excluded from both the training pool and the retrieval index before any model sees data. `overlap_after_exclusion = 0` is asserted at runtime in `src/agent.py` and `src/evaluate.py`.

**Why:** Tweet IDs are unstable across re-downloads (deleted tweets change the parquet). The constructed `pair_id` is the only stable join key between golden candidates and processed pairs, and exact matching is fully auditable.

**Alternative:** Fuzzy text matching on customer messages to catch near-duplicate wording.

**Trade-off:** ID-level exclusion misses semantically near-duplicate re-tweets. Stated explicitly as a limitation: leakage freedom is *ID-level, not semantic-level*.

---

## D4 — All 322 golden IDs excluded (not just the 150 reviewed)

**Decision:** The 172 *pending* (unannotated) golden candidates are excluded from *both* training and evaluation — not just from the test set.

**Why:** If pending rows remained in training, a future annotator completing them could inadvertently evaluate on data the model has seen. Excluding all 322 pair_ids now prevents this forward leakage at near-zero cost (pool shrinks 43,092 → 42,770, −0.75%).

**Alternative:** Only exclude the 150 reviewed rows from test; keep pending 172 in training.

**Trade-off:** Training pool is marginally smaller; evaluation n=150 is smaller than it could theoretically be, but this is the only honest option.

---

## D5 — Language: retain all data, English-only classification

**Decision:** All 91,889 ecosystem tweets are retained in `spotify_clean.parquet`. Intent taxonomy, pattern matching, and all evaluation operate on English text only (87.74% of customer messages).

**Why:** Language detection is noisy on short tweets. Dropping non-English rows would make raw counts non-reproducible and lose code-switched messages containing English keywords. Restricting *evaluation* to English keeps classification quality assessable without discarding data.

**Alternative:** Drop non-English rows upfront; apply multilingual embeddings.

**Trade-off:** Non-English messages (~12%) receive intents from English-only patterns, producing systematically lower accuracy for that slice. FM4 documents exactly this failure: Indonesian/French messages → `other_miscellaneous` → auto-handled.

---

## D6 — TF-IDF kNN agent; keyword rules kept as a separate baseline

**Decision:** The agent uses TF-IDF kNN (k=5, similarity-weighted vote, keyword fallback at zero similarity) over 42,770 weak-labeled pairs — *not* the keyword taxonomy classifier directly.

**Why:** Reusing the keyword classifier would make the "agent" a relabeled baseline with identical accuracy (0.58). kNN gives a genuinely different, reproducible operating point (accuracy 0.4867, macro F1 0.5086) and enables retrieval-grounded reply generation from the same vector space. The lower-than-baseline accuracy reveals the supervision ceiling of weak labels (FM3: sim=1.000 yet wrong label).

**Alternative:** Fine-tune a transformer on 42k rows; or LLM zero-shot classification.

**Trade-off:** kNN inherits weak-label noise. Its accuracy is *lower* than the keyword baseline — reported plainly rather than tuned away, because the gap is informative.

---

## D7 — Extractive top-1 reply as the default (zero hallucination by construction)

**Decision:** Default `use_llm=False`: the generated reply *is* the top-1 retrieved historical SpotifyCares response. Low-confidence queries receive a fixed clarification template. An opt-in LLM path (`use_llm=True`) exists, fail-closed without an API key.

**Why:** An abstractive default requires judge scores to demonstrate it does not hallucinate — scores we cannot produce without human calibration and an API key. Extractive grounding makes `unsupported_claims ≈ 0` a *structural property of the architecture*, not a measured claim.

**Alternative:** LLM-first generation with retrieved evidence as context.

**Trade-off:** Replies can be stale or topic-mismatched for follow-up fragments (FM1). The trade-off is honesty: what is measurable now is measured; what requires a judge is marked PENDING.

---

## D8 — Escalation as transparent policy; no "escalation accuracy" metric

**Decision:** Escalation uses four deterministic rules: (1) security/financial intent → escalate; (2) `unclear_insufficient_context` → escalate; (3) top-1 similarity < 0.15 → escalate; (4) ambiguity margin < 0.05 → escalate. Only escalation *rate* (0.38) is reported — never accuracy.

**Why:** No human escalation ground truth exists in the dataset. Any "accuracy" metric would measure agreement with our own rules, which is circular. Per-decision reasons are logged in `agent_outputs.jsonl` so a reviewer can audit each decision independently.

**Alternative:** Train a binary escalation classifier on heuristic labels.

**Trade-off:** The policy under-fires on paraphrase-based security cases (FM5: "someone is in my account" → `other_miscellaneous` → auto-handled), because escalation is only as good as the classifier feeding it.

---

## D9 — Weak training labels explicitly distinguished from ground truth

**Decision:** All 43,092 intent labels in `spotify_pairs.parquet` are designated **heuristic / weak labels** throughout code, documentation, and the report. The 150 reviewed golden labels are the only ground truth.

**Why:** Calling automated pattern-match outputs "ground truth" would create a false accuracy ceiling and mislead evaluators. The kNN accuracy of 0.4867 is measured *against human labels*, not against the heuristic labels used for training. The gap between the weak-label ceiling and actual performance is a core finding.

**Alternative:** Treat heuristic labels as ground truth; report kNN accuracy on the full 43k pool.

**Trade-off:** The distinction requires careful communication — FM3 explicitly names weak-label noise as a failure mode — but it is essential for honest evaluation.

---

## D10 — 0–3 rubric with inverted `unsupported_claims`; 50-example human calibration

**Decision:** Five judge dimensions on 0–3; four are higher-is-better, `unsupported_claims` is lower-is-better (0 = no fabrication = best). Human calibration uses a deterministic 50-example subset (seed 42) × 2 baselines = 100 rows. Agreement is reported as exact rate + linear-weighted Cohen's kappa per dimension.

**Why:** 0–3 is annotator-friendly and ordinal enough for weighted kappa. Inverting `unsupported_claims` makes the safety dimension read naturally ("0 unsupported claims" = clean). 50 examples balances annotation cost against statistical signal; linear kappa credits close misses (1 vs 2) more than far misses (0 vs 3) on an ordinal scale.

**Alternative:** Uniform higher-is-better 1–5 Likert; full 150-example calibration; quadratic kappa.

**Trade-off:** At n=50, agreement confidence intervals are wide and findings are indicative, not definitive. The inverted scale is a known annotator-confusion risk — mitigated by explicit labeling on every UI prompt and column header.

---

## D11 — Fail-closed judge: no API key → RuntimeError, never a heuristic stand-in

**Decision:** `get_judge_config()` raises an actionable `RuntimeError` without `OPENAI_API_KEY`. No local heuristic fallback exists. Without a key, all reply-quality cells in the report remain PENDING.

**Why:** A heuristic masquerading as an LLM judge would silently invalidate the entire judge-human agreement claim. Honest PENDING cells are more useful to a reviewer than fabricated numbers. A separately-tracked machine-heuristic diagnostic (`results/machine_metrics.json`) exists for development use only, explicitly labeled "NOT human labels, NOT judge scores, NO agreement".

**Alternative:** Use ROUGE or BERTScore as a local stand-in.

**Trade-off:** Evaluation halts without credentials — by design. The machine diagnostic is documented so developers can verify the pipeline without an API key, but its outputs are never presented as evaluation results.

---

## D12 — Resume-safe annotator that never auto-confirms

**Decision:** The calibration annotator saves after every row, skips already-reviewed rows on resume, and marks a row `reviewed` only when all six scores are explicitly entered. `Enter` alone never advances. The optional `--suggest` flag shows AI scores labeled "AI SUGGESTION — NOT HUMAN LABEL"; each field requires explicit `y` / `e` / `n` — tracked in `ai_suggestion_shown` for anchoring audit.

**Why:** The 0→100 human-labeling step is the highest data-integrity risk in the pipeline. Any auto-fill or default-approval path could silently fabricate labels and corrupt judge-human agreement.

**Alternative:** Pre-fill scores from AI suggestions; human edits where needed.

**Trade-off:** Annotation is slower. The `--suggest` speedup is opt-in with full provenance tracking so agreement can be recomputed on suggestion-free rows only if anchoring is suspected.

---

## D13 — Vectorized `lexsort` retrieval instead of per-query Python sort

**Decision:** Replaced per-query `sorted()` over 42,770 documents with precomputed pair-rank + `np.lexsort((-sims, pair_rank))`, preserving exact (−similarity, pair_id ascending) tie-breaking order.

**Why:** The first full 150-example evaluation run timed out at 10 minutes (150 queries × Python sort over 42k docs each). `lexsort` cuts per-query ordering to milliseconds with identical determinism — verified by comparing top-k outputs before and after the change.

**Alternative:** Approximate nearest neighbours (annoy / faiss) for sub-linear retrieval.

**Trade-off:** Exact search retained (better for auditability and small-n evaluation). ANN deferred to a scale-out next step.
