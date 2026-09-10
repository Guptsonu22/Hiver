# DECISIONS.md — SpotifyCares Phase 2

> **Format:** Decision → Reason → Alternative considered → Trade-off / consequence

---

## D1 — Graph-expansion approach for conversation reconstruction

**Decision:** Use iterative BFS/graph-expansion to collect ALL tweets in a conversation component — not a simple parent-join.

**Reason:** SpotifyCares replies to customer tweets, those customers reply back, SpotifyCares replies again, and other brands occasionally co-tag. A simple `WHERE in_response_to_tweet_id IN (brand_tweet_ids)` join misses multi-turn continuations and ancestor context. The empirical data confirmed that starting from 43,265 SpotifyCares tweets and expanding iteratively converges in ~5 iterations to 91,889 tweets — more than 2× the starting set.

**Alternative:** Single-level parent join (only direct customer→SpotifyCares pairs).

**Trade-off:** Slower on large datasets (O(n) per expansion iteration), but context richness is essential for Phase 3 retrieval. The expansion converges quickly in practice and is re-used from the persisted parquet rather than re-run.

---

## D2 — Language strategy: retain all tweets, restrict evaluation to English

**Decision:** Retain all 91,889 tweets in `spotify_clean.parquet` regardless of detected language. Taxonomy, pattern matching, and intent evaluation operate on English text only.

**Reason:** Language detection is noisy on short tweets. Removing non-English data would lose potentially usable code-switched messages and would make the raw counts non-reproducible. Restricting evaluation to English (87.74% of customer messages by stopword heuristic) keeps the classification quality assessable without losing raw data.

**Alternative:** Drop non-English rows upfront.

**Trade-off:** Non-English messages receive intents from English-language patterns (lower accuracy), but this is acceptable at Phase 2 where no LLM-based classifier is deployed yet. Phase 3 can introduce language gates.

---

## D3 — Multi-intent tie-breaking priority rules (choose one primary intent)

**Decision:** When multiple intent patterns match a single message, apply a deterministic priority order: (1) billing_subscription_payment over plan/account if financial keywords present, (2) offline_downloads over playback, (3) account_access over plan_management if login/security keywords present, (4) playlist_library over playback if shuffle/repeat/playlist keywords present, (5) first-matched otherwise.

**Reason:** Downstream agent routing needs a single label. Sending a billing dispute to the wrong queue creates customer harm. The priority order was derived from business impact: financial issues > security issues > feature-specific diagnosis.

**Alternative:** Emit all matching intents and route to a multi-label classifier in Phase 3.

**Trade-off:** ~3,050 genuinely multi-intent messages are reduced to one label. For Phase 2 analysis this is acceptable; Phase 3 can revisit with a proper multi-label model.

---

## D4 — Brand broadcasts excluded from Customer→Brand pairs

**Decision:** 22 SpotifyCares tweets that are NOT replies to any customer tweet (i.e., have no `in_response_to_tweet_id` pointing to a customer) are NOT included in `spotify_pairs.parquet`.

**Reason:** The pair structure represents a customer need + support response. Proactive broadcasts (outage notices, feature announcements) have no corresponding customer inquiry and would pollute the response-type distribution with atypical responses.

**Alternative:** Include as a "proactive_broadcast" pair type.

**Trade-off:** 22 tweets (~0.05% of SpotifyCares output) are excluded. The broadcast pattern is documented in `docs/spotify_data_quality.md` for future reference.

---

## D5 — `unclear_insufficient_context` is a first-class intent, not a fallback label

**Decision:** `unclear_insufficient_context` is defined as a named intent in `INTENT_TAXONOMY` with its own patterns, examples, actionability, and escalation flag — not merely an `else` branch.

**Reason:** 16.55% (4,464 / 26,966) of initial customer inquiries are genuinely under-specified. Treating this as a labeled intent class rather than a null result allows: (a) explicit counting and monitoring, (b) a distinct routing action (prompt the customer with clarifying questions), (c) accurate distribution reporting.

**Alternative:** Return `None` for unclassifiable messages; treat as special case in downstream code.

**Trade-off:** Slightly inflates "non-actionable" categories, but produces cleaner, more truthful distribution statistics and simpler downstream handling.

---

## D6 — Response type != Resolution (heuristic classifier limitation)

**Decision:** The response classifier in `src/data/response_classifier.py` classifies the *functional type* of a SpotifyCares reply, NOT whether the customer's issue was resolved.

**Reason:** Resolution assessment requires the full conversation outcome — whether the customer's follow-up confirms success or continues complaining. That level of annotation is not present in the raw dataset and would require LLM-based inference (Phase 3 scope). The heuristic captures what SpotifyCares *said*, not whether it *worked*.

**Alternative:** Attempt resolution scoring via pattern matching on follow-up turns.

**Trade-off:** Heuristic classification is clearly labeled as such throughout the codebase. All classification outputs include the explicit note "⚠️ HEURISTIC — not ground truth" in documentation.

---

## D7 — k=8 clusters for KMeans intent discovery

**Decision:** KMeans with k=8 clusters and random_state=42 was used for initial intent discovery.

**Reason:** TF-IDF + KMeans was used as an exploratory, interpretable tool — not as the deployed classifier. k=8 was selected because it aligns with 8 observable technical problem domains in Spotify support (playback, offline, account, billing, plan, catalog, playlist, device). The cluster→intent mapping was validated manually against representative examples.

**Alternative:** Use silhouette score or elbow method to auto-select k.

**Trade-off:** Manual k selection introduces subjectivity. However, domain-knowledge-driven k is more interpretable than a statistical optimum for this use case. The full taxonomy was then refined beyond the clusters (e.g., separating `unclear_insufficient_context` and `other_miscellaneous` which the KMeans clusters were too coarse to distinguish).

---

## D8 — Orphan brand replies (37 tweets) kept in the ecosystem but excluded from pairs

**Decision:** 37 SpotifyCares replies whose parent tweet ID exists in the parquet but the parent tweet is NOT in the SpotifyCares ecosystem (edge case where the parent may have been deleted or belongs to a different brand) are included in `spotify_clean.parquet` but do NOT generate pairs.

**Reason:** Their parent context is unavailable so they cannot be meaningfully used as training examples. Including them in the parquet preserves data completeness; excluding from pairs prevents partial-context noise.

**Alternative:** Attempt to look up the parent from the full `twcs.csv` at pair-time.

**Trade-off:** 37 SpotifyCares replies (0.09% of all brand output) are non-pairable. Documented in `docs/spotify_data_quality.md`.

---

## D9 — Hard-case annotation is heuristic, not human-labeled

**Decision:** The `data/processed/hard_cases.csv` file is produced by automated heuristics (pattern matching for sarcasm/hostility, multi-intent detection, follow-up detection, insufficient context scoring) rather than human annotation.

**Reason:** Manual annotation of 8,624+ hard cases is outside the Phase 2 scope. The heuristics are calibrated to surface likely-difficult examples for future human review or Phase 3 LLM-assisted labeling.

**Alternative:** Defer hard-case identification entirely to Phase 3.

**Trade-off:** Heuristic false-positive and false-negative rates are unknown. The output serves as a starting point for human review, not a ground truth.

---

## D10 — Parquet as primary processed data format

**Decision:** All processed datasets are stored as `.parquet` files (`spotify_clean.parquet`, `spotify_pairs.parquet`). Raw JSONL for conversations, CSV only for human-readable samples.

**Reason:** Parquet provides column-oriented compression (10–20× smaller than equivalent CSV for this dataset), fast pandas read-back, and preserves nullable Int64 types across serialization. The raw `twcs.csv` is ~250 MB; the processed parquet files are <20 MB combined.

**Alternative:** Store all outputs as CSV for maximum portability.

**Trade-off:** Parquet requires pyarrow or fastparquet. Both are standard in the project requirements. CSV fallback sample (100 rows) is always co-generated.

---

## D11 — Intent label methodology: heuristic/weak labels, not ground truth

**Decision:** Intent labels assigned in `spotify_pairs.parquet` (`customer_candidate_intents`, `customer_primary_intent`) are explicitly designated as **heuristic / candidate / weak labels**, not ground truth.

**Reason:** No human-annotated gold labels exist for all 43,092 pairs in the raw Twitter dataset. The labels are generated by deterministic pattern matching (`detect_candidate_intents`) and prioritized domain rules (`classify_customer_intent`). Calling automated classifications "ground truth" creates a false impression of accuracy and misleads downstream evaluation.

**Alternative:** Use zero-shot LLM classification on the entire dataset.

**Trade-off:** LLM generation across 43k rows is cost-prohibitive and violates Phase 2 scoping rules. The deterministic heuristic approach is fast, 100% reproducible, and mathematically auditable, but must be treated as weak supervision for Phase 3.

---

## D12 — Population definition for intent distribution reporting (n = 26,966)

**Decision:** The canonical intent distribution reported in the Phase 2 specification represents **all 26,966 Turn-0 initial customer inquiries** (`is_initial_inquiry == True` in `spotify_pairs.parquet`).

**Reason:** In an earlier iteration, the table was computed on a filtered subset of 26,568 inquiries with clean text length > 10 characters (excluding 398 ultra-short inquiries). This caused a confusing mismatch where the table summed to 26,568 while the text cited $n=26,966$. Computing the distribution over the exact, full initial inquiry population ($n=26,966$) ensures mathematical integrity: $\sum \text{counts} = 26,966$ (100.00%).

**Alternative:** Report distribution only on the NLP-filtered subset ($n=26,568$) or only on English inquiries.

**Trade-off:** Including ultra-short inquiries increases the proportion of `unclear_insufficient_context` from 15.33% to 16.55%, which is a truthful representation of raw Twitter support reality.

