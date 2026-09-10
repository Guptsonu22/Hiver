# Implementation Plan


---

## Phase 2 — Conversation Reconstruction & Intent Discovery

### Scope

Convert raw SpotifyCares Twitter data into a structured, conversation-aware dataset with per-pair intent labels, ready for Phase 3 response generation.

### Data Pipeline (Steps 2–6, 18)

**Script:** `scripts/prepare_spotify_data.py`

1. Load `data/raw/twcs.csv` (2,811,774 tweets) via `src/data/loader.load_raw_data()`
2. Isolate SpotifyCares ecosystem via iterative BFS graph-expansion: 91,889 tweets in ~5 iterations
3. Reconstruct 28,280 conversation trees via `src/data/conversation_builder.reconstruct_conversations()`
4. Extract 43,092 Customer→Brand pairs via `build_customer_response_pairs()` with turn context
5. Classify response types heuristically via `src/data/response_classifier.classify_response_type()`
6. Classify customer intents via `src/intents/taxonomy.classify_customer_intent()`
7. Output: `spotify_clean.parquet`, `spotify_conversations.jsonl`, `spotify_pairs.parquet`, `spotify_pairs_sample.csv`, `hard_cases.csv`

### Data Quality & Language Analysis (Steps 7–8)

**Script:** `scripts/analyze_spotify_conversations.py`

- Language detection via English stopword heuristic (87.74% English)
- Orphan detection: 37 brand replies with missing parents, 22 brand broadcasts
- Co-tagged brand analysis: 81 messages mention other brands (hulu_support, AppleSupport, etc.)
- Output: `docs/spotify_data_quality.md`

### Intent Discovery (Steps 9–17)

**Script:** `scripts/discover_intents.py`

- TF-IDF vectorization of customer message corpus
- KMeans clustering (k=8, seed=42) for initial grouping
- Manual cluster→intent mapping and refinement
- Pattern-based taxonomy validation
- Output: `docs/intent_taxonomy.md`, `docs/intent_examples.md`

### Final Taxonomy

8 technical intents + `unclear_insufficient_context` + `other_miscellaneous` (10 total).
Escalation recommended for `account_access_credentials` and `billing_subscription_payment`.

### Intent Distribution (Initial Inquiries, Exact Population n = 26,966)

> **Population Definition:** Exactly all 26,966 Turn-0 initial customer inquiries in `spotify_pairs.parquet` (`is_initial_inquiry == True`). Labels are mutually exclusive (single primary intent per inquiry).

| Intent | Count | % |
|:---|:---:|:---:|
| other_miscellaneous | 9,252 | 34.31% |
| unclear_insufficient_context | 4,464 | 16.55% |
| billing_subscription_payment | 3,043 | 11.28% |
| playlist_library_curation | 2,788 | 10.34% |
| account_access_credentials | 1,820 | 6.75% |
| app_technical_device | 1,183 | 4.39% |
| plan_management_discount | 1,174 | 4.35% |
| playback_streaming_issue | 1,138 | 4.22% |
| offline_downloads_issue | 1,080 | 4.01% |
| content_catalog_licensing | 1,024 | 3.80% |
| **Total (Sum)** | **26,966** | **100.00%** |

### Response Type Distribution (All 43,092 Pairs)

| Response Type | Count | % |
|:---|:---:|:---:|
| Other / Miscellaneous | 19,283 | 44.75% |
| Redirect / DM | 11,191 | 25.97% |
| Clarification / Request for Info | 5,759 | 13.36% |
| Generic Acknowledgement | 3,109 | 7.21% |
| Information / Explanation | 1,892 | 4.39% |
| Troubleshooting / Actionable | 1,856 | 4.31% |
| Redirect / External Support | 2 | 0.00% |

### Hard-Case Breakdown (n=8,624)

| Category | Count | % |
|:---|:---:|:---:|
| context_dependent_followup | 4,334 | 50.2% |
| multi_intent | 3,050 | 35.4% |
| insufficient_context | 830 | 9.6% |
| sarcasm_slang_hostile | 408 | 4.7% |
| noisy_or_attachment_only | 2 | 0.02% |

### Test Coverage

69 pytest tests across 8 test classes covering:
brand/customer classification, ecosystem isolation (with empirical regression guards),
conversation reconstruction (BFS ordering, cycle protection), pair extraction
(pair_id format, context availability), taxonomy validation (all 9 intents, pattern compilation,
field completeness), response classifier (all 7 categories), intent classifier
(all 10 intent outcomes, tie-breaking rules), malformed data handling (NaN text, float NaN parent IDs, empty input).

All 69 tests pass.

### Phase 2 Deliverables — Status

| Deliverable | Status |
|---|:---:|
| `data/processed/spotify_clean.parquet` | ✅ |
| `data/processed/spotify_conversations.jsonl` | ✅ |
| `data/processed/spotify_pairs.parquet` | ✅ |
| `data/processed/hard_cases.csv` | ✅ |
| `docs/spotify_data_quality.md` | ✅ |
| `docs/intent_taxonomy.md` | ✅ |
| `docs/intent_examples.md` | ✅ |
| `src/data/loader.py` | ✅ |
| `src/data/conversation_builder.py` | ✅ |
| `src/data/response_classifier.py` | ✅ |
| `src/intents/taxonomy.py` | ✅ |
| `tests/test_phase2.py` (69 tests, all pass) | ✅ |
| `DECISIONS.md` | ✅ |
| `README.md` Phase 2 section | ✅ |
