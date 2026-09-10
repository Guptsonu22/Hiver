# Hiver Customer Support System 

Automated customer support intelligence platform built on real-world customer support interactions from the Twitter Customer Support dataset.

---

## Phase 1: Exploratory Data Analysis & Brand Selection

### Dataset Overview
* **Source Dataset:** Customer Support on Twitter (`data/raw/twcs.csv`)
* **Volume:** 2,811,774 customer-support tweets across 108 international brands.
* **Top Evaluated Candidates:** AmazonHelp (177k pairs), AppleSupport (106k pairs), Uber_Support (56k pairs), SpotifyCares (43k pairs), Delta (42k pairs).

### Selected Brand: `SpotifyCares`
* **Direct Interaction Pairs:** 43,092 Customer $\rightarrow$ Brand interactions.
* **Why SpotifyCares over Higher-Volume Brands?**
  1. **Actionable Troubleshooting Guidance:** While airline handles (e.g. Delta, AmericanAir) primarily issue flight status notices, and retail handles (e.g. AmazonHelp) predominantly issue logistics/tracking redirects, SpotifyCares responses contain dense, repeatable, step-by-step technical procedures (cache clearing, offline mode toggling, clean reinstall routines, audio driver resets).
  2. **High Domain Specificity & Diversity:** Encompasses 8 clear recurring technical failure domains (playback, offline downloads, authentication, billing, plan tiers, catalog licensing, playlist curation, app stability).
  3. **Natural Escalation Boundaries:** Sharp distinctions exist between automatable technical inquiries (deflectable via retrieval) and high-risk account takeover or double-billing issues (requiring immediate escalation to human support via DM).
  4. **Grounding Suitability:** Rich conversational context allows historically grounded, policy-compliant response generation without hallucination.

---

## Phase 2: SpotifyCares Conversation Reconstruction & Intent Discovery

### Core Objectives
1. Convert raw isolated tweets into structured, hierarchical conversation trees.
2. Extract direct Customer $\rightarrow$ SpotifyCares dialogue pairs enriched with preceding conversation context.
3. Classify response types heuristically to understand brand behavior.
4. Empirically discover customer support intents and build a machine-readable taxonomy.
5. Identify and categorize hard, ambiguous, and multi-intent edge cases.

---

### Empirical Key Metrics (Audited & Verified)

| Metric | Value | Population / Denominator |
|:---|:---:|:---|
| **Raw Dataset Tweets** | 2,811,774 | Total rows in `data/raw/twcs.csv` |
| **SpotifyCares Outbound Tweets** | 43,265 | `author_id == 'SpotifyCares' & inbound == False` |
| **Customer Inbound Tweets (Ecosystem)** | 48,543 | All customer tweets in SpotifyCares conversation components |
| **Co-Tagged Third-Party Brands** | 81 | Other brand handles mentioned in Spotify threads (e.g. `@hulu_support`) |
| **Total Ecosystem Tweets** | **91,889** | $43,265 + 48,543 + 81 = 91,889$ (captured via graph expansion) |
| **Reconstructed Conversation Trees** | **28,280** | Connected components rooted at initial inquiries/broadcasts |
| **Direct Customer $\rightarrow$ Brand Pairs** | **43,092** | Turn 0 initial inquiries ($26,966$) + follow-up turns ($16,126$) |
| **Customer Messages in Pairs** | 41,585 | Unique customer tweets receiving replies (some received 2 brand replies) |
| **Unpaired Customer Tweets in Ecosystem** | 6,958 | Final replies with no subsequent brand tweet, or consecutive user tweets |
| **English Language Prevalence** | **87.74%** | 37,811 / 43,092 customer messages in pairs (88.28% of all 48,543) |
| **Brand DM Redirect Rate** | **25.97%** | 11,191 / 43,092 pairs redirecting to private Twitter DM |
| **Flagged Hard Cases** | **8,624** | Ambiguous, multi-intent, hostile, or context-dependent interactions |

---

### Intent Taxonomy (10 Categories)

The taxonomy was derived via unsupervised TF-IDF + KMeans clustering ($k=8$) and refined into 8 domain-specific technical intents, 1 ambiguity category, and 1 miscellaneous category.

#### Initial Inquiries Distribution ($n = 26,966$)
> **Population Definition:** Exactly all 26,966 Turn-0 initial customer inquiries in `spotify_pairs.parquet` (`is_initial_inquiry == True`). Labels are mutually exclusive.

| Intent Identifier | Display Name | Count | % | Escalation? | Actionability / Description |
|:---|:---|:---:|:---:|:---:|:---|
| `other_miscellaneous` | Other / Miscellaneous | 9,252 | 34.31% | No | Social praise, general chatter, no actionable failure |
| `unclear_insufficient_context` | Unclear / Insufficient Context | 4,464 | 16.55% | No | Vague expressions (*"broken"*, *"help"*); prompt for device/OS |
| `billing_subscription_payment` | Billing / Payment / Charges | 3,043 | 11.28% | **Yes** | Double charges, disputed renewal fees, refund requests |
| `playlist_library_curation` | Playlist / Library / Curation | 2,788 | 10.34% | No | Recover deleted playlists, shuffle algorithm bugs, queue order |
| `account_access_credentials` | Account Access / Credentials | 1,820 | 6.75% | **Yes** | Password resets, account compromise/hack, login loops |
| `app_technical_device` | App Stability / Device Integration | 1,183 | 4.39% | No | App crashes, freezes, Apple Watch/CarPlay/Sonos connectivity |
| `plan_management_discount` | Plan Management / Student & Family | 1,174 | 4.35% | No | SheerID verification, Family Plan address verification rules |
| `playback_streaming_issue` | Playback / Streaming Issue | 1,138 | 4.22% | No | Online audio stuttering, skips, buffer errors, silence |
| `offline_downloads_issue` | Download / Offline Issue | 1,080 | 4.01% | No | Disappeared downloads, 3-device limit, offline sync failure |
| `content_catalog_licensing` | Content Catalog / Licensing | 1,024 | 3.80% | No | Greyed-out songs, regional licensing restrictions, release dates |
| **Total** | — | **26,966** | **100.00%** | — | $\\sum \\text{counts} == 26,966$ (Validated) |

---

### Response Type Distribution ($n = 43,092$)

Every customer-brand interaction pair is classified into exactly one mutually exclusive functional response category:

| Response Category | Count | % | Description |
|:---|:---:|:---:|:---|
| **Other / Miscellaneous** | 19,283 | 44.75% | Canned conversational replies and social chatter |
| **Redirect / DM** | 11,191 | 25.97% | Transitioning customer to private DM for account privacy/lookups |
| **Clarification / Request for Info** | 5,759 | 13.36% | Diagnostic questions (device model, OS version, app version) |
| **Generic Acknowledgement** | 3,109 | 7.21% | Gratitude and social closing (*"Enjoy the music!"*, *"Rock on!"*) |
| **Information / Explanation** | 1,892 | 4.39% | Explaining catalog availability, rights restrictions, or policy |
| **Troubleshooting / Actionable** | 1,856 | 4.31% | Step-by-step guidance (clean reinstall, cache clearing, reboot) |
| **Redirect / External Support** | 2 | 0.00% | Third-party device manufacturer or partner redirects |
| **Total** | **43,092** | **100.00%** | $\\sum \\text{counts} == 43,092$ (0 nulls) |

---

### Hard Cases Distribution ($n = 8,624$)

Extracted from `spotify_pairs.parquet` to evaluate model robustness against dialogue edge cases:
* **Context-Dependent Follow-up:** 4,334 (50.26%) — Ultra-short replies (*"iPhone 7"*, *"tried that"*) requiring preceding turns.
* **Multi-Intent Messages:** 3,050 (35.37%) — Inquiries spanning multiple topics; resolved via business priority rules.
* **Insufficient Context:** 830 (9.62%) — Initial inquiries lacking technical symptoms (*"why is Spotify not working"*).
* **Sarcasm / Hostile Tone:** 408 (4.73%) — Frustrated or aggressive complaints requiring empathetic de-escalation.
* **Noisy / Attachment-Only:** 2 (0.02%) — Tweets containing only screenshot URLs and no text.

---

### Methodological & Safety Disclaimers

> [!IMPORTANT]
> **Heuristic / Weak Labels Notice:**
> The intent labels and response categories in `spotify_pairs.parquet` and documentation are **heuristic / weak labels** generated by deterministic pattern matching and priority rules. They do **not** constitute ground-truth annotations and were not human-audited at 100% scale.

> [!NOTE]
> **Evaluation Separation:**
> The Phase 3 golden evaluation set is intentionally kept separate and was **not** used during Phase 2 taxonomy development.

---

### Reproducibility Guide

All processed data can be deterministically reproduced from the raw dataset (`data/raw/twcs.csv`):

```bash
# 1. Reconstruct conversation trees, isolate ecosystem, extract pairs, and flag hard cases
python scripts/prepare_spotify_data.py

# 2. Run data hygiene, conversation topology, and language audits
python scripts/analyze_spotify_conversations.py

# 3. Fit TF-IDF/KMeans, evaluate intent distributions, and regenerate taxonomy documentation
python scripts/discover_intents.py

# 4. Run automated test suite (72 unit tests with mathematical integrity assertions)
python -m pytest tests/test_phase2.py -v
```

---

### Deliverables & Artifact Inventory

* `data/processed/spotify_clean.parquet` — 91,889 ecosystem tweets *(reproducible via script; excluded from Git)*
* `data/processed/spotify_conversations.jsonl` — 28,280 conversation trees *(reproducible via script; excluded from Git)*
* `data/processed/spotify_pairs.parquet` — 43,092 Customer $\rightarrow$ Brand pairs *(reproducible via script; excluded from Git)*
* `data/processed/spotify_pairs_sample.csv` — 100-row human-readable inspection sample *(committed)*
* `data/processed/hard_cases.csv` — 8,624 categorized edge cases *(reproducible via script; excluded from Git)*
* `docs/spotify_data_quality.md` — Full empirical data hygiene and language report.
* `docs/intent_taxonomy.md` — Machine-readable taxonomy specifications and boundaries.
* `docs/intent_examples.md` — Representative real tweet examples per intent.
* `DECISIONS.md` — Formal architectural decision log (D1–D12).
* `src/data/` — Core modules for loading, conversation tree reconstruction, and response classification.
* `src/intents/taxonomy.py` — Taxonomy definitions, regex patterns, and tie-breaking priority rules.
* `tests/test_phase2.py` — Pytest suite (**72/72 tests passing**).
