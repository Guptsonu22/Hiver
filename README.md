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
* `DECISIONS.md` — Formal architectural decision log (D1–D13 consolidated).
* `src/data/` — Core modules for loading, conversation tree reconstruction, and response classification.
* `src/intents/taxonomy.py` — Taxonomy definitions, regex patterns, and tie-breaking priority rules.
* `tests/test_phase2.py` — Pytest suite (**72/72 tests passing**).

---

## Phase 4: LLM-as-Judge Reply Quality + Human Agreement (scaffolding complete, calibration pending)

Phase 3 reply metrics (length / non-empty rate) cannot judge usefulness or
grounding. Phase 4 adds semantic reply-quality evaluation with a rigorous,
reproducible LLM-judge + human-agreement protocol. Full rubric:
`docs/judge_rubric.md`.

* **Population:** strictly the 150 reviewed golden examples (172 pending never
  evaluated). Full judge inputs: 300 rows (150 × 2 baselines) in
  `data/judge/judge_inputs_full150.jsonl`, each with `generated_reply`,
  `retrieval_source_pair_id`, and `retrieved_historical_response` from the
  leakage-free 42,770-pair pool (asserted disjoint from the 322 golden IDs).
* **Rubric v1 (0–3):** relevance, helpfulness, groundedness, appropriateness
  (0=poor…3=strong); unsupported_claims INVERTED (0=clean/best…3=fabricated/worst);
  plus overall_score, confidence, concise evidence-based reason.
* **Calibration:** deterministic 50-example subset (seed 42) →
  `data/judge/human_calibration.csv` (100 rows, currently all `pending`, scores
  EMPTY — awaiting genuine human review via
  `scripts/annotate_judge_calibration.py`).
* **Judge:** `src/judge.py` (schema-validated JSON, env-configured
  `JUDGE_MODEL`, fail-closed without `OPENAI_API_KEY`, cached outputs in
  `results/judge_outputs.jsonl`); runner `scripts/run_judge.py` enforces
  calibration-first ordering and writes `results/judge_metrics.json`
  (agreement: exact rate + linear-weighted kappa; per-baseline means).
* **Status:** implementation + judge/agent/annotator unit tests done (full suite 185 passed,
  1 skipped); calibration 0/100 (one unverifiable row reset to pending for integrity);
  NO human labels / judge scores / agreement numbers fabricated —
  `results/judge_metrics.json` will be generated only after real annotation +
  real API calls.

---

## Reproducibility (setup → headline results)

**Setup (Windows PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt  # pandas, numpy, pyarrow, scikit-learn, scipy, pytest
```
`python` alone may resolve to an interpreter without dependencies — always use
`.\.venv\Scripts\python.exe`. No `openai` package needed (LLM calls use stdlib `urllib`).

**Commands:**
```powershell
.\.venv\Scripts\python.exe -m pytest -q                          # full suite (expect 185 passed, 1 skipped)
.\.venv\Scripts\python.exe scripts/build_judge_inputs.py         # 300 judge inputs + 50-ex calibration + 100-row template
.\.venv\Scripts\python.exe scripts/annotate_judge_calibration.py # human calibration (100 rows; --status to check; --suggest optional, needs key)
.\.venv\Scripts\python.exe scripts/generate_machine_calibration.py # MACHINE diagnostic only (never human labels; see docs/machine_calibration.md)
.\.venv\Scripts\python.exe scripts/run_judge.py                  # LLM judge → agreement + results/judge_metrics.json (needs 100/100 + key)
.\.venv\Scripts\python.exe scripts/run_judge.py --machine-calibration # machine diagnostic → results/machine_metrics.json (no agreement)
.\.venv\Scripts\python.exe scripts/evaluate_agent.py             # agent on 150 → results/agent_metrics.json (+ --judge after calibration)
.\.venv\Scripts\python.exe scripts/analyze_failures.py --top 5   # confusion pairs, weak-evidence cases, escalation reasons
```

**Expected artifacts:** `results/baseline_metrics.json`, `results/agent_metrics.json`,
`results/agent_outputs.jsonl`, `results/judge_metrics.json` + `results/judge_outputs.jsonl`
(after genuine human calibration + judge run); machine-diagnostic only:
`data/judge/machine_calibration_100.csv` + `results/machine_metrics.json` (NOT human
labels, NO agreement — see `docs/machine_calibration.md`);
`data/judge/{judge_inputs_full150.jsonl, calibration_sample.csv,
human_calibration.csv}`; docs: `judge_rubric.md`, `failure_modes.md`, `decision_log.md`
(D1–D13), `final_report.md` (15 sections).

**API configuration:** set a freshly rotated `OPENAI_API_KEY` in the environment
(optional `JUDGE_MODEL`, default `gpt-4o-mini`); never commit secrets (`.env`/`.env.*`/
`credentials.json` are git-ignored; `.env.example` is a blank template). Without a key
every LLM step fails closed with an actionable message.

**Calibration → judge workflow:** annotate 100/100 rows (scores 0–3, `unsupported_claims`
inverted; fast path: `annotate_judge_calibration.py --fast` shows one labeled AI
suggestion per row for explicit `y` (confirm) / `e` (edit) / `n` (manual) / `q` (quit) —
Enter alone never confirms; suggestions cached in `data/judge/suggestion_cache.jsonl`;
provenance per row in `ai_suggestion_shown`/`human_confirmed`/`human_edited`/
`annotation_mode`) → `run_judge.py` validates (100/100, ⊂ reviewed 150, ∩ pending =
∅, evidence ∩ 322 golden = ∅) → judges calibration (agreement: exact + linear-weighted
kappa) → judges full 150×2 → `evaluate_agent.py --judge` adds agent reply quality.
