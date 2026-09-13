# Top 5 Failure Modes — AI Support Agent (real examples, 150-example eval)

Source: `results/agent_outputs.jsonl` (150 rows) via `scripts/analyze_failures.py`.
Agent: TF-IDF kNN intent + TF-IDF top-1 extractive reply + rule escalation.
Intent accuracy 0.4867 (77/150 errors), escalation rate 0.38 (57/150).

---

## FM1 — Context-dependent follow-ups misclassified even at HIGH similarity
**Real examples:**
- `pair_110043_1822595` — "@115888 are u serious" (true=plan_management_discount, pred=offline_downloads_issue, sim=0.865, auto_handle — confidently wrong)
- `pair_396203_396204` — "Just sent you a DM" (true=playlist_library_curation, pred=offline_downloads_issue, sim=0.969, auto_handle)
- `pair_500990_2920668` — "Down in NYC" (true=playback_streaming_issue, pred=unclear_insufficient_context, sim=0.898, escalated)
- `pair_997168_997167` — "Linky no worky" (true=plan_management_discount, pred=unclear, sim=0.843, escalated)

**Why it failed:** Follow-up fragments carry almost no lexical signal; TF-IDF matches
surface-similar historical fragments whose weak labels point anywhere. High similarity
(0.86–0.97) gives false confidence — similarity measures wording overlap, not intent match.
**Hypothesis:** The agent under-uses `context_before_customer` for short messages.
**Possible fix:** Follow-up detector (message < ~6 content words + non-initial turn) →
inherit the parent-turn intent or force-escalate for human triage.

## FM2 — Ultra-short / low-signal messages collapse to `unclear_insufficient_context`
**Real examples (largest confusion pairs: catalog→unclear ×9, playback→unclear ×6):**
- `pair_785533_785532` — "get rbd" (true=content_catalog_licensing, pred=unclear, sim=0.511)
- `pair_627556_627555` — "It's 2017 and I'm still waiting for @11619 to be on @115888" (true=catalog, pred=unclear)
- `pair_888581_888582` — "Windows 950xl phone." (true=app_technical_device, pred=unclear, sim=1.000)
- `pair_2757611_2757610` / `pair_2895441_2895443` — "Thank you!!" / "Thanks!" (true=app_technical / catalog, pred=unclear, escalated)

**Why it failed:** Short messages match other short vague messages in the pool
(kNN inherits the majority label of low-signal neighbors); 38/150 escalations come
from this single rule. Some cases are arguably debatable human labels (a bare
"Thanks!" carries no diagnosable intent).
**Hypothesis:** kNN has no minimum-signal prior; weak-label noise concentrates in short texts.
**Possible fix:** Minimum-signal rule + keyword priors for short texts; audit bare-social
messages as a label-quality slice rather than pure model errors.

## FM3 — Weak-label noise: near-duplicates with wrong heuristic labels (sim=1.000, still wrong)
**Real examples:**
- `pair_199121_199122` — "Please don't make this the default view of artwork…" (true=other_miscellaneous, pred=unclear, sim=1.000)
- `pair_1106328_1106329` — "Why did you remove the ability to play from all sub-folders…" (true=playlist_library_curation, pred=content_catalog_licensing, sim=1.000)
- `pair_888581_888582` — same message exists in the training pool labeled `unclear` (FM2).

**Why it failed:** The 42,770 training labels are heuristic/weak (Phase 2 D11). kNN
amplifies label noise: a similarity of 1.000 only means "same wording seen before",
not "correct label".
**Hypothesis:** Deterministic retrieval is working as designed; the supervision signal is the ceiling.
**Possible fix:** Label cleaning on high-frequency duplicates, keyword cross-check override,
confidence-weighted voting; report kNN accuracy as a lower bound, not a product claim.

## FM4 — Non-English messages fail (English-only pipeline)
**Real examples:**
- `pair_662956_662955` — "udah daftar premium tapi gak bisa log in, bisa dibantu" (Indonesian; true=account_access_credentials, pred=other_miscellaneous, sim=0.485, auto_handle)
- `pair_1095355_1095354` — French playlist/app request (true=playlist_library_curation, pred=other_miscellaneous, sim=0.340, auto_handle)
- `pair_926727_926726` — "c'est très urgent svp" (French; true=unclear, pred=other_miscellaneous)

**Why it failed:** TF-IDF vocabulary and taxonomy patterns are English-only (Phase 2 D2
kept all languages in data but English-only modeling). Non-English tokens are OOV → near-zero
informative similarity → fallback labels.
**Hypothesis:** Coverage gap, not a modeling subtlety; ~12% of traffic is non-English.
**Possible fix:** Language gate → route non-English to human/DM immediately (escalate, don't guess).

## FM5 — Security-critical misses: account-takeover phrasing → wrong intent → auto_handle (highest severity)
**Real examples:**
- `pair_883440_883439` — "someone is playing music that i never played, i think someone is in my account" (true=account_access_credentials, pred=other_miscellaneous, sim=0.416, **auto_handle** — should have escalated)
- `pair_315118_315117` — "Hi, I can't login on android. App keeps telling me the device is offline…" (true=account_access_credentials, pred=offline_downloads_issue, sim=0.700, **auto_handle**)
- `pair_647116_647115` — "Spotify has paused cos your account is being used elsewhere… Whoever is using my account…" (true=account_access_credentials, pred=playback_streaming_issue, sim=0.381, **auto_handle**)

**Why it failed:** The escalation policy is only as good as the classifier: paraphrases
("someone is in my account" vs "hacked") miss both keyword patterns and kNN majority,
so the security-escalation rule never fires and the case is auto-handled.
**Hypothesis:** Security recall is the binding safety constraint; precision-style tuning elsewhere is secondary.
**Possible fix:** Security-first override (account-takeover lexicon: "someone…my account",
"used elsewhere", "logged out…all" → force account_access_credentials + escalate);
ensemble kNN with keyword vote where security keywords veto; red-team slice in every eval.
