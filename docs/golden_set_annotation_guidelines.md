# Golden Evaluation Set Human Annotation Guidelines

**Target:** Human annotation of candidate examples for the official frozen golden evaluation set ($n = 150 - 250$).  
**Annotator:** Project Owner (Sonu).  
**Source Dataset:** `data/golden/golden_candidates.csv` (322 candidate pairs).  
**Frozen Output:** `data/golden/golden_set.csv`.

---

## 1. Purpose & Core Philosophy

The purpose of this golden evaluation set is to provide an uncompromised, ground-truth benchmark for evaluating intent classification, diagnostic clarification, and response generation in subsequent phases.

### Absolute Grounding Principles
1. **Human Labeling Only:** The candidate CSV contains weak heuristic signals (`existing_heuristic_intent`, `existing_candidate_intents`) provided **strictly for orientation**. You must independently inspect the actual dialogue and assign the final `human_primary_intent`.
2. **Context-Aware Classification:** Never evaluate follow-up turns in isolation. If a customer writes *"still broken"* or *"iPhone 7"*, use the preceding turns in `context_before_customer` to determine the root problem space.
3. **No Spurious Confidence:** If an inquiry lacks diagnostic symptoms even after checking conversational context, assign `unclear_insufficient_context`. Do not guess or hallucinate an underlying technical fault.
4. **Preserve Multi-Intent Nuance:** If a customer reports two legitimate problems (e.g., app crashing AND double billing charge), mark `human_is_multi_intent = True`, record the primary intent following priority rules, and list the secondary intent in `human_secondary_intents`.

---

## 2. Official 10-Intent Taxonomy Reference

### 1. `playback_streaming_issue`
* **Definition:** Problems playing, streaming, buffering, or continuing audio online.
* **Inclusion:** Audio cutting out, songs pausing unexpectedly, buffer stalls, stuttering, crossfade failures, silent playback while time progress bar moves.
* **Exclusion:** Audio failure occurring strictly with downloaded songs while offline (route to `offline_downloads_issue`). Missing/unplayable tracks due to licensing (route to `content_catalog_licensing`).
* **Confusable Boundaries:** Distinguish from `app_technical_device` (app crashing/freezing is technical device; playback failure where UI is responsive is playback).

### 2. `offline_downloads_issue`
* **Definition:** Downloading tracks for offline listening, offline sync failures, disappeared downloads, or offline device limits.
* **Inclusion:** Downloaded songs unplayable in airplane/offline mode, downloads stuck on "waiting to download", downloaded playlists wiped after app update, 3-device or 10,000 song limit warnings.
* **Exclusion:** Online streaming buffering over WiFi/cellular.
* **Confusable Boundaries:** If customer explicitly mentions *"offline mode"* or *"downloaded songs"*, prioritize `offline_downloads_issue`.

### 3. `account_access_credentials`
* **Definition:** Login failures, password reset problems, compromised/hacked accounts, or credential lockouts.
* **Inclusion:** Password reset email not arriving, account hacked/email changed by intruder, locked out of account, Facebook login disconnection loops.
* **Exclusion:** Customer logged in normally but disputing a billing charge (route to `billing_subscription_payment`).
* **Escalation Policy:** **Immediate Human Escalation Required** (security/privacy).

### 4. `billing_subscription_payment`
* **Definition:** Payment transaction errors, unexpected/duplicate charges, refund requests, or credit card updates.
* **Inclusion:** Charged twice in one month (double charge), charged after subscription cancellation, PayPal/credit card declined, refund demands, VAT receipt requests.
* **Exclusion:** Inquiries about student verification or family plan member invites without active billing dispute (route to `plan_management_discount`).
* **Escalation Policy:** **Immediate Human Escalation Required** (financial transaction).

### 5. `plan_management_discount`
* **Definition:** Subscription tier rules, Family Plan invitations and address verification, Student discounts (SheerID/UNiDAYS), or partner bundles.
* **Inclusion:** Family Plan member unable to join due to address verification failure, renewing Student Discount via SheerID, activating Hulu/Showtime student bundle, how to cancel subscription without financial dispute.
* **Exclusion:** Direct credit card payment rejection or unexpected bank charges (route to `billing_subscription_payment`).

### 6. `content_catalog_licensing`
* **Definition:** Missing or greyed-out songs/albums, geographic/regional rights restrictions, explicit vs. clean version availability, release dates.
* **Inclusion:** Album greyed out in search, songs removed from Spotify, song unavailable in customer's country, release timing inquiries.
* **Exclusion:** User's personal local audio MP3 files not showing up (route to `playlist_library_curation`).

### 7. `playlist_library_curation`
* **Definition:** Managing custom playlists, library organization, shuffle/repeat playback modes, local files syncing, or algorithmic playlists.
* **Inclusion:** Accidentally deleted playlist recovery, shuffle algorithm repeating the same 10 songs, repeat button not looping track, Discover Weekly/Release Radar not updating, local MP3 file import.
* **Exclusion:** Online streaming playback buffering (route to `playback_streaming_issue`).

### 8. `app_technical_device`
* **Definition:** Application crashes, UI freezes, battery drain, OS-level performance issues, or external hardware connectivity.
* **Inclusion:** App crashing on startup, black screen on launch, excessive battery/CPU drain, CarPlay/Android Auto black screen, Bluetooth speaker connection failure, desktop app frozen.
* **Exclusion:** Pure audio buffering when the app UI is completely responsive (route to `playback_streaming_issue`).

### 9. `unclear_insufficient_context`
* **Definition:** Inquiries that express frustration or ask for help but lack sufficient diagnostic, account, or behavioral detail to categorize.
* **Inclusion:** *"help me"*, *"not working"*, *"fix this"*, *"why is this happening"*, or isolated follow-up fragments where prior context is genuinely missing.
* **Exclusion:** Any message containing a concrete error symptom (e.g. *"crashing"* -> `app_technical_device`, *"charged"* -> `billing_subscription_payment`).

### 10. `other_miscellaneous`
* **Definition:** General conversational banter, social praise, artist shoutouts, feature requests that do not fit support, or uncategorizable edge cases.
* **Inclusion:** *"Love the new UI!"*, *"Can you sponsor my podcast?"*, *"Spotify is the best app ever"*.
* **Caution:** Do NOT use `other_miscellaneous` as a shortcut for complex or difficult technical complaints.

---

## 3. Multi-Intent Priority Rules

When a customer message legitimately describes multiple intents, assign `human_primary_intent` according to business impact priority:

1. **Financial & Security Priority:**
   * If a message mentions billing disputes or duplicate charges $\rightarrow$ `billing_subscription_payment`.
   * If a message mentions account compromise or credential lockout $\rightarrow$ `account_access_credentials`.
2. **Feature Specificity Priority:**
   * `offline_downloads_issue` takes precedence over generic streaming `playback_streaming_issue`.
   * `playlist_library_curation` (shuffle/repeat/playlist) takes precedence over generic audio complaints.
3. **Secondary Recording:**
   * Record the other intent(s) in `human_secondary_intents` (e.g. `playback_streaming_issue`).
   * Set `human_is_multi_intent = True`.

---

## 4. Context Evaluation Rules

For follow-up turns (`is_initial_inquiry == False`):
* **Example:**
  * **Customer message:** *"iPhone 7 on iOS 11"*
  * **Preceding context:**
    * *Customer:* *"My app crashes immediately every time I tap the icon."*
    * *Brand:* *"What device and OS are you using?"*
  * **Correct Annotation:**
    * `human_primary_intent`: `app_technical_device` (inherited from root problem statement).
    * `human_is_ambiguous`: `False` (unambiguous with context).
    * `human_notes`: *"Device clarification follow-up for app crash inquiry."*

---

## 5. Fields to Fill in `golden_candidates.csv`

| Column | Type / Allowed Values | Description |
|:---|:---|:---|
| `human_primary_intent` | One of the 10 official intent names | The single primary intent representing the core customer problem. |
| `human_secondary_intents` | Intent name(s), comma-separated, or blank | Any secondary intent(s) present in multi-intent messages. |
| `human_is_ambiguous` | `True` or `False` | True if the inquiry is vague, underspecified, or borderline. |
| `human_is_multi_intent` | `True` or `False` | True if the message expresses more than one distinct support need. |
| `human_notes` | Free text string (optional) | Reasoning for edge cases, borderline calls, or context dependencies. |
| `annotation_status` | `"reviewed"` | Change from `"pending"` to `"reviewed"` once validated. |
