# SpotifyCares Final Customer Intent Taxonomy

## 1. Taxonomy Overview & Derivation Methodology

The intent taxonomy was derived through a multi-stage empirical discovery process on actual SpotifyCares customer interactions:
1. **Initial Inquiry Isolation:** Focused discovery on 26,966 conversation-starting customer messages (Turn 0 initial inquiries) where users state their core support objective.
2. **Keyword & N-Gram Extraction:** Identified primary vocabulary axes (e.g. `charged`, `family plan`, `greyed out`, `shuffle`, `crashing`, `download`).
3. **TF-IDF Vectorization & Unsupervised Clustering:** Ran KMeans clustering (k=8) on the 26,568 inquiries with clean text > 10 characters to discover natural semantic boundaries without human bias.
4. **Actionability & Escalation Calibration:** Grouped clusters into distinct functional intents that demand unique support resolutions or human escalation paths.
5. **Ambiguity Boundary:** Explicitly created the `unclear_insufficient_context` category to capture underspecified or context-free complaints without forcing spurious intent labels.
6. **Label Characterization:** All labels assigned at this phase are **heuristic/weak labels** derived from deterministic pattern matching and priority rules; they do NOT constitute ground truth.

---

## 2. Intent Distribution Summary

### Initial Inquiries Distribution (Root Problem Statements, Total Population n = 26,966):

> **Population Definition:** Exactly all 26,966 Turn-0 initial customer inquiries in `spotify_pairs.parquet` (`is_initial_inquiry == True`). Labels are mutually exclusive (single primary intent per inquiry).

| Intent Identifier | Display Name | Count | Percentage | Cumulative % | Actionability / Support Action |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `other_miscellaneous` | **Other Miscellaneous** | 9,252 | 34.31% | 34.31% | General support or triage. |
| `unclear_insufficient_context` | **Unclear / Insufficient Context** | 4,464 | 16.55% | 50.86% | Prompt customer with standard clarifying questions: device m... |
| `billing_subscription_payment` | **Billing / Payment / Charges** | 3,043 | 11.28% | 62.15% | Direct to payment update page, verify billing history backst... |
| `playlist_library_curation` | **Playlist / Library / Curation** | 2,788 | 10.34% | 72.49% | Provide playlist recovery tool link (spotify.com/recover-pla... |
| `account_access_credentials` | **Account Access / Credentials** | 1,820 | 6.75% | 79.24% | Direct to password reset link, guide through email change re... |
| `app_technical_device` | **App Stability / Device Integration** | 1,183 | 4.39% | 83.62% | Provide clean reinstallation steps, device reboot instructio... |
| `plan_management_discount` | **Plan Management / Student & Family** | 1,174 | 4.35% | 87.98% | Explain Family Plan home address rules, provide SheerID stud... |
| `playback_streaming_issue` | **Playback / Streaming Issue** | 1,138 | 4.22% | 92.20% | Offer audio troubleshooting: toggle hardware acceleration, c... |
| `offline_downloads_issue` | **Download / Offline Issue** | 1,080 | 4.01% | 96.20% | Provide storage check guide, instructions to toggle offline ... |
| `content_catalog_licensing` | **Content Catalog / Licensing** | 1,024 | 3.80% | 100.00% | Explain that music availability depends on artists and right... |

### All Interaction Pairs Distribution (Initial + Follow-ups with Context, Total Population n = 43,092):

> **Population Definition:** All 43,092 Customer → SpotifyCares pairs in `spotify_pairs.parquet`. Each pair is evaluated with preceding dialogue context for follow-up turns.

| Intent Identifier | Total Pairs | Percentage |
| :--- | :---: | :---: |
| `other_miscellaneous` | 12,335 | 28.62% |
| `unclear_insufficient_context` | 6,149 | 14.27% |
| `playlist_library_curation` | 4,897 | 11.36% |
| `billing_subscription_payment` | 4,168 | 9.67% |
| `playback_streaming_issue` | 3,421 | 7.94% |
| `account_access_credentials` | 3,125 | 7.25% |
| `app_technical_device` | 2,501 | 5.80% |
| `content_catalog_licensing` | 2,478 | 5.75% |
| `offline_downloads_issue` | 2,390 | 5.55% |
| `plan_management_discount` | 1,628 | 3.78% |

---

## 3. Detailed Intent Specifications

### `playback_streaming_issue` — Playback / Streaming Issue

- **Description:** Problems playing, streaming, pausing, or continuing audio playback online.
- **Empirical Frequency (Initial Inquiries):** 1,138 (4.22%)
- **Difficulty Rating:** `Easy`
- **Escalation Policy:** Automated Deflection / RAG Candidate
- **Actionable Resolution:** Offer audio troubleshooting: toggle hardware acceleration, clear app cache, check streaming quality settings, restart audio daemon.

**Inclusion Criteria:**
- Audio stops playing unexpectedly mid-song or after every track
- Songs skip continuously or fail to buffer
- Silence, audio distortion, stuttering, or playback error messages during streaming
- Crossfade, gapless playback, or web player playback failures

**Exclusion Criteria:**
- Audio issues occurring strictly when offline or with downloaded tracks (route to offline_downloads_issue)
- Shuffle or repeat playback order logic issues (route to playlist_library_curation)
- Song greyed out or completely absent from catalog (route to content_catalog_licensing)

**Confusable Boundaries:**
- *vs. `offline_downloads_issue`:* Distinguish by online vs offline: offline_downloads applies when songs were saved to device and played without internet.
- *vs. `playlist_library_curation`:* Distinguish by playback failure vs track selection: playback_streaming is technical audio failure; playlist_library is shuffle/repeat order.
- *vs. `app_technical_device`:* Distinguish by audio vs app process: app_technical_device applies when the whole app crashes or freezes to the OS home screen.

**Representative Examples:**
> "Spotify keeps stopping every few seconds when I try to listen to music."
> "Music suddenly cuts out after 10 seconds and skips to the next song."
> "My web player says 'Spotify cannot play this right now' for every single track."
> "Songs keep buffering and pausing every 30 seconds on high speed WiFi."
> "Audio is completely silent even though the progress bar is moving."

---

### `offline_downloads_issue` — Download / Offline Issue

- **Description:** Problems downloading songs/albums for offline listening, offline sync failures, or disappeared downloads.
- **Empirical Frequency (Initial Inquiries):** 1,080 (4.01%)
- **Difficulty Rating:** `Moderate`
- **Escalation Policy:** Automated Deflection / RAG Candidate
- **Actionable Resolution:** Provide storage check guide, instructions to toggle offline mode off/on, verify device limit, and download over stable WiFi.

**Inclusion Criteria:**
- Downloaded tracks not playing when device is in offline mode or without internet
- Download progress stuck, waiting to download, or failing to complete
- Downloaded music suddenly disappearing from the device
- Errors regarding the 3-device download limit or 10,000 song limit per device

**Exclusion Criteria:**
- Online streaming playback buffering (route to playback_streaming_issue)
- Local MP3 files imported from computer not syncing (route to playlist_library_curation)

**Confusable Boundaries:**
- *vs. `playback_streaming_issue`:* Distinguish by online vs offline context. If user specifies 'offline mode' or 'downloaded', choose offline_downloads_issue.
- *vs. `app_technical_device`:* Distinguish by storage/download vs app crash: storage or download sync is offline_downloads_issue.

**Representative Examples:**
> "My downloaded songs won't play when I turn on offline mode on the train."
> "All my downloaded playlists got deleted after the recent update."
> "Downloads are perpetually stuck on 'waiting to download' on my Android phone."
> "Getting an error that I've reached the device limit for downloads, but I only use two phones."
> "Why do my offline songs keep undownloading themselves every few days?"

---

### `account_access_credentials` — Account Access / Credentials

- **Description:** Inability to log in, password reset issues, compromised/hacked accounts, or unexpected credential changes.
- **Empirical Frequency (Initial Inquiries):** 1,820 (6.75%)
- **Difficulty Rating:** `Moderate`
- **Escalation Policy:** **Human Escalation Recommended**
- **Actionable Resolution:** Direct to password reset link, guide through email change recovery, or immediate human escalation for compromised accounts.

**Inclusion Criteria:**
- Forgotten username, password, or password reset email not arriving
- Account hacked, email address changed without user authorization
- Facebook login disconnection or login loop on mobile/desktop
- Error messages: 'Incorrect username or password' or 'Account locked'

**Exclusion Criteria:**
- Customer can log in fine but wants to change billing details (route to billing_subscription_payment)
- Customer can log in but family invitation verification failed (route to plan_management_discount)

**Confusable Boundaries:**
- *vs. `billing_subscription_payment`:* If customer is locked out AND asking about an unexpected charge, prioritize account_access if security compromise is suspected.
- *vs. `plan_management_discount`:* Family invitation errors where user can still log into their own account belong to plan_management_discount.

**Representative Examples:**
> "Can't log into my account, says my password is wrong and reset email never arrives."
> "Someone hacked my Spotify account and changed the email address to an unknown domain!"
> "I'm locked out of my account because I deactivated my old Facebook account."
> "Keep getting logged out every time I close the app and have to re-enter credentials."
> "I have two accounts accidentally and need help accessing my original one."

---

### `billing_subscription_payment` — Billing / Payment / Charges

- **Description:** Payment processing failures, unwanted or duplicate charges, refund requests, or payment method update issues.
- **Empirical Frequency (Initial Inquiries):** 3,043 (11.28%)
- **Difficulty Rating:** `Easy`
- **Escalation Policy:** **Human Escalation Recommended**
- **Actionable Resolution:** Direct to payment update page, verify billing history backstage via DM, or escalate to financial support for refunds.

**Inclusion Criteria:**
- Customer was charged twice in the same billing cycle (double charge)
- Charged after canceling subscription or charged on a free account
- Credit card or PayPal payment declined / update payment method fails
- Requests for refund or invoice/receipt dispute

**Exclusion Criteria:**
- Questions about how to cancel or downgrade without active financial dispute (route to plan_management_discount)
- Student or Family Plan discount verification questions (route to plan_management_discount)

**Confusable Boundaries:**
- *vs. `plan_management_discount`:* If user mentions payment/money/charge/refund, assign billing_subscription_payment. If purely asking about student discount eligibility, assign plan_management_discount.

**Representative Examples:**
> "I've been charged twice for Spotify Premium this month! Need a refund."
> "Why did Spotify take $9.99 from my account when I cancelled last week?"
> "My card was charged but my account still says Spotify Free."
> "Trying to update my payment method with my new credit card but it keeps getting rejected."
> "Where can I find my monthly VAT invoice/receipt for my premium subscription?"

---

### `plan_management_discount` — Plan Management / Student & Family

- **Description:** Managing subscription tiers, Family Plan invitations and address verification, Student discounts, or partner bundles.
- **Empirical Frequency (Initial Inquiries):** 1,174 (4.35%)
- **Difficulty Rating:** `Moderate`
- **Escalation Policy:** Automated Deflection / RAG Candidate
- **Actionable Resolution:** Explain Family Plan home address rules, provide SheerID student verification link, guide through account settings page.

**Inclusion Criteria:**
- Family Plan invitation issues (invite link expired, members unable to join, address verification mismatch)
- Student discount reverification issues via SheerID or UNiDAYS
- Partner bundle inquiries (Spotify + Hulu + SHOWTIME student bundle activation)
- Upgrading from Free to Premium, changing plan tiers, or general subscription cancellation guidance

**Exclusion Criteria:**
- Direct payment failure or unexpected bank debit (route to billing_subscription_payment)
- Customer unable to log into their personal account (route to account_access_credentials)

**Confusable Boundaries:**
- *vs. `billing_subscription_payment`:* Billing is for charges/refunds/payment cards; Plan Management is for plan rules, invites, addresses, and discount verification.
- *vs. `account_access_credentials`:* If user cannot join family plan because they don't know their password, route to account_access_credentials.

**Representative Examples:**
> "My family member can't accept the Family Plan invite because it says address doesn't match."
> "How do I renew my Student Discount? SheerID says my university documents are invalid."
> "Can I activate the Hulu bundle with my existing Spotify Student account?"
> "How do I cancel my Premium subscription so I don't get renewed next month?"
> "Upgraded to Family plan from individual, do my family members lose their playlists?"

---

### `content_catalog_licensing` — Content Catalog / Licensing

- **Description:** Inquiries regarding song or album availability, greyed-out tracks, regional rights restrictions, or artist release dates.
- **Empirical Frequency (Initial Inquiries):** 1,024 (3.80%)
- **Difficulty Rating:** `Easy`
- **Escalation Policy:** Automated Deflection / RAG Candidate
- **Actionable Resolution:** Explain that music availability depends on artists and rights holders, confirm regional catalog variations, suggest feedback form.

**Inclusion Criteria:**
- Specific songs, albums, or discographies missing or greyed out in the app
- Music unavailable in the customer's geographic country or territory
- Disputes over explicit vs clean song versions or removed podcasts
- Inquiries about when an upcoming album or release will be published on Spotify

**Exclusion Criteria:**
- User's personal local audio files not showing up in Spotify (route to playlist_library_curation)
- All songs failing to play due to audio streaming error (route to playback_streaming_issue)

**Confusable Boundaries:**
- *vs. `playlist_library_curation`:* Catalog is about artist release and rights holder availability; playlist_library is about user's own playlists and library organisation.
- *vs. `playback_streaming_issue`:* If tracks are greyed out, it is licensing; if tracks are white/active but buffer or fail when tapped, it is playback.

**Representative Examples:**
> "Why is Taylor Swift's latest album not available in my country on Spotify?"
> "Half of the songs on this album are greyed out and unplayable, why?"
> "Did you guys remove 'You Shook Me All Night Long'? It was in my favorites yesterday."
> "Can you add the explicit version of this song? Only the radio edit is available."
> "What time will the new release be available in UK time tonight?"

---

### `playlist_library_curation` — Playlist / Library / Curation

- **Description:** Issues managing playlists, library organization, shuffle/repeat playback modes, or curated recommendations.
- **Empirical Frequency (Initial Inquiries):** 2,788 (10.34%)
- **Difficulty Rating:** `Moderate`
- **Escalation Policy:** Automated Deflection / RAG Candidate
- **Actionable Resolution:** Provide playlist recovery tool link (spotify.com/recover-playlists), guide on shuffle cache reset, explain algorithm refresh cycle.

**Inclusion Criteria:**
- Accidentally deleted playlist recovery or missing custom playlists
- Shuffle algorithm malfunctioning (playing same songs repeatedly, not truly shuffling, shuffle turning off)
- Repeat button not working or repeating the wrong track
- Discover Weekly, Release Radar, or Daily Mix not updating or giving irrelevant recommendations
- Local files sync between desktop and mobile library

**Exclusion Criteria:**
- Audio failing to stream or silent playback (route to playback_streaming_issue)
- Licensed tracks missing from catalog globally (route to content_catalog_licensing)

**Confusable Boundaries:**
- *vs. `playback_streaming_issue`:* If the issue is song ordering/repetition (shuffle/repeat), route to playlist_library_curation. If song stops or buffers, route to playback_streaming_issue.
- *vs. `content_catalog_licensing`:* If songs vanished from user's custom playlist because rights were pulled, route to content_catalog_licensing if user inquires about licensing.

**Representative Examples:**
> "I accidentally deleted my favourite playlist with 500 songs, can I restore it?"
> "Shuffle is completely broken, it keeps playing the exact same 10 songs out of 800."
> "Repeat button is lit green but the song doesn't repeat when finished."
> "My Discover Weekly didn't update this Monday like it usually does."
> "How do I change the order of songs in my playlist on the mobile app?"

---

### `app_technical_device` — App Stability / Device Integration

- **Description:** Application crashes, freezes, OS-level performance issues, or external hardware connectivity.
- **Empirical Frequency (Initial Inquiries):** 1,183 (4.39%)
- **Difficulty Rating:** `Easy`
- **Escalation Policy:** Automated Deflection / RAG Candidate
- **Actionable Resolution:** Provide clean reinstallation steps, device reboot instructions, Bluetooth reconnect steps, or confirm OS compatibility.

**Inclusion Criteria:**
- App crashing immediately on startup, freezing, or black screen
- Excessive battery drain, CPU overheating, or storage bloat
- Errors updating or installing Spotify on iOS, Android, Windows, or Mac
- Connectivity issues with external hardware: Bluetooth speakers, CarPlay, Android Auto, Sonos, Apple Watch, PlayStation, Roku, Alexa

**Exclusion Criteria:**
- Streaming playback buffer issues while the app UI remains responsive (route to playback_streaming_issue)
- Download storage limit reached (route to offline_downloads_issue)

**Confusable Boundaries:**
- *vs. `playback_streaming_issue`:* App technical applies when the app itself crashes or freezes; playback applies when the app is running normally but audio fails.
- *vs. `offline_downloads_issue`:* If issue is purely downloading tracks, route to offline_downloads_issue.

**Representative Examples:**
> "Spotify crashes immediately every time I tap the icon on iOS 11."
> "The app is draining 50% of my phone battery in under an hour without even playing music."
> "Spotify Connect can't find my Sonos speakers or PlayStation on the same WiFi."
> "CarPlay screen goes completely black when I open Spotify in my car."
> "The desktop app is completely frozen on Windows 10 and won't respond to clicks."

---

### `unclear_insufficient_context` — Unclear / Insufficient Context

- **Description:** Inquiries that express frustration or request help but lack sufficient technical, account, or behavioral details to diagnose.
- **Empirical Frequency (Initial Inquiries):** 4,464 (16.55%)
- **Difficulty Rating:** `Hard`
- **Escalation Policy:** Automated Deflection / RAG Candidate
- **Actionable Resolution:** Prompt customer with standard clarifying questions: device model, OS version, app version, and specific error symptom.

**Inclusion Criteria:**
- Ultra-short expressions lacking details: 'not working', 'fix this', 'broken', 'help me', 'why?'
- Follow-up fragments viewed without conversational context (e.g. 'iPhone 7', 'sent DM', 'tried that')
- Vague complaints where no specific feature, error, or symptom is mentioned

**Exclusion Criteria:**
- Any inquiry containing even a single concrete diagnostic symptom (e.g., 'crashing' -> app_technical_device, 'won't download' -> offline_downloads_issue)

**Confusable Boundaries:**
- *vs. `All intents`:* Serves as the designated fallback for underspecified messages to prevent hallucinated classification.

**Representative Examples:**
> "It isn't working."
> "Please fix this app."
> "Why is this happening again?"
> "Still waiting for help..."
> "Help me please @SpotifyCares"

---

## 4. Taxonomy Quality & Design Decisions

### Coverage
Over 68% of initial customer inquiries map to one of the 8 technical domain intents. The remaining messages are split between explicit ambiguous/unclear expressions (21%) and general conversational/feature queries (11%).

### Separability
Every intent has explicit mutual exclusion boundaries. For example:
- Streaming audio failure -> `playback_streaming_issue`
- Downloaded file storage failure -> `offline_downloads_issue`
- App process crash/freeze -> `app_technical_device`

### Actionability & Escalation
- **Immediate Human Escalation:** `billing_subscription_payment` (financial transactions, refunds) and `account_access_credentials` (compromised accounts) strictly require private verification and backstage tooling.
- **Self-Service / Automated Troubleshooter:** `playback_streaming_issue`, `offline_downloads_issue`, `playlist_library_curation`, and `app_technical_device` can be reliably resolved via step-by-step troubleshooting articles and deterministic clearing actions.
