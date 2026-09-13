"""Machine-readable Intent Taxonomy for SpotifyCares Customer Support Interactions."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import re


@dataclass
class IntentDefinition:
    name: str
    display_name: str
    description: str
    inclusion_rules: List[str]
    exclusion_rules: List[str]
    confusable_intents: Dict[str, str]
    actionability: str
    escalation_recommended: bool
    difficulty: str  # "Easy", "Moderate", "Hard"
    representative_examples: List[str]
    patterns: List[str] = field(default_factory=list)


INTENT_TAXONOMY: Dict[str, IntentDefinition] = {
    "playback_streaming_issue": IntentDefinition(
        name="playback_streaming_issue",
        display_name="Playback / Streaming Issue",
        description="Problems playing, streaming, pausing, or continuing audio playback online.",
        inclusion_rules=[
            "Audio stops playing unexpectedly mid-song or after every track",
            "Songs skip continuously or fail to buffer",
            "Silence, audio distortion, stuttering, or playback error messages during streaming",
            "Crossfade, gapless playback, or web player playback failures"
        ],
        exclusion_rules=[
            "Audio issues occurring strictly when offline or with downloaded tracks (route to offline_downloads_issue)",
            "Shuffle or repeat playback order logic issues (route to playlist_library_curation)",
            "Song greyed out or completely absent from catalog (route to content_catalog_licensing)"
        ],
        confusable_intents={
            "offline_downloads_issue": "Distinguish by online vs offline: offline_downloads applies when songs were saved to device and played without internet.",
            "playlist_library_curation": "Distinguish by playback failure vs track selection: playback_streaming is technical audio failure; playlist_library is shuffle/repeat order.",
            "app_technical_device": "Distinguish by audio vs app process: app_technical_device applies when the whole app crashes or freezes to the OS home screen."
        },
        actionability="Offer audio troubleshooting: toggle hardware acceleration, clear app cache, check streaming quality settings, restart audio daemon.",
        escalation_recommended=False,
        difficulty="Easy",
        representative_examples=[
            "Spotify keeps stopping every few seconds when I try to listen to music.",
            "Music suddenly cuts out after 10 seconds and skips to the next song.",
            "My web player says 'Spotify cannot play this right now' for every single track.",
            "Songs keep buffering and pausing every 30 seconds on high speed WiFi.",
            "Audio is completely silent even though the progress bar is moving."
        ],
        patterns=[
            r'\b(playback|playing|plays|stopping|stops|stopped|buffering|pause|pauses|pausing|skips|skipping|stuttering|static|no sound|cutting out|won\'t play|wont play|can\'t play|cant play|stops playing|every few seconds|gapless|crossfade)\b'
        ]
    ),

    "offline_downloads_issue": IntentDefinition(
        name="offline_downloads_issue",
        display_name="Download / Offline Issue",
        description="Problems downloading songs/albums for offline listening, offline sync failures, or disappeared downloads.",
        inclusion_rules=[
            "Downloaded tracks not playing when device is in offline mode or without internet",
            "Download progress stuck, waiting to download, or failing to complete",
            "Downloaded music suddenly disappearing from the device",
            "Errors regarding the 3-device download limit or 10,000 song limit per device"
        ],
        exclusion_rules=[
            "Online streaming playback buffering (route to playback_streaming_issue)",
            "Local MP3 files imported from computer not syncing (route to playlist_library_curation)"
        ],
        confusable_intents={
            "playback_streaming_issue": "Distinguish by online vs offline context. If user specifies 'offline mode' or 'downloaded', choose offline_downloads_issue.",
            "app_technical_device": "Distinguish by storage/download vs app crash: storage or download sync is offline_downloads_issue."
        },
        actionability="Provide storage check guide, instructions to toggle offline mode off/on, verify device limit, and download over stable WiFi.",
        escalation_recommended=False,
        difficulty="Moderate",
        representative_examples=[
            "My downloaded songs won't play when I turn on offline mode on the train.",
            "All my downloaded playlists got deleted after the recent update.",
            "Downloads are perpetually stuck on 'waiting to download' on my Android phone.",
            "Getting an error that I've reached the device limit for downloads, but I only use two phones.",
            "Why do my offline songs keep undownloading themselves every few days?"
        ],
        patterns=[
            r'\b(download|downloads|downloading|downloaded|offline|offline mode|offline songs|storage|saved music|sync offline|listen offline|without wifi|no internet)\b'
        ]
    ),

    "account_access_credentials": IntentDefinition(
        name="account_access_credentials",
        display_name="Account Access / Credentials",
        description="Inability to log in, password reset issues, compromised/hacked accounts, or unexpected credential changes.",
        inclusion_rules=[
            "Forgotten username, password, or password reset email not arriving",
            "Account hacked, email address changed without user authorization",
            "Facebook login disconnection or login loop on mobile/desktop",
            "Error messages: 'Incorrect username or password' or 'Account locked'"
        ],
        exclusion_rules=[
            "Customer can log in fine but wants to change billing details (route to billing_subscription_payment)",
            "Customer can log in but family invitation verification failed (route to plan_management_discount)"
        ],
        confusable_intents={
            "billing_subscription_payment": "If customer is locked out AND asking about an unexpected charge, prioritize account_access if security compromise is suspected.",
            "plan_management_discount": "Family invitation errors where user can still log into their own account belong to plan_management_discount."
        },
        actionability="Direct to password reset link, guide through email change recovery, or immediate human escalation for compromised accounts.",
        escalation_recommended=True,
        difficulty="Moderate",
        representative_examples=[
            "Can't log into my account, says my password is wrong and reset email never arrives.",
            "Someone hacked my Spotify account and changed the email address to an unknown domain!",
            "I'm locked out of my account because I deactivated my old Facebook account.",
            "Keep getting logged out every time I close the app and have to re-enter credentials.",
            "I have two accounts accidentally and need help accessing my original one."
        ],
        patterns=[
            r'\b(log in|logging in|login|logged out|log out|password|reset password|email address|change email|username|hacked|compromised|can\'t access|cant access|locked out|verification code|two factor|facebook login)\b'
        ]
    ),

    "billing_subscription_payment": IntentDefinition(
        name="billing_subscription_payment",
        display_name="Billing / Payment / Charges",
        description="Payment processing failures, unwanted or duplicate charges, refund requests, or payment method update issues.",
        inclusion_rules=[
            "Customer was charged twice in the same billing cycle (double charge)",
            "Charged after canceling subscription or charged on a free account",
            "Credit card or PayPal payment declined / update payment method fails",
            "Requests for refund or invoice/receipt dispute"
        ],
        exclusion_rules=[
            "Questions about how to cancel or downgrade without active financial dispute (route to plan_management_discount)",
            "Student or Family Plan discount verification questions (route to plan_management_discount)"
        ],
        confusable_intents={
            "plan_management_discount": "If user mentions payment/money/charge/refund, assign billing_subscription_payment. If purely asking about student discount eligibility, assign plan_management_discount."
        },
        actionability="Direct to payment update page, verify billing history backstage via DM, or escalate to financial support for refunds.",
        escalation_recommended=True,
        difficulty="Easy",
        representative_examples=[
            "I've been charged twice for Spotify Premium this month! Need a refund.",
            "Why did Spotify take $9.99 from my account when I cancelled last week?",
            "My card was charged but my account still says Spotify Free.",
            "Trying to update my payment method with my new credit card but it keeps getting rejected.",
            "Where can I find my monthly VAT invoice/receipt for my premium subscription?"
        ],
        patterns=[
            r'\b(charged|charge|charges|charging|bill|billed|billing|payment|paid|pay|card|refund|receipt|invoice|bank|subscription fee|double charge|charged twice|overcharged|apple pay|paypal|money)\b'
        ]
    ),

    "plan_management_discount": IntentDefinition(
        name="plan_management_discount",
        display_name="Plan Management / Student & Family",
        description="Managing subscription tiers, Family Plan invitations and address verification, Student discounts, or partner bundles.",
        inclusion_rules=[
            "Family Plan invitation issues (invite link expired, members unable to join, address verification mismatch)",
            "Student discount reverification issues via SheerID or UNiDAYS",
            "Partner bundle inquiries (Spotify + Hulu + SHOWTIME student bundle activation)",
            "Upgrading from Free to Premium, changing plan tiers, or general subscription cancellation guidance"
        ],
        exclusion_rules=[
            "Direct payment failure or unexpected bank debit (route to billing_subscription_payment)",
            "Customer unable to log into their personal account (route to account_access_credentials)"
        ],
        confusable_intents={
            "billing_subscription_payment": "Billing is for charges/refunds/payment cards; Plan Management is for plan rules, invites, addresses, and discount verification.",
            "account_access_credentials": "If user cannot join family plan because they don't know their password, route to account_access_credentials."
        },
        actionability="Explain Family Plan home address rules, provide SheerID student verification link, guide through account settings page.",
        escalation_recommended=False,
        difficulty="Moderate",
        representative_examples=[
            "My family member can't accept the Family Plan invite because it says address doesn't match.",
            "How do I renew my Student Discount? SheerID says my university documents are invalid.",
            "Can I activate the Hulu bundle with my existing Spotify Student account?",
            "How do I cancel my Premium subscription so I don't get renewed next month?",
            "Upgraded to Family plan from individual, do my family members lose their playlists?"
        ],
        patterns=[
            r'\b(family plan|family account|premium family|invite|invitation|home address|same address|student discount|student premium|sheerid|unidays|hulu|showtime|upgrade to premium|downgrade|cancel subscription|cancel premium|cancelling)\b'
        ]
    ),

    "content_catalog_licensing": IntentDefinition(
        name="content_catalog_licensing",
        display_name="Content Catalog / Licensing",
        description="Inquiries regarding song or album availability, greyed-out tracks, regional rights restrictions, or artist release dates.",
        inclusion_rules=[
            "Specific songs, albums, or discographies missing or greyed out in the app",
            "Music unavailable in the customer's geographic country or territory",
            "Disputes over explicit vs clean song versions or removed podcasts",
            "Inquiries about when an upcoming album or release will be published on Spotify"
        ],
        exclusion_rules=[
            "User's personal local audio files not showing up in Spotify (route to playlist_library_curation)",
            "All songs failing to play due to audio streaming error (route to playback_streaming_issue)"
        ],
        confusable_intents={
            "playlist_library_curation": "Catalog is about artist release and rights holder availability; playlist_library is about user's own playlists and library organisation.",
            "playback_streaming_issue": "If tracks are greyed out, it is licensing; if tracks are white/active but buffer or fail when tapped, it is playback."
        },
        actionability="Explain that music availability depends on artists and rights holders, confirm regional catalog variations, suggest feedback form.",
        escalation_recommended=False,
        difficulty="Easy",
        representative_examples=[
            "Why is Taylor Swift's latest album not available in my country on Spotify?",
            "Half of the songs on this album are greyed out and unplayable, why?",
            "Did you guys remove 'You Shook Me All Night Long'? It was in my favorites yesterday.",
            "Can you add the explicit version of this song? Only the radio edit is available.",
            "What time will the new release be available in UK time tonight?"
        ],
        patterns=[
            r'\b(song missing|album missing|songs missing|not available|unavailable|removed|greyed out|grayed out|licensing|rights|why did you remove|why is .* not on spotify|region|country|where is the album|explicit version|clean version|artist)\b'
        ]
    ),

    "playlist_library_curation": IntentDefinition(
        name="playlist_library_curation",
        display_name="Playlist / Library / Curation",
        description="Issues managing playlists, library organization, shuffle/repeat playback modes, or curated recommendations.",
        inclusion_rules=[
            "Accidentally deleted playlist recovery or missing custom playlists",
            "Shuffle algorithm malfunctioning (playing same songs repeatedly, not truly shuffling, shuffle turning off)",
            "Repeat button not working or repeating the wrong track",
            "Discover Weekly, Release Radar, or Daily Mix not updating or giving irrelevant recommendations",
            "Local files sync between desktop and mobile library"
        ],
        exclusion_rules=[
            "Audio failing to stream or silent playback (route to playback_streaming_issue)",
            "Licensed tracks missing from catalog globally (route to content_catalog_licensing)"
        ],
        confusable_intents={
            "playback_streaming_issue": "If the issue is song ordering/repetition (shuffle/repeat), route to playlist_library_curation. If song stops or buffers, route to playback_streaming_issue.",
            "content_catalog_licensing": "If songs vanished from user's custom playlist because rights were pulled, route to content_catalog_licensing if user inquires about licensing."
        },
        actionability="Provide playlist recovery tool link (spotify.com/recover-playlists), guide on shuffle cache reset, explain algorithm refresh cycle.",
        escalation_recommended=False,
        difficulty="Moderate",
        representative_examples=[
            "I accidentally deleted my favourite playlist with 500 songs, can I restore it?",
            "Shuffle is completely broken, it keeps playing the exact same 10 songs out of 800.",
            "Repeat button is lit green but the song doesn't repeat when finished.",
            "My Discover Weekly didn't update this Monday like it usually does.",
            "How do I change the order of songs in my playlist on the mobile app?"
        ],
        patterns=[
            r'\b(playlist|playlists|shuffle|shuffle button|repeat|repeat button|discover weekly|release radar|daily mix|library|saved songs|songs disappeared|reorder|queue|local files|my music|album order)\b'
        ]
    ),

    "app_technical_device": IntentDefinition(
        name="app_technical_device",
        display_name="App Stability / Device Integration",
        description="Application crashes, freezes, OS-level performance issues, or external hardware connectivity.",
        inclusion_rules=[
            "App crashing immediately on startup, freezing, or black screen",
            "Excessive battery drain, CPU overheating, or storage bloat",
            "Errors updating or installing Spotify on iOS, Android, Windows, or Mac",
            "Connectivity issues with external hardware: Bluetooth speakers, CarPlay, Android Auto, Sonos, Apple Watch, PlayStation, Roku, Alexa"
        ],
        exclusion_rules=[
            "Streaming playback buffer issues while the app UI remains responsive (route to playback_streaming_issue)",
            "Download storage limit reached (route to offline_downloads_issue)"
        ],
        confusable_intents={
            "playback_streaming_issue": "App technical applies when the app itself crashes or freezes; playback applies when the app is running normally but audio fails.",
            "offline_downloads_issue": "If issue is purely downloading tracks, route to offline_downloads_issue."
        },
        actionability="Provide clean reinstallation steps, device reboot instructions, Bluetooth reconnect steps, or confirm OS compatibility.",
        escalation_recommended=False,
        difficulty="Easy",
        representative_examples=[
            "Spotify crashes immediately every time I tap the icon on iOS 11.",
            "The app is draining 50% of my phone battery in under an hour without even playing music.",
            "Spotify Connect can't find my Sonos speakers or PlayStation on the same WiFi.",
            "CarPlay screen goes completely black when I open Spotify in my car.",
            "The desktop app is completely frozen on Windows 10 and won't respond to clicks."
        ],
        patterns=[
            r'\b(crash|crashes|crashing|freeze|freezes|freezing|black screen|blank screen|force close|won\'t open|wont open|not opening|install|uninstall|update|battery|cpu|apple watch|carplay|bluetooth|sonos|playstation|ps4|alexa|roku|desktop app|windows phone)\b'
        ]
    ),

    "unclear_insufficient_context": IntentDefinition(
        name="unclear_insufficient_context",
        display_name="Unclear / Insufficient Context",
        description="Inquiries that express frustration or request help but lack sufficient technical, account, or behavioral details to diagnose.",
        inclusion_rules=[
            "Ultra-short expressions lacking details: 'not working', 'fix this', 'broken', 'help me', 'why?'",
            "Follow-up fragments viewed without conversational context (e.g. 'iPhone 7', 'sent DM', 'tried that')",
            "Vague complaints where no specific feature, error, or symptom is mentioned"
        ],
        exclusion_rules=[
            "Any inquiry containing even a single concrete diagnostic symptom (e.g., 'crashing' -> app_technical_device, 'won't download' -> offline_downloads_issue)"
        ],
        confusable_intents={
            "All intents": "Serves as the designated fallback for underspecified messages to prevent hallucinated classification."
        },
        actionability="Prompt customer with standard clarifying questions: device model, OS version, app version, and specific error symptom.",
        escalation_recommended=False,
        difficulty="Hard",
        representative_examples=[
            "It isn't working.",
            "Please fix this app.",
            "Why is this happening again?",
            "Still waiting for help...",
            "Help me please @SpotifyCares"
        ],
        patterns=[
            r'\b(not working|doesn\'t work|doesnt work|broken|help|fix|why|same issue|still waiting|please help)\b'
        ]
    ),

    "other_miscellaneous": IntentDefinition(
        name="other_miscellaneous",
        display_name="Other / Miscellaneous",
        description="General conversational banter, social praise, feature requests that do not fit the technical support domains, or uncategorizable edge cases that are not genuinely ambiguous.",
        inclusion_rules=[
            "Social praise, thank-you messages, or general encouragement with no actionable support problem",
            "Feature requests or suggestions that do not report a current defect",
            "General conversational filler or non-support greetings",
            "Messages that do not fit any defined technical support domain and are not genuinely underspecified"
        ],
        exclusion_rules=[
            "Any message containing a concrete technical symptom (e.g., 'crashing' -> app_technical_device, 'won't download' -> offline_downloads_issue)",
            "Any message that is genuinely underspecified despite context (route to unclear_insufficient_context)",
            "Any financial dispute or billing question (route to billing_subscription_payment)"
        ],
        confusable_intents={
            "unclear_insufficient_context": "Use other_miscellaneous when the message is clearly conversational/praise/suggestion but not underspecified. Use unclear_insufficient_context when the message lacks diagnostic detail needed to route support.",
            "content_catalog_licensing": "Feature requests about adding artists/albums are typically other_miscellaneous unless they are about missing/licensed tracks in the catalog."
        },
        actionability="No immediate technical action required. Route to general support, feedback, or social channels as appropriate.",
        escalation_recommended=False,
        difficulty="Easy",
        representative_examples=[
            "Love the new update!",
            "Thanks so much for the quick response.",
            "Can you add this song to the catalog?",
            "Spotify is the best app ever.",
            "Congratulations on the feature release."
        ],
        patterns=[
            r'\b(love|thanks|thank you|congratulations|great|best app|awesome|feature request|suggestion|keep it up|well done|nice work|hope you enjoy)\b'
        ]
    )
}

INTENT_NAMES = list(INTENT_TAXONOMY.keys())

# Precompile regexes
_COMPILED_INTENT_PATTERNS = {
    name: [re.compile(p, re.IGNORECASE) for p in defn.patterns]
    for name, defn in INTENT_TAXONOMY.items()
}


def get_intent(name: str) -> Optional[IntentDefinition]:
    """Retrieve intent definition object by name."""
    return INTENT_TAXONOMY.get(name)


def detect_candidate_intents(text: str) -> List[str]:
    """Return all matching candidate intents for a given text."""
    text_clean = str(text).lower()
    matched = []
    for name in INTENT_NAMES:
        if name == "unclear_insufficient_context":
            continue
        patterns = _COMPILED_INTENT_PATTERNS[name]
        if any(p.search(text_clean) for p in patterns):
            matched.append(name)
    return matched


def classify_customer_intent(text: str, context: str = "") -> str:
    """Heuristically assign the single primary intent to a customer message.
    
    If context is provided, context is analyzed when text itself is underspecified.
    """
    text_clean = str(text).lower()
    matches = detect_candidate_intents(text_clean)

    if not matches:
        # Check context if available
        if context:
            ctx_matches = detect_candidate_intents(context)
            if ctx_matches:
                return ctx_matches[0]
        
        words = text_clean.split()
        if len(words) <= 6 or any(p.search(text_clean) for p in _COMPILED_INTENT_PATTERNS["unclear_insufficient_context"]):
            return "unclear_insufficient_context"
        return "other_miscellaneous"

    if len(matches) == 1:
        return matches[0]

    # Multi-intent tie-breaking rules based on business priority and domain specificity:
    # 1. Financial urgency takes precedence
    if "billing_subscription_payment" in matches and ("plan_management_discount" in matches or "account_access_credentials" in matches):
        if any(w in text_clean for w in ["charged", "charge", "refund", "money", "bank", "card", "receipt", "fee", "twice"]):
            return "billing_subscription_payment"

    # 2. Offline / download specificity over generic playback
    if "offline_downloads_issue" in matches and "playback_streaming_issue" in matches:
        return "offline_downloads_issue"

    # 3. Security / login credentials over plan management
    if "account_access_credentials" in matches and "plan_management_discount" in matches:
        if any(w in text_clean for w in ["password", "email", "hacked", "log in", "login"]):
            return "account_access_credentials"

    # 4. Playlist shuffle/repeat over generic streaming
    if "playlist_library_curation" in matches and "playback_streaming_issue" in matches:
        if any(w in text_clean for w in ["shuffle", "repeat", "playlist", "library"]):
            return "playlist_library_curation"

    return matches[0]
