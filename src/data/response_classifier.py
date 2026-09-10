"""Heuristic classification of brand responses into functional support categories."""
from typing import Dict, Any, Tuple
import re

RESPONSE_TYPES = [
    "Troubleshooting / Actionable",
    "Clarification / Request for Info",
    "Redirect / DM",
    "Redirect / External Support",
    "Information / Explanation",
    "Generic Acknowledgement",
    "Other / Miscellaneous"
]

# Regex patterns grounded in empirical SpotifyCares response data
DM_PATTERNS = [
    r'\b(send|drop|shoot)\s+(us\s+)?(a\s+)?dm\b',
    r'\bdm\s+us\b',
    r'\bdirect\s+message\b',
    r'\bbackstage\b',
    r'https://t\.co/ldfdzrinat'  # Standard SpotifyCares DM shortlink
]

CLARIFY_PATTERNS = [
    r'\b(what|which)\s+(device|version|os|operating system|model|phone|country)\b',
    r'\b(free\s+or\s+premium|premium\s+or\s+free)\b',
    r'\b(can|could)\s+you\s+(send|give|tell|let\s+us\s+know|share|confirm|provide)\b',
    r'\bdoes\s+(this|it)\s+happen\b',
    r'\bwhat\s+happens\s+when\b',
    r'\bis\s+(this|it)\s+happening\s+on\b',
    r'\bany\s+error\s+(message|code)\b',
    r'\bcan\s+you\s+check\s+if\b',
    r'\bwhat\s+are\s+you\s+seeing\b',
    r'\bdo\s+you\s+recall\s+when\b'
]

TROUBLESHOOT_PATTERNS = [
    r'\b(clean\s+reinstall|reinstalling|re-install|reinstall|uninstall\s+and\s+reinstall)\b',
    r'\b(restart|reboot|power\s+cycle)\s+(your\s+)?(device|phone|app|computer|router|mac|pc)\b',
    r'\b(log\s+out|logging\s+out|log\s+back\s+in|sign\s+out|sign\s+back\s+in)\b',
    r'\b(clear\s+(the\s+)?cache|offline\s+mode|airplane\s+mode|toggle)\b',
    r'\b(update|updating)\s+(the\s+app|spotify|your\s+device|ios|android)\b',
    r'\bhold(ing)?\s+(the\s+)?(sleep/wake|power|volume)\b',
    r'\b(check|turn\s+on|turn\s+off|disable|enable)\s+(the\s+)?(setting|settings|hardware\s+acceleration|offline)\b',
    r'\btry\s+(restarting|reinstalling|logging|clearing|toggling|updating)\b'
]

EXT_REDIRECT_PATTERNS = [
    r'\b(support\.spotify\.com|community\.spotify\.com|spotify\.com/help)\b',
    r'\bcontact\s+(support|touchtunes|apple|google|samsung|playstation|xbox|sonos|roku)\b',
    r'\breach\s+out\s+to\s+(apple|google|samsung|playstation|xbox|sonos|roku|touchtunes)\b',
    r'\bhead\s+over\s+to\s+(our\s+)?community\b'
]

INFO_PATTERNS = [
    r'\b(not\s+possible|currently\s+unavailable|not\s+supported|feature\s+request|pass\s+(this|it|your\s+feedback)\s+on)\b',
    r'\b(rights\s+holder|licensing|availability|depends\s+on\s+the\s+artist)\b',
    r'\b(maintenance\s+mode|discontinued|no\s+longer\s+supported)\b',
    r'\b(student\s+discount|family\s+plan|billing\s+cycle|subscription\s+renews)\b',
    r'\b(albums?|tracks?|songs?)\s+(are|is)\s+currently\s+unavailable\b'
]

ACK_PATTERNS = [
    r'\b(you\'re\s+(very\s+)?welcome|no\s+problem|no\s+worries|glad\s+to\s+(hear|help)|happy\s+to\s+help|anytime|rock\s+on|enjoy\s+the\s+tunes|happy\s+listening|you\s+rock)\b',
    r'\b(thanks\s+for\s+(reaching\s+out|the\s+love|letting\s+us\s+know|confirming)|we\s+appreciate\s+the\s+feedback)\b',
    r'\b(give\s+us\s+a\s+shout|we\'re\s+just\s+a\s+tweet\s+away)\b'
]

# Precompile
COMPILED_DM = [re.compile(p, re.IGNORECASE) for p in DM_PATTERNS]
COMPILED_CLARIFY = [re.compile(p, re.IGNORECASE) for p in CLARIFY_PATTERNS]
COMPILED_TROUBLESHOOT = [re.compile(p, re.IGNORECASE) for p in TROUBLESHOOT_PATTERNS]
COMPILED_EXT = [re.compile(p, re.IGNORECASE) for p in EXT_REDIRECT_PATTERNS]
COMPILED_INFO = [re.compile(p, re.IGNORECASE) for p in INFO_PATTERNS]
COMPILED_ACK = [re.compile(p, re.IGNORECASE) for p in ACK_PATTERNS]


def extract_response_features(text: str) -> Dict[str, bool]:
    """Extract individual functional capability flags from a brand response."""
    text_clean = str(text).lower()
    return {
        "has_dm_redirect": any(p.search(text_clean) for p in COMPILED_DM),
        "has_clarification": any(p.search(text_clean) for p in COMPILED_CLARIFY) or ("?" in text_clean and any(w in text_clean for w in ["what", "which", "could you", "can you", "have you"])),
        "has_troubleshooting": any(p.search(text_clean) for p in COMPILED_TROUBLESHOOT),
        "has_external_redirect": any(p.search(text_clean) for p in COMPILED_EXT),
        "has_info_explanation": any(p.search(text_clean) for p in COMPILED_INFO),
        "has_acknowledgement": any(p.search(text_clean) for p in COMPILED_ACK),
        "has_link": bool(re.search(r'https?://\S+', text_clean))
    }


def classify_response_type(text: str) -> str:
    """Classify brand response into a mutually exclusive functional category.
    
    Priority order reflects actionable utility:
    1. Troubleshooting / Actionable (direct technical steps)
    2. Clarification / Request for Info (diagnostic inquiries)
    3. Redirect / DM (escalation to private support)
    4. Redirect / External Support (links to external guides/partners)
    5. Information / Explanation (licensing/policy/status explanation)
    6. Generic Acknowledgement (social courtesies/thanks)
    7. Other / Miscellaneous (unclassified edge cases)
    """
    feats = extract_response_features(text)
    words = str(text).split()

    if feats["has_troubleshooting"]:
        return "Troubleshooting / Actionable"
    elif feats["has_clarification"] and not feats["has_dm_redirect"]:
        return "Clarification / Request for Info"
    elif feats["has_dm_redirect"]:
        return "Redirect / DM"
    elif feats["has_external_redirect"]:
        return "Redirect / External Support"
    elif feats["has_info_explanation"]:
        return "Information / Explanation"
    elif feats["has_acknowledgement"] and len(words) <= 25 and not feats["has_clarification"]:
        return "Generic Acknowledgement"
    elif feats["has_clarification"]:
        return "Clarification / Request for Info"
    else:
        return "Other / Miscellaneous"
