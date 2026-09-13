"""Deterministic machine (non-human, non-LLM-by-default) reply-quality scorer.

Purpose: time-saving DIAGNOSTIC artifact when genuine human calibration and/or
the LLM API are unavailable. This is explicitly NOT:
  - human labeling (no human_* columns, no human_confirmed anywhere),
  - an LLM judge (no API call happens in this module),
  - a substitute for human-vs-judge agreement (never compute agreement from it).

Method ``deterministic_heuristic_v1`` (crude, fully transparent, arbitrary bins):
  All dimensions use the 0-3 rubric-v1 scales so outputs are shape-compatible
  with the judge schema, but every threshold below is an admittedly arbitrary
  diagnostic cutoff, documented here and recorded per row in ``machine_reason``.

  relevance:      word-F1(customer, reply): >=0.35->3, >=0.20->2, >=0.08->1, else 0
  groundedness:   word-recall(reply, evidence): >=0.80->3, >=0.50->2, >=0.25->1, else 0
                  (extractive top-1 replies therefore score 3 by construction)
  helpfulness:    reply has '?' and <12 words -> 1 (bare question);
                  >=12 words + action verb/DM mention -> 3 if groundedness>=2 else 2;
                  >=12 words -> 2; >=6 words -> 1; else 0
  appropriateness: profanity marker in reply -> 1; <4 words -> 2; else 3
  unsupported_claims (INVERTED like the rubric): URL in reply absent from
                  evidence -> 2; guarantee/refund language in reply absent from
                  evidence -> 1; else 0
  overall_score:  clip(round(mean(first four)) - (1 if unsupported>=2 else 0), 0, 3)

Inputs are ONLY the machine-observable texts (customer/reply/evidence). The
function signature provably accepts no human scores and reads no golden files.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

import pandas as pd

METHOD = "deterministic_heuristic_v1"
RUBRIC_TARGET = "v1-scales (NOT LLM-scored)"
SOURCE_LOCAL = "deterministic_heuristic_v1 (no LLM call; API unavailable or bypassed)"

MACHINE_DIMS = [
    "machine_relevance",
    "machine_helpfulness",
    "machine_groundedness",
    "machine_appropriateness",
    "machine_unsupported_claims",
    "machine_overall",
]

_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "it", "this", "that", "you", "your", "we", "us", "me", "my", "i", "at",
    "with", "please", "hi", "hey", "hello", "thanks", "thank", "so", "do",
    "does", "did", "not", "no", "just", "can", "get", "got", "be", "was",
}

_PROFANITY = {"fuck", "shit", "damn", "trash", "stupid", "idiot", "hate",
              "sucks", "dumb", "hell"}

_ACTION_VERBS = {"try", "check", "update", "reinstall", "restart", "clear",
                 "toggle", "verify", "follow", "send", "dm", "settings",
                 "reboot", "refresh", "install"}

_GUARANTEE_WORDS = {"refund", "refunded", "guarantee", "promise", "promised",
                    "will fix", "fixed", "definitely", "assure", "assured",
                    "compensat"}

_URL_RE = re.compile(r"https?://|www\.|t\.co\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> Set[str]:
    return {w for w in _WORD_RE.findall(str(text or "").lower())} - _STOPWORDS


def _f1(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    prec = inter / len(b)
    rec = inter / len(a)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _recall(a: Set[str], b: Set[str]) -> float:
    """Fraction of A's words also present in B."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def heuristic_reply_scores(
    customer_message: str,
    generated_reply: str,
    retrieved_evidence: str,
) -> Dict[str, object]:
    """Score a reply deterministically. No human labels, no API, no file reads."""
    cust, reply, evid = _tokens(customer_message), _tokens(generated_reply), _tokens(retrieved_evidence)
    reply_raw = str(generated_reply or "")
    evid_raw = str(retrieved_evidence or "")
    n_words = len(reply_raw.split())

    f1_cr = _f1(cust, reply)
    relevance = 3 if f1_cr >= 0.35 else (2 if f1_cr >= 0.20 else (1 if f1_cr >= 0.08 else 0))

    ov_re = _recall(reply, evid)
    groundedness = 3 if ov_re >= 0.80 else (2 if ov_re >= 0.50 else (1 if ov_re >= 0.25 else 0))

    has_q = "?" in reply_raw
    has_action = bool(reply & _ACTION_VERBS) or "dm" in reply_raw.lower()
    if has_q and n_words < 12:
        helpfulness = 1
    elif n_words >= 12 and has_action:
        helpfulness = 3 if groundedness >= 2 else 2
    elif n_words >= 12:
        helpfulness = 2
    elif n_words >= 6:
        helpfulness = 1
    else:
        helpfulness = 0

    appropriateness = 1 if (reply & _PROFANITY) else (2 if n_words < 4 else 3)

    reply_urls = set(_URL_RE.findall(reply_raw))
    evid_urls = set(_URL_RE.findall(evid_raw))
    extra_urls = {u.lower() for u in reply_urls} - {u.lower() for u in evid_urls}
    guar_extra = {g for g in _GUARANTEE_WORDS if g in reply_raw.lower()} - {
        g for g in _GUARANTEE_WORDS if g in evid_raw.lower()}
    if extra_urls:
        unsupported = 2
        uc_note = f"extra-url({sorted(extra_urls)[0]})"
    elif guar_extra:
        unsupported = 1
        uc_note = f"extra-guarantee({sorted(guar_extra)[0]})"
    else:
        unsupported = 0
        uc_note = "no-extra-claims"

    overall = int(round((relevance + helpfulness + groundedness + appropriateness) / 4.0))
    if unsupported >= 2:
        overall -= 1
    overall = max(0, min(3, overall))

    reason = (f"rel_f1={f1_cr:.2f}->R{relevance}; ov_re={ov_re:.2f}->G{groundedness}; "
              f"len={n_words}{'q' if has_q else ''}{'act' if has_action else ''}->H{helpfulness}; "
              f"{'profanity' if (reply & _PROFANITY) else 'clean'}->A{appropriateness}; "
              f"{uc_note}->U{unsupported}; overall={overall}")
    return {
        "machine_relevance": relevance,
        "machine_helpfulness": helpfulness,
        "machine_groundedness": groundedness,
        "machine_appropriateness": appropriateness,
        "machine_unsupported_claims": unsupported,
        "machine_overall": overall,
        "machine_reason": reason,
    }


def score_inputs_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Score every row of a judge-inputs frame with the deterministic heuristic.

    Adds machine_* columns + machine_reason + method/source(SOURCE_LOCAL).
    Pure function of the input texts; reads no other files.
    """
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        h = heuristic_reply_scores(str(r["customer_message"]), str(r["generated_reply"]),
                                   str(r["retrieved_historical_response"]))
        rows.append({"pair_id": str(r["pair_id"]), "baseline": str(r["baseline"]),
                     **h, "method": METHOD, "source": SOURCE_LOCAL})
    return pd.DataFrame(rows)


def summarize_machine_baseline(scored: pd.DataFrame, baseline: str) -> Dict[str, Dict[str, object]]:
    """Per-dimension mean + distribution for one baseline (machine dims only)."""
    sub = scored[scored["baseline"] == baseline]
    dims: Dict[str, Dict[str, object]] = {}
    for dim in MACHINE_DIMS:
        vals = sub[dim].astype(int).tolist()
        dist = {str(k): int(sum(1 for v in vals if v == k)) for k in range(4)}
        dims[dim] = {"mean": round(float(sum(vals) / len(vals)), 4) if vals else 0.0,
                     "n": len(vals), "distribution": dist}
    return dims
