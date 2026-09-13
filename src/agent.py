"""Phase 4+: Leakage-safe AI Support Agent for SpotifyCares.

Architecture:
    INPUT (customer message + context)
      -> Intent classifier (TF-IDF kNN over weak heuristic labels; keyword fallback)
      -> Retrieval (TF-IDF cosine over the 42,770 leakage-free training pool, top-k)
      -> Evidence selection (top-k with pair_id + similarity + texts)
      -> Response generation (extractive: top-1 historical response; hallucination
         impossible by construction. Optional LLM generator, fail-closed without key.)
      -> Escalation policy (transparent rules)
      -> Final structured output

Frozen constraints:
  - NEVER trains on the 150 reviewed golden examples.
  - NEVER retrieves from any of the 322 golden pair_ids.
  - Runtime assertion: retrieved_pair_ids ∩ golden_pair_ids == empty (always).
  - Escalation is a POLICY output. No "escalation accuracy" is reported because
    no human escalation ground truth exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.intents.taxonomy import (
    INTENT_NAMES,
    INTENT_TAXONOMY,
    classify_customer_intent,
    detect_candidate_intents,
)

ROOT = Path(__file__).resolve().parent.parent

RETRIEVAL_TOP_K = 3
KNN_K = 5
LOW_CONFIDENCE_THRESHOLD = 0.15
AMBIGUITY_MARGIN = 0.05
SECURITY_FINANCIAL_INTENTS = {"account_access_credentials", "billing_subscription_payment"}

CLARIFICATION_REPLY = (
    "Thanks for reaching out! Could you share a few more details so we can help "
    "— e.g. your device model, the Spotify app version, and exactly what happens "
    "when the issue occurs? You can also send us a DM and we'll take a closer look."
)


def _doc_text(message: str, context: str) -> str:
    msg = str(message or "").strip()
    ctx = str(context or "").strip().lower()
    ctx = "" if ctx in ("", "nan", "none") else str(context).strip()
    return (msg + " " + ctx).strip()


@dataclass
class Evidence:
    pair_id: str
    similarity: float
    customer_message: str
    brand_response: str
    customer_primary_intent: str
    has_dm_redirect: bool = False


class TfidfRetriever:
    """TF-IDF cosine retriever over the leakage-free training pool.

    Deterministic: ties broken by (similarity desc, pair_id asc).
    """

    def __init__(
        self,
        training_pool: pd.DataFrame,
        golden_pair_ids: Optional[Set[str]] = None,
        top_k: int = RETRIEVAL_TOP_K,
    ):
        self.golden_pair_ids: Set[str] = set(str(p) for p in (golden_pair_ids or []))
        pool_ids = set(training_pool["pair_id"].astype(str))
        overlap = pool_ids & self.golden_pair_ids
        assert len(overlap) == 0, (
            f"LEAKAGE ASSERTION FAILED: {len(overlap)} golden pair_ids in retriever pool!"
        )
        self.top_k = top_k
        self.pool = training_pool.reset_index(drop=True).copy()
        docs = [
            _doc_text(m, c)
            for m, c in zip(
                self.pool["customer_message"].fillna("").astype(str),
                self.pool["context_before_customer"].fillna("").astype(str),
            )
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2,
            max_features=20000, sublinear_tf=True,
        )
        self.doc_matrix = self.vectorizer.fit_transform(docs)
        # Precomputed pair_id ranks for deterministic (-similarity, pair_id) ordering
        # via vectorized lexsort (avoids O(n log n) Python-level sorts per query).
        pair_ids = self.pool["pair_id"].astype(str).to_numpy()
        self._pair_ids = pair_ids
        self._pair_rank = np.empty(len(pair_ids), dtype=np.int64)
        self._pair_rank[np.argsort(pair_ids, kind="stable")] = np.arange(len(pair_ids))

    def retrieve(
        self,
        message: str,
        context: str = "",
        top_k: Optional[int] = None,
        intent_filter: Optional[str] = None,
    ) -> List[Evidence]:
        """Return top-k evidence. Always asserts zero golden overlap."""
        k = top_k or self.top_k
        q = self.vectorizer.transform([_doc_text(message, context)])
        sims = np.asarray(linear_kernel(q, self.doc_matrix)).ravel()
        # Deterministic order: similarity desc, pair_id asc (vectorized lexsort).
        order = np.lexsort((self._pair_rank, -sims))
        out: List[Evidence] = []
        for i in order:
            if len(out) >= k and intent_filter is None:
                break
            row = self.pool.iloc[int(i)]
            if intent_filter and str(row.get("customer_primary_intent", "")) != intent_filter:
                continue
            out.append(Evidence(
                pair_id=str(row["pair_id"]),
                similarity=round(float(sims[int(i)]), 4),
                customer_message=str(row["customer_message"]),
                brand_response=str(row["brand_response"]),
                customer_primary_intent=str(row.get("customer_primary_intent", "")),
                has_dm_redirect=bool(row.get("has_dm_redirect", False)),
            ))
            if len(out) >= k:
                break
        retrieved_ids = {e.pair_id for e in out}
        assert retrieved_ids.isdisjoint(self.golden_pair_ids), (
            f"LEAKAGE DETECTED: retrieved golden pair_ids {retrieved_ids & self.golden_pair_ids}!"
        )
        return out


class SupportAgent:
    """Leakage-safe support agent: kNN classify -> TF-IDF retrieve -> respond -> escalate."""

    def __init__(
        self,
        training_pool: pd.DataFrame,
        golden_pair_ids: Optional[Set[str]] = None,
        top_k: int = RETRIEVAL_TOP_K,
        knn_k: int = KNN_K,
        use_llm: bool = False,
    ):
        self.golden_pair_ids = set(str(p) for p in (golden_pair_ids or []))
        self.top_k = top_k
        self.knn_k = knn_k
        self.use_llm = use_llm
        self.retriever = TfidfRetriever(training_pool, golden_pair_ids, top_k=top_k)
        self.is_fitted = True

    # -- classification ----------------------------------------------------
    def classify(self, message: str, context: str = "") -> Dict[str, Any]:
        """TF-IDF kNN (similarity-weighted vote) over weak heuristic labels.

        Falls back to the keyword taxonomy classifier when all similarities
        are zero. Tie-breaks follow canonical taxonomy order (deterministic).
        """
        cands = self.retriever.retrieve(message, context, top_k=self.knn_k)
        if not cands or all(c.similarity <= 0 for c in cands):
            intent = classify_customer_intent(str(message), context=str(context or ""))
            return {"predicted_intent": intent, "method": "keyword_fallback",
                    "retrieval_confidence": 0.0}
        votes: Dict[str, float] = {}
        for c in cands:
            votes[c.customer_primary_intent] = votes.get(c.customer_primary_intent, 0.0) + c.similarity
        best = max(votes.values())
        tied = [i for i, v in votes.items() if v == best]
        tied.sort(key=lambda i: INTENT_NAMES.index(i) if i in INTENT_NAMES else 999)
        return {"predicted_intent": tied[0], "method": "tfidf_knn",
                "retrieval_confidence": round(cands[0].similarity, 4)}

    def predict(self, messages: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        contexts = contexts or [""] * len(messages)
        return [self.classify(m, c)["predicted_intent"] for m, c in zip(messages, contexts)]

    # -- response generation ------------------------------------------------
    def _extractive_reply(self, evidence: List[Evidence]) -> str:
        if not evidence:
            return CLARIFICATION_REPLY
        top = evidence[0]
        if top.similarity < LOW_CONFIDENCE_THRESHOLD:
            return CLARIFICATION_REPLY
        return top.brand_response

    def _llm_reply(self, message: str, context: str, intent: str,
                   evidence: List[Evidence]) -> str:
        """Optional LLM generator: grounded, fail-closed without API key."""
        import os
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Cannot generate LLM agent reply. "
                "Use use_llm=False for the extractive grounded reply."
            )
        from src.judge import get_judge_config
        get_judge_config()  # raises clearly if misconfigured
        import json
        import urllib.request
        cfg = get_judge_config()
        model = os.environ.get("AGENT_MODEL", cfg["model"])
        ev_text = "\n".join(f"- ({e.pair_id}, intent={e.customer_primary_intent}): {e.brand_response}"
                            for e in evidence[: self.top_k])
        prompt = (
            "You are SpotifyCares support. Draft ONE concise customer-support reply.\n"
            f"Predicted intent: {intent}\n"
            f"Customer message: {message}\n"
            f"Context: {(context or '').strip() or '(none)'}\n"
            "Historical evidence (ground ONLY in this; do not invent policies, refunds, "
            f"guarantees, URLs, or fixes):\n{ev_text}\n"
            "Rules: no invented policies/refunds/URLs; no claim the issue is fixed; "
            "if evidence is insufficient, ask a clarifying question or suggest a DM. "
            "Return ONLY the reply text."
        )
        body = {"model": model, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            cfg["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {cfg['api_key']}"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        return str(payload["choices"][0]["message"]["content"]).strip()

    # -- escalation ----------------------------------------------------------
    def decide_escalation(self, predicted_intent: str, evidence: List[Evidence],
                          candidate_intents: Optional[List[str]] = None) -> Dict[str, Any]:
        top_sim = evidence[0].similarity if evidence else 0.0
        if not evidence or top_sim < LOW_CONFIDENCE_THRESHOLD:
            return {"decision": "escalate",
                    "reason": f"Retrieval confidence too low (top similarity {top_sim:.3f}); "
                              "insufficient evidence to answer safely.",
                    "confidence": round(top_sim, 3)}
        if predicted_intent in SECURITY_FINANCIAL_INTENTS:
            disp = INTENT_TAXONOMY[predicted_intent].display_name
            return {"decision": "escalate",
                    "reason": f"Intent '{predicted_intent}' ({disp}) is security/financial "
                              "sensitive and requires human review.",
                    "confidence": round(top_sim, 3)}
        if predicted_intent == "unclear_insufficient_context":
            return {"decision": "escalate",
                    "reason": "Message lacks sufficient context to diagnose; needs human "
                      "clarification.",
                    "confidence": round(top_sim, 3)}
        if candidate_intents and len(candidate_intents) > 1 and len(evidence) > 1:
            e0, e1 = evidence[0], evidence[1]
            if (e0.customer_primary_intent != e1.customer_primary_intent
                    and (e0.similarity - e1.similarity) < AMBIGUITY_MARGIN):
                return {"decision": "escalate",
                        "reason": f"Ambiguous between '{e0.customer_primary_intent}' and "
                                  f"'{e1.customer_primary_intent}' (similarity margin "
                                  f"{e0.similarity - e1.similarity:.3f}); needs human triage.",
                        "confidence": round(top_sim, 3)}
        return {"decision": "auto_handle",
                "reason": f"Intent '{predicted_intent}' supported by {len(evidence)} historical "
                          f"case(s) with top similarity {top_sim:.3f}.",
                "confidence": round(top_sim, 3)}

    # -- end-to-end -----------------------------------------------------------
    def handle(self, message: str, context: str = "") -> Dict[str, Any]:
        cls = self.classify(message, context)
        intent = cls["predicted_intent"]
        evidence = self.retriever.retrieve(message, context, top_k=self.top_k)
        assert {e.pair_id for e in evidence}.isdisjoint(self.golden_pair_ids)
        if self.use_llm:
            reply = self._llm_reply(message, context, intent, evidence)
        else:
            reply = self._extractive_reply(evidence)
        candidates = detect_candidate_intents(str(message).lower())
        esc = self.decide_escalation(intent, evidence, candidates)
        assert esc["decision"] in ("auto_handle", "escalate")
        return {
            "predicted_intent": intent,
            "classification_method": cls["method"],
            "retrieval_confidence": cls["retrieval_confidence"],
            "retrieved_evidence": [
                {"pair_id": e.pair_id, "similarity": e.similarity,
                 "customer_message": e.customer_message,
                 "brand_response": e.brand_response,
                 "customer_primary_intent": e.customer_primary_intent,
                 "has_dm_redirect": e.has_dm_redirect} for e in evidence
            ],
            "generated_reply": reply,
            "escalation": esc,
            "leakage_assertion": "PASSED (retrieved_pair_ids ∩ golden_pair_ids == empty)",
        }

    def reply(self, messages: List[str], contexts: Optional[List[str]] = None) -> List[str]:
        contexts = contexts or [""] * len(messages)
        return [self.handle(m, c)["generated_reply"] for m, c in zip(messages, contexts)]
