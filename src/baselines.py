"""Baseline models for intent classification, response retrieval, and escalation policy.

Implements:
  1. MostFrequentBaseline: Predicts the empirical majority heuristic intent
     derived from the leakage-free training pool.
  2. KeywordBaseline: Predicts intent using the 10-intent keyword taxonomy rules
     from src.intents.taxonomy.classify_customer_intent.

Both baselines adhere to the uniform API:
  - predict(messages, contexts=None) -> List[str]
  - reply(messages, contexts=None) -> List[str]
  - decide_escalation(messages, contexts=None, predicted_intents=None) -> List[dict]

CRITICAL CONSTRAINTS:
  - Golden evaluation set pair_ids must be strictly excluded before fitting or retrieval.
  - Training labels are heuristic/weak signals, NOT human ground truth.
  - Escalation decisions are baseline policy outputs, NOT human labels.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Any
import pandas as pd

from src.intents.taxonomy import (
    INTENT_NAMES,
    INTENT_TAXONOMY,
    classify_customer_intent,
)


def most_frequent_response(
    training_pool: pd.DataFrame,
    intent: str,
    golden_pair_ids: Optional[Set[str]] = None,
    fallback: str = "Hi there! How can we help you today? Send us a DM with your account details and we\'ll check it out.",
) -> str:
    """Retrieve the most frequent historical brand response for a given intent.

    Applies strict leakage verification: asserts that no golden pair_id is in the retrieval pool.
    Uses deterministic lexical tie-breaking: sorts alphabetically and picks the first.
    Returns fallback if no valid response exists in the pool.
    """
    if golden_pair_ids is not None:
        pool_ids = set(training_pool["pair_id"].astype(str))
        assert pool_ids.isdisjoint(golden_pair_ids), (
            f"LEAKAGE DETECTED: {len(pool_ids & golden_pair_ids)} golden pair_ids "
            f"found in retrieval pool!"
        )

    if "customer_primary_intent" not in training_pool.columns or "brand_response" not in training_pool.columns:
        return fallback

    subset = training_pool[training_pool["customer_primary_intent"] == intent]
    if subset.empty:
        return fallback

    # Filter non-null, non-empty responses
    valid_responses = subset["brand_response"].dropna().astype(str).str.strip()
    valid_responses = valid_responses[valid_responses.ne("")]

    if valid_responses.empty:
        return fallback

    counts = valid_responses.value_counts()
    if counts.empty:
        return fallback

    max_count = counts.iloc[0]
    tied = sorted(counts[counts == max_count].index.tolist())
    return tied[0]


class MostFrequentBaseline:
    """Most Frequent (heuristic-label) baseline.

    Derives the empirical majority intent from the leakage-free training pool.
    Because raw historical interactions do not contain human labels, the training
    distribution is derived from heuristic classification (customer_primary_intent).
    These heuristic labels are NOT human ground truth.

    Deterministic tie-breaking follows canonical taxonomy order (INTENT_NAMES).
    After fit(), every prediction is the same majority intent.
    """

    def __init__(
        self,
        training_pool: Optional[pd.DataFrame] = None,
        golden_pair_ids: Optional[Set[str]] = None,
    ):
        self.majority_intent: str = "other_miscellaneous"
        self.majority_response: str = ""
        self.golden_pair_ids: Set[str] = set(str(pid) for pid in (golden_pair_ids or []))
        self.is_fitted: bool = False

        if training_pool is not None:
            self.fit(training_pool, golden_pair_ids=self.golden_pair_ids)

    def fit(
        self,
        training_pool: pd.DataFrame,
        golden_pair_ids: Optional[Set[str]] = None,
    ) -> MostFrequentBaseline:
        """Fit the baseline on the leakage-free training pool."""
        if golden_pair_ids is not None:
            self.golden_pair_ids = set(str(pid) for pid in golden_pair_ids)

        # Runtime leakage assertion
        if self.golden_pair_ids:
            pool_ids = set(training_pool["pair_id"].astype(str))
            overlap = pool_ids & self.golden_pair_ids
            assert len(overlap) == 0, (
                f"LEAKAGE ASSERTION FAILED: {len(overlap)} golden pair_ids "
                f"found in MostFrequentBaseline training pool!"
            )

        # Determine majority intent from training pool
        if "customer_primary_intent" in training_pool.columns:
            counts = training_pool["customer_primary_intent"].value_counts()
            if not counts.empty:
                max_count = counts.iloc[0]
                tied = [intent for intent in INTENT_NAMES if counts.get(intent, 0) == max_count]
                self.majority_intent = tied[0] if tied else counts.index[0]
        else:
            # Classify heuristically if customer_primary_intent not present
            classified = [
                classify_customer_intent(str(msg), str(ctx))
                for msg, ctx in zip(
                    training_pool.get("customer_message", []),
                    training_pool.get("context_before_customer", [""] * len(training_pool)),
                )
            ]
            counts = pd.Series(classified).value_counts()
            max_count = counts.iloc[0]
            tied = [intent for intent in INTENT_NAMES if counts.get(intent, 0) == max_count]
            self.majority_intent = tied[0] if tied else counts.index[0]

        # Retrieve the most frequent response for the majority intent
        self.majority_response = most_frequent_response(
            training_pool,
            self.majority_intent,
            golden_pair_ids=self.golden_pair_ids,
        )

        self.is_fitted = True
        return self

    def predict(
        self,
        messages: List[str],
        contexts: Optional[List[str]] = None,
    ) -> List[str]:
        """Predict intent label for each message. Always returns majority intent.

        Returns:
            List[str] of predicted intent names (length equals len(messages)).
        """
        return [self.majority_intent] * len(messages)

    def reply(
        self,
        messages: List[str],
        contexts: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate response for each message. Always returns majority response.

        Returns:
            List[str] of reply strings (length equals len(messages)).
        """
        return [self.majority_response] * len(messages)

    def decide_escalation(
        self,
        messages: List[str],
        contexts: Optional[List[str]] = None,
        predicted_intents: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Return escalation decision based on deterministic policy rules.

        Escalation decisions are policy outputs, NOT human labels.
        Financial/billing and account/security intents require human escalation.

        Returns:
            List[dict] containing {"decision": "AUTO_HANDLE" | "ESCALATE", "reason": "..."}
        """
        if predicted_intents is None:
            predicted_intents = self.predict(messages, contexts)

        decisions: List[Dict[str, str]] = []
        for intent in predicted_intents:
            defn = INTENT_TAXONOMY.get(intent)
            if defn is not None and defn.escalation_recommended:
                decisions.append({
                    "decision": "ESCALATE",
                    "reason": f"Intent \'{intent}\' requires escalation per security/financial policy ({defn.display_name}).",
                })
            else:
                decisions.append({
                    "decision": "AUTO_HANDLE",
                    "reason": f"Intent \'{intent}\' can be handled automatically by baseline policy.",
                })
        return decisions


class KeywordBaseline:
    """Keyword baseline classifier and historical response retriever.

    Uses the machine-readable 10-intent taxonomy from src.intents.taxonomy.classify_customer_intent.
    Does not invent a new taxonomy.

    For response generation:
      - Retrieves the most frequent historical brand response from the training pool for each predicted intent.
      - Never retrieves from any of the 322 golden candidate pairs.
      - Uses deterministic lexical tie-breaking.
    """

    def __init__(
        self,
        training_pool: Optional[pd.DataFrame] = None,
        golden_pair_ids: Optional[Set[str]] = None,
    ):
        self.intent_to_response: Dict[str, str] = {}
        self.golden_pair_ids: Set[str] = set(str(pid) for pid in (golden_pair_ids or []))
        self.fallback_response: str = (
            "Hi there! How can we help you today? Send us a DM with more info and we\'ll check it out /SpotifyCares"
        )
        self.is_fitted: bool = False

        if training_pool is not None:
            self.fit(training_pool, golden_pair_ids=self.golden_pair_ids)

    def fit(
        self,
        training_pool: pd.DataFrame,
        golden_pair_ids: Optional[Set[str]] = None,
    ) -> KeywordBaseline:
        """Fit response retrieval map from leakage-free training pool."""
        if golden_pair_ids is not None:
            self.golden_pair_ids = set(str(pid) for pid in golden_pair_ids)

        # Explicit runtime assertion against leakage
        if self.golden_pair_ids:
            pool_ids = set(training_pool["pair_id"].astype(str))
            overlap = pool_ids & self.golden_pair_ids
            assert len(overlap) == 0, (
                f"LEAKAGE ASSERTION FAILED: {len(overlap)} golden pair_ids "
                f"found in KeywordBaseline retrieval pool!"
            )

        # Build response retrieval dictionary per intent
        self.intent_to_response = {}
        for intent in INTENT_NAMES:
            resp = most_frequent_response(
                training_pool,
                intent,
                golden_pair_ids=self.golden_pair_ids,
                fallback=self.fallback_response,
            )
            self.intent_to_response[intent] = resp

        self.is_fitted = True
        return self

    def predict(
        self,
        messages: List[str],
        contexts: Optional[List[str]] = None,
    ) -> List[str]:
        """Predict intent label using canonical taxonomy keyword classifier.

        Returns:
            List[str] of predicted canonical intent names.
        """
        if contexts is None:
            contexts = [""] * len(messages)

        predictions: List[str] = []
        for msg, ctx in zip(messages, contexts):
            intent = classify_customer_intent(str(msg), context=str(ctx))
            predictions.append(intent)
        return predictions

    def reply(
        self,
        messages: List[str],
        contexts: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate response via historical retrieval from the training pool.

        Returns:
            List[str] of retrieved response strings.
        """
        intents = self.predict(messages, contexts)
        return [self.intent_to_response.get(i, self.fallback_response) for i in intents]

    def decide_escalation(
        self,
        messages: List[str],
        contexts: Optional[List[str]] = None,
        predicted_intents: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Return escalation decision based on deterministic policy rules.

        Escalation decisions are policy outputs, NOT human labels.
        Financial/billing and account/security intents require human escalation.

        Returns:
            List[dict] containing {"decision": "AUTO_HANDLE" | "ESCALATE", "reason": "..."}
        """
        if predicted_intents is None:
            predicted_intents = self.predict(messages, contexts)

        decisions: List[Dict[str, str]] = []
        for intent in predicted_intents:
            defn = INTENT_TAXONOMY.get(intent)
            if defn is not None and defn.escalation_recommended:
                decisions.append({
                    "decision": "ESCALATE",
                    "reason": f"Intent \'{intent}\' requires escalation per security/financial policy ({defn.display_name}).",
                })
            else:
                decisions.append({
                    "decision": "AUTO_HANDLE",
                    "reason": f"Intent \'{intent}\' can be handled automatically by baseline policy.",
                })
        return decisions
