"""Unit tests for baseline policies in src/baselines.py.

Covers:
  - Base API conformance: predict() -> List[str], reply() -> List[str], decide_escalation() -> List[dict]
  - MostFrequentBaseline constant prediction and majority selection
  - KeywordBaseline deterministic classification and historical retrieval
  - Retrieval never uses golden pairs
  - Deterministic lexical tie-breaking
  - Escalation schema conformance
  - Leakage assertion failure when golden pair inserted
"""
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines import KeywordBaseline, MostFrequentBaseline, most_frequent_response
from src.intents.taxonomy import INTENT_NAMES, INTENT_TAXONOMY


@pytest.fixture
def small_training_pool() -> pd.DataFrame:
    """Create a small deterministic synthetic training pool."""
    return pd.DataFrame([
        {
            "pair_id": "pair_100_101",
            "conversation_id": 100,
            "turn_index": 0,
            "customer_message": "Hello, my song paused unexpectedly",
            "context_before_customer": "",
            "brand_response": "Hi there! Try restarting your device.",
            "is_initial_inquiry": True,
            "customer_primary_intent": "playback_streaming_issue",
        },
        {
            "pair_id": "pair_102_103",
            "conversation_id": 101,
            "turn_index": 0,
            "customer_message": "Audio stopped playing mid track",
            "context_before_customer": "",
            "brand_response": "Hi there! Try restarting your device.",
            "is_initial_inquiry": True,
            "customer_primary_intent": "playback_streaming_issue",
        },
        {
            "pair_id": "pair_104_105",
            "conversation_id": 102,
            "turn_index": 0,
            "customer_message": "I was charged twice for premium",
            "context_before_customer": "",
            "brand_response": "Hey! Can you send us a DM with your account email?",
            "is_initial_inquiry": True,
            "customer_primary_intent": "billing_subscription_payment",
        },
        {
            "pair_id": "pair_106_107",
            "conversation_id": 103,
            "turn_index": 0,
            "customer_message": "Can't log into my account password failed",
            "context_before_customer": "",
            "brand_response": "Hi! Check out our password reset steps at spotify.com",
            "is_initial_inquiry": True,
            "customer_primary_intent": "account_access_credentials",
        },
        # Tie-break test rows: two equal frequency responses for other_miscellaneous
        {
            "pair_id": "pair_108_109",
            "conversation_id": 104,
            "turn_index": 0,
            "customer_message": "Love your app",
            "context_before_customer": "",
            "brand_response": "Beta response thanks",
            "is_initial_inquiry": True,
            "customer_primary_intent": "other_miscellaneous",
        },
        {
            "pair_id": "pair_110_111",
            "conversation_id": 105,
            "turn_index": 0,
            "customer_message": "Great service",
            "context_before_customer": "",
            "brand_response": "Alpha response thanks",
            "is_initial_inquiry": True,
            "customer_primary_intent": "other_miscellaneous",
        },
    ])


class TestBaselineAPI:
    """Test standard API contracts: predict, reply, decide_escalation."""

    def test_predict_returns_list_str(self, small_training_pool):
        model = MostFrequentBaseline(small_training_pool)
        preds = model.predict(["Hello", "Help"])
        assert isinstance(preds, list)
        assert all(isinstance(p, str) for p in preds)
        assert len(preds) == 2

    def test_reply_returns_list_str(self, small_training_pool):
        model = MostFrequentBaseline(small_training_pool)
        replies = model.reply(["Hello", "Help"])
        assert isinstance(replies, list)
        assert all(isinstance(r, str) for r in replies)
        assert len(replies) == 2

    def test_escalation_schema(self, small_training_pool):
        """test_escalation_schema: decide_escalation returns List[dict] with valid keys and decisions."""
        model = MostFrequentBaseline(small_training_pool)
        decisions = model.decide_escalation(["Hello", "Help"])
        assert isinstance(decisions, list)
        assert len(decisions) == 2
        for d in decisions:
            assert isinstance(d, dict)
            assert "decision" in d
            assert "reason" in d
            assert d["decision"] in ("AUTO_HANDLE", "ESCALATE")
            assert isinstance(d["reason"], str)


class TestMostFrequentBaseline:
    """Test MostFrequentBaseline properties and behavior."""

    def test_majority_prediction_is_constant(self, small_training_pool):
        """test_majority_prediction_is_constant: every prediction must be the majority intent."""
        model = MostFrequentBaseline(small_training_pool)
        # playback_streaming_issue has 2 rows; others have 1
        assert model.majority_intent == "playback_streaming_issue"

        messages = ["Charged twice", "Can't log in", "Random text", "Playlist gone"]
        preds = model.predict(messages)
        assert len(preds) == 4
        assert all(p == "playback_streaming_issue" for p in preds)

    def test_predictions_are_canonical_taxonomy_labels(self, small_training_pool):
        """test_predictions_are_canonical_taxonomy_labels: predictions must belong to canonical set."""
        model = MostFrequentBaseline(small_training_pool)
        preds = model.predict(["Test message"])
        for p in preds:
            assert p in INTENT_NAMES

    def test_reply_uses_majority_response(self, small_training_pool):
        model = MostFrequentBaseline(small_training_pool)
        replies = model.reply(["Any message"])
        assert replies[0] == "Hi there! Try restarting your device."


class TestKeywordBaseline:
    """Test KeywordBaseline classification and retrieval behavior."""

    def test_keyword_baseline_is_deterministic(self, small_training_pool):
        """test_keyword_baseline_is_deterministic: identical input produces identical output."""
        model = KeywordBaseline(small_training_pool)
        msg = "I was charged twice on my credit card"
        pred1 = model.predict([msg])
        pred2 = model.predict([msg])
        assert pred1 == pred2 == ["billing_subscription_payment"]

        reply1 = model.reply([msg])
        reply2 = model.reply([msg])
        assert reply1 == reply2

    def test_retrieval_is_deterministic(self, small_training_pool):
        """test_retrieval_is_deterministic: retrieval uses lexical tie-break on equal counts."""
        # other_miscellaneous has tied responses: "Beta response thanks" vs "Alpha response thanks"
        resp = most_frequent_response(small_training_pool, "other_miscellaneous")
        # Alphabetically: "Alpha response thanks" < "Beta response thanks"
        assert resp == "Alpha response thanks"

    def test_retrieval_never_uses_golden_pairs(self, small_training_pool):
        """test_retrieval_never_uses_golden_pairs: asserting disjointness holds on clean pool."""
        golden_ids = {"pair_999_998", "pair_997_996"}
        model = KeywordBaseline(small_training_pool, golden_pair_ids=golden_ids)
        assert model.is_fitted

    def test_leakage_assertion_fails_if_golden_pair_inserted(self, small_training_pool):
        """test_leakage_assertion_fails_if_golden_pair_inserted: runtime error if golden pair in pool."""
        # Insert a golden pair into the training pool
        dirty_pool = small_training_pool.copy()
        golden_ids = {"pair_100_101"}  # This ID exists in small_training_pool!

        with pytest.raises(AssertionError) as excinfo:
            KeywordBaseline(dirty_pool, golden_pair_ids=golden_ids)
        assert "LEAKAGE" in str(excinfo.value)

        with pytest.raises(AssertionError) as excinfo2:
            MostFrequentBaseline(dirty_pool, golden_pair_ids=golden_ids)
        assert "LEAKAGE" in str(excinfo2.value)


class TestEscalationPolicy:
    """Test policy routing rules."""

    def test_financial_and_security_escalate(self, small_training_pool):
        model = KeywordBaseline(small_training_pool)
        messages = [
            "You charged me twice on my credit card",
            "Someone hacked my account and changed my password",
            "My music stopped playing",
        ]
        decisions = model.decide_escalation(messages)
        assert len(decisions) == 3
        assert decisions[0]["decision"] == "ESCALATE"
        assert decisions[1]["decision"] == "ESCALATE"
        assert decisions[2]["decision"] == "AUTO_HANDLE"
