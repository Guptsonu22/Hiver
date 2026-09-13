"""Unit and integration tests for src/evaluate.py using real repository data.

Verifies:
  1. test_reviewed_golden_set_is_exactly_150
  2. test_pending_golden_set_is_172
  3. test_all_322_golden_pair_ids_match
  4. test_all_322_golden_pairs_excluded_from_training
  5. test_zero_golden_overlap_after_exclusion
  6. test_evaluation_never_includes_pending
  7. test_all_150_reviewed_examples_reach_evaluation
  8. test_predictions_length_equals_150
  9. test_replies_length_equals_150
  10. test_escalation_length_equals_150
  17. test_metrics_correctness
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import (
    Evaluator,
    compute_intent_metrics,
    compute_reply_metrics,
    compute_escalation_metrics,
    load_data,
    prepare_leakage_free_split,
)
from src.intents.taxonomy import INTENT_NAMES

GOLDEN_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"
PAIRS_PARQUET = ROOT / "data" / "processed" / "spotify_pairs.parquet"


@pytest.mark.skipif(
    not GOLDEN_CSV.exists() or not PAIRS_PARQUET.exists(),
    reason="Actual repository data files not found",
)
class TestRealDataVerification:
    """Rigorous leakage-free checks executing on real repository datasets."""

    @pytest.fixture
    def evaluator(self):
        return Evaluator()

    def test_reviewed_golden_set_is_exactly_150(self, evaluator):
        """1. test_reviewed_golden_set_is_exactly_150"""
        reviewed = evaluator.golden[evaluator.golden["annotation_status"] == "reviewed"]
        assert len(reviewed) == 150, f"Expected 150 reviewed rows, got {len(reviewed)}"

    def test_pending_golden_set_is_172(self, evaluator):
        """2. test_pending_golden_set_is_172"""
        pending = evaluator.golden[evaluator.golden["annotation_status"] == "pending"]
        assert len(pending) == 172, f"Expected 172 pending rows, got {len(pending)}"

    def test_all_322_golden_pair_ids_match(self, evaluator):
        """3. test_all_322_golden_pair_ids_match"""
        golden_ids = set(evaluator.golden["pair_id"].astype(str))
        pairs_ids = set(evaluator.pairs["pair_id"].astype(str))
        matched = golden_ids & pairs_ids
        assert len(golden_ids) == 322
        assert len(matched) == 322, f"Expected 322 matched golden pairs, got {len(matched)}"
        assert len(golden_ids - pairs_ids) == 0

    def test_all_322_golden_pairs_excluded_from_training(self, evaluator):
        """4. test_all_322_golden_pairs_excluded_from_training"""
        golden_ids = set(evaluator.golden["pair_id"].astype(str))
        train_ids = set(evaluator.train_df["pair_id"].astype(str))
        assert len(train_ids) == 42770, f"Expected 42,770 training rows, got {len(train_ids)}"
        assert len(golden_ids & train_ids) == 0

    def test_zero_golden_overlap_after_exclusion(self, evaluator):
        """5. test_zero_golden_overlap_after_exclusion"""
        golden_ids = set(evaluator.golden["pair_id"].astype(str))
        overlap = evaluator.train_df["pair_id"].astype(str).isin(golden_ids).sum()
        assert overlap == 0, f"Expected zero overlap after exclusion, got {overlap}"

    def test_evaluation_never_includes_pending(self, evaluator):
        """6. test_evaluation_never_includes_pending"""
        pending_ids = set(evaluator.pending_df["pair_id"].astype(str))
        test_ids = set(evaluator.test_df["pair_id"].astype(str))
        assert len(pending_ids) == 172
        assert len(test_ids) == 150
        assert len(pending_ids & test_ids) == 0, "Pending examples leaked into test set!"
        assert evaluator.test_df["annotation_status"].eq("reviewed").all()

    def test_all_150_reviewed_examples_reach_evaluation(self, evaluator):
        """7. test_all_150_reviewed_examples_reach_evaluation"""
        assert len(evaluator.test_df) == 150
        assert evaluator.test_df["human_primary_intent"].notna().all()
        assert evaluator.test_df["human_primary_intent"].ne("").all()

    def test_predictions_length_equals_150(self, evaluator):
        """8. test_predictions_length_equals_150"""
        results = evaluator.evaluate_baselines()
        mf = results["baselines"]["MostFrequentBaseline"]
        kw = results["baselines"]["KeywordBaseline"]
        # Check support across confusion matrix or intent classification
        assert sum(v["support"] for v in mf["intent_classification"]["per_class"].values()) == 150
        assert sum(v["support"] for v in kw["intent_classification"]["per_class"].values()) == 150

    def test_replies_length_equals_150(self, evaluator):
        """9. test_replies_length_equals_150"""
        test_messages = evaluator.test_df["customer_message"].astype(str).tolist()
        mf_replies = evaluator.training_pool  # verify reply method directly
        from src.baselines import MostFrequentBaseline, KeywordBaseline
        mf = MostFrequentBaseline(evaluator.training_pool, golden_pair_ids=evaluator.all_golden_ids)
        kw = KeywordBaseline(evaluator.training_pool, golden_pair_ids=evaluator.all_golden_ids)
        assert len(mf.reply(test_messages)) == 150
        assert len(kw.reply(test_messages)) == 150

    def test_escalation_length_equals_150(self, evaluator):
        """10. test_escalation_length_equals_150"""
        test_messages = evaluator.test_df["customer_message"].astype(str).tolist()
        from src.baselines import MostFrequentBaseline, KeywordBaseline
        mf = MostFrequentBaseline(evaluator.training_pool, golden_pair_ids=evaluator.all_golden_ids)
        kw = KeywordBaseline(evaluator.training_pool, golden_pair_ids=evaluator.all_golden_ids)
        assert len(mf.decide_escalation(test_messages)) == 150
        assert len(kw.decide_escalation(test_messages)) == 150

    def test_metrics_correctness(self):
        """17. test_metrics_correctness: verify mathematical calculation of accuracy and F1."""
        y_true = ["playback_streaming_issue", "billing_subscription_payment", "playback_streaming_issue", "other_miscellaneous"]
        y_pred = ["playback_streaming_issue", "billing_subscription_payment", "app_technical_device", "other_miscellaneous"]

        # 3 correct out of 4 -> accuracy = 0.75
        metrics = compute_intent_metrics(y_true, y_pred, classes=INTENT_NAMES)
        assert metrics["accuracy"] == 0.75

        # Check precision and recall for playback_streaming_issue
        # TP = 1, FP = 0, FN = 1 -> prec = 1.0, rec = 0.5, f1 = 2/3 = 0.6667
        pb = metrics["per_class"]["playback_streaming_issue"]
        assert pb["precision"] == 1.0
        assert pb["recall"] == 0.5
        assert pb["f1"] == 0.6667

        # Check reply metrics
        replies = ["Hello there", "Thanks for reaching out", ""]
        rep_metrics = compute_reply_metrics(replies)
        assert rep_metrics["non_empty_reply_rate"] == round(2 / 3, 4)

        # Check escalation metrics
        escalations = [
            {"decision": "ESCALATE", "reason": "billing issue"},
            {"decision": "AUTO_HANDLE", "reason": "playback handled"},
        ]
        esc_metrics = compute_escalation_metrics(escalations)
        assert esc_metrics["auto_handle_rate"] == 0.5
        assert esc_metrics["escalation_rate"] == 0.5
