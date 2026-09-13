"""Leakage-free baseline evaluation on the frozen golden evaluation set (N=150).

Evaluates MostFrequentBaseline and KeywordBaseline on the 150 human-reviewed
golden examples, using the strictly leakage-free training/retrieval pool (42,770 pairs).

CRITICAL CONSTRAINTS:
  - All 322 golden candidate pairs are strictly excluded from training and retrieval.
  - Exactly 150 human-reviewed examples form the evaluation test set.
  - 172 pending candidates are strictly excluded from evaluation and training/retrieval.
  - Heuristic training labels are weak signals, NOT human ground truth.
  - Escalation decisions are baseline policy outputs, NOT human labels.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.baselines import KeywordBaseline, MostFrequentBaseline, most_frequent_response
from src.intents.taxonomy import INTENT_NAMES, INTENT_TAXONOMY, classify_customer_intent

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"
PAIRS_PARQUET = ROOT / "data" / "processed" / "spotify_pairs.parquet"
RESULTS_DIR = ROOT / "results"
BASELINE_METRICS_JSON = RESULTS_DIR / "baseline_metrics.json"


def load_data(
    golden_path: str | Path = GOLDEN_CSV,
    pairs_path: str | Path = PAIRS_PARQUET,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load golden candidates CSV and processed spotify pairs parquet."""
    golden = pd.read_csv(golden_path)
    pairs = pd.read_parquet(pairs_path)
    return golden, pairs


def prepare_leakage_free_split(
    golden_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition pairs and golden set into leakage-free training pool and reviewed test set.

    Returns:
        (train_df, test_df, pending_df, verification_info)
    """
    # Enforce string pair_id representation
    golden_clean = golden_df.copy()
    pairs_clean = pairs_df.copy()
    golden_clean["pair_id"] = golden_clean["pair_id"].astype(str)
    pairs_clean["pair_id"] = pairs_clean["pair_id"].astype(str)

    all_golden_pair_ids: Set[str] = set(golden_clean["pair_id"])
    pairs_pair_ids: Set[str] = set(pairs_clean["pair_id"])

    # Overlap before exclusion
    matched_ids = all_golden_pair_ids & pairs_pair_ids
    unmatched_ids = all_golden_pair_ids - pairs_pair_ids
    overlap_before = pairs_clean["pair_id"].isin(all_golden_pair_ids).sum()

    # Assert all 322 golden candidates originate from spotify_pairs
    assert len(all_golden_pair_ids) == 322, f"Expected 322 golden pairs, got {len(all_golden_pair_ids)}"
    assert len(matched_ids) == 322, f"Expected 322 matched pairs, got {len(matched_ids)}"
    assert len(unmatched_ids) == 0, f"Expected 0 unmatched pairs, got {len(unmatched_ids)}"
    assert overlap_before == 322, f"Expected overlap_before=322, got {overlap_before}"

    # Construct leakage-free training pool (43,092 - 322 = 42,770)
    train_df = pairs_clean[~pairs_clean["pair_id"].isin(all_golden_pair_ids)].copy()
    overlap_after = train_df["pair_id"].isin(all_golden_pair_ids).sum()

    # Assert zero leakage after exclusion
    assert overlap_after == 0, f"LEAKAGE DETECTED: overlap_after={overlap_after}"
    assert len(train_df) == 42770, f"Expected 42,770 training pairs, got {len(train_df)}"

    # Construct test set: strictly reviewed rows (N=150)
    test_df = golden_clean[golden_clean["annotation_status"] == "reviewed"].copy()
    pending_df = golden_clean[golden_clean["annotation_status"] == "pending"].copy()

    # Verification assertions
    assert len(test_df) == 150, f"Expected exactly 150 reviewed test examples, got {len(test_df)}"
    assert len(pending_df) == 172, f"Expected 172 pending examples, got {len(pending_df)}"
    assert test_df["annotation_status"].eq("reviewed").all()
    assert test_df["human_primary_intent"].notna().all()
    assert test_df["human_primary_intent"].ne("").all()

    # Ensure no test or pending pair_id appears in training pool
    assert set(test_df["pair_id"]).isdisjoint(set(train_df["pair_id"]))
    assert set(pending_df["pair_id"]).isdisjoint(set(train_df["pair_id"]))

    verification = {
        "golden_total": len(golden_clean),
        "golden_reviewed": len(test_df),
        "golden_pending": len(pending_df),
        "matched_golden_pair_ids": len(matched_ids),
        "unmatched_golden_pair_ids": len(unmatched_ids),
        "overlap_before": int(overlap_before),
        "overlap_after": int(overlap_after),
        "training_count": len(train_df),
        "test_count": len(test_df),
    }

    return train_df, test_df, pending_df, verification


def compute_intent_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate accuracy, macro F1, per-class precision, recall, F1, and confusion matrix."""
    if classes is None:
        classes = INTENT_NAMES

    n = len(y_true)
    assert n > 0, "Empty truth list"
    assert len(y_pred) == n, f"Length mismatch: {len(y_pred)} != {n}"

    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []

    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    # Confusion matrix (rows = true, columns = predicted)
    cm: Dict[str, Dict[str, int]] = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_reply_metrics(replies: List[str]) -> Dict[str, Any]:
    """Calculate response delivery metrics on N=150."""
    total = len(replies)
    non_empty = [r for r in replies if r and str(r).strip() != ""]
    non_empty_rate = len(non_empty) / total if total > 0 else 0.0
    lengths = [len(str(r)) for r in non_empty]
    avg_char_length = float(np.mean(lengths)) if lengths else 0.0
    word_counts = [len(str(r).split()) for r in non_empty]
    avg_word_count = float(np.mean(word_counts)) if word_counts else 0.0

    return {
        "non_empty_reply_rate": round(non_empty_rate, 4),
        "average_reply_length": round(avg_char_length, 2),
        "average_reply_words": round(avg_word_count, 2),
    }


def compute_escalation_metrics(escalations: List[Dict[str, str]]) -> Dict[str, Any]:
    """Calculate escalation policy metrics on N=150."""
    total = len(escalations)
    auto_count = sum(1 for e in escalations if e.get("decision") == "AUTO_HANDLE")
    escalate_count = sum(1 for e in escalations if e.get("decision") == "ESCALATE")

    reason_counts = Counter(e.get("reason", "unknown") for e in escalations)

    return {
        "auto_handle_rate": round(auto_count / total, 4) if total > 0 else 0.0,
        "escalation_rate": round(escalate_count / total, 4) if total > 0 else 0.0,
        "escalation_reason_counts": dict(reason_counts),
        "note": "Escalation decisions are baseline policy outputs, not human-annotated ground truth.",
    }


class Evaluator:
    """Evaluates baseline policies against the golden evaluation set."""

    def __init__(
        self,
        golden_path: str | Path = GOLDEN_CSV,
        pairs_path: str | Path = PAIRS_PARQUET,
    ):
        self.golden_path = Path(golden_path)
        self.pairs_path = Path(pairs_path)
        self.golden, self.pairs = load_data(self.golden_path, self.pairs_path)
        self.train_df, self.test_df, self.pending_df, self.verification = prepare_leakage_free_split(
            self.golden, self.pairs
        )
        self.all_golden_ids = set(self.golden["pair_id"].astype(str))

    @property
    def training_pool(self) -> pd.DataFrame:
        """Return the leakage-free training pool."""
        return self.train_df

    def run_verification(self) -> Dict[str, int]:
        """Return verification counters and assert zero leakage."""
        return self.verification

    def evaluate_baselines(self) -> Dict[str, Any]:
        """Evaluate MostFrequentBaseline and KeywordBaseline on the 150 reviewed examples."""
        test_messages = self.test_df["customer_message"].astype(str).tolist()
        test_contexts = self.test_df["context_before_customer"].fillna("").astype(str).tolist()
        y_true = self.test_df["human_primary_intent"].astype(str).tolist()

        assert len(test_messages) == 150
        assert len(y_true) == 150

        # Fit MostFrequentBaseline
        mf_model = MostFrequentBaseline(
            training_pool=self.train_df,
            golden_pair_ids=self.all_golden_ids,
        )
        mf_preds = mf_model.predict(test_messages, test_contexts)
        mf_replies = mf_model.reply(test_messages, test_contexts)
        mf_escalations = mf_model.decide_escalation(test_messages, test_contexts, mf_preds)

        assert len(mf_preds) == 150
        assert len(mf_replies) == 150
        assert len(mf_escalations) == 150

        mf_intent_metrics = compute_intent_metrics(y_true, mf_preds)
        mf_reply_metrics = compute_reply_metrics(mf_replies)
        mf_esc_metrics = compute_escalation_metrics(mf_escalations)

        # Fit KeywordBaseline
        kw_model = KeywordBaseline(
            training_pool=self.train_df,
            golden_pair_ids=self.all_golden_ids,
        )
        kw_preds = kw_model.predict(test_messages, test_contexts)
        kw_replies = kw_model.reply(test_messages, test_contexts)
        kw_escalations = kw_model.decide_escalation(test_messages, test_contexts, kw_preds)

        assert len(kw_preds) == 150
        assert len(kw_replies) == 150
        assert len(kw_escalations) == 150

        kw_intent_metrics = compute_intent_metrics(y_true, kw_preds)
        kw_reply_metrics = compute_reply_metrics(kw_replies)
        kw_esc_metrics = compute_escalation_metrics(kw_escalations)

        results = {
            "dataset": {
                "raw_spotify_pairs_count": len(self.pairs),
                "golden_total": self.verification["golden_total"],
                "golden_reviewed": self.verification["golden_reviewed"],
                "golden_pending": self.verification["golden_pending"],
                "evaluation_test_count": self.verification["test_count"],
                "training_retrieval_count": self.verification["training_count"],
                "golden_pair_ids_matched": self.verification["matched_golden_pair_ids"],
                "golden_pair_ids_unmatched": self.verification["unmatched_golden_pair_ids"],
                "overlap_before_exclusion": self.verification["overlap_before"],
                "overlap_after_exclusion": self.verification["overlap_after"],
                "identifier_strategy": "Canonical pair_id string match (e.g. pair_43459_43458)",
                "leakage_assertion_result": "PASSED (0 golden pairs in training/retrieval pool)",
                "evaluation_population": "Strictly the 150 human-reviewed golden candidates; 172 pending excluded.",
            },
            "baselines": {
                "MostFrequentBaseline": {
                    "methodology": "Majority intent from heuristic training pool distribution; constant prediction.",
                    "majority_intent": mf_model.majority_intent,
                    "intent_classification": mf_intent_metrics,
                    "reply": mf_reply_metrics,
                    "escalation": mf_esc_metrics,
                },
                "KeywordBaseline": {
                    "methodology": "Deterministic 10-intent keyword taxonomy rules with context disambiguation; historical response retrieval.",
                    "intent_classification": kw_intent_metrics,
                    "reply": kw_reply_metrics,
                    "escalation": kw_esc_metrics,
                },
            },
            "limitations": {
                "training_heuristic_labels": "Heuristic training labels in spotify_pairs are weak signals, NOT human ground truth.",
                "golden_evaluation_ground_truth": "Only the 150 reviewed golden examples are human evaluation ground truth.",
                "pending_exclusion": "172 pending examples are excluded from both evaluation and training/retrieval.",
                "escalation_policy": "Escalation decisions are deterministic baseline policy outputs, not human labels. No escalation ground truth exists in the dataset.",
                "response_quality": "No LLM-as-judge or automated semantic similarity metric implemented yet. Reply metrics only measure delivery and length.",
            },
        }

        return results


def evaluate() -> Dict[str, Any]:
    """Execute evaluation, print diagnostics, and save baseline_metrics.json."""
    evaluator = Evaluator()
    v = evaluator.run_verification()

    print("=" * 70)
    print("SPOTIFYCARES BASELINE EVALUATION — LEAKAGE-FREE SPLIT DIAGNOSTICS")
    print("=" * 70)
    print(f"GOLDEN_TOTAL = {v['golden_total']}")
    print(f"GOLDEN_REVIEWED = {v['golden_reviewed']}")
    print(f"GOLDEN_PENDING = {v['golden_pending']}")
    print(f"MATCHED_GOLDEN_PAIR_IDS = {v['matched_golden_pair_ids']}")
    print(f"UNMATCHED_GOLDEN_PAIR_IDS = {v['unmatched_golden_pair_ids']}")
    print(f"OVERLAP_BEFORE = {v['overlap_before']}")
    print(f"OVERLAP_AFTER = {v['overlap_after']}")
    print(f"TRAINING_COUNT = {v['training_count']}")
    print(f"TEST_COUNT = {v['test_count']}")
    print("=" * 70)

    # Compute baseline results
    results = evaluator.evaluate_baselines()

    mf = results["baselines"]["MostFrequentBaseline"]
    kw = results["baselines"]["KeywordBaseline"]

    print("\nINTENT CLASSIFICATION METRICS (N=150):")
    print(f"  MostFrequentBaseline ({mf['majority_intent']}):")
    print(f"    Accuracy: {mf['intent_classification']['accuracy']:.4f}")
    print(f"    Macro F1: {mf['intent_classification']['macro_f1']:.4f}")
    print(f"  KeywordBaseline:")
    print(f"    Accuracy: {kw['intent_classification']['accuracy']:.4f}")
    print(f"    Macro F1: {kw['intent_classification']['macro_f1']:.4f}")

    print("\nREPLY RETRIEVAL METRICS (N=150):")
    print(f"  MostFrequentBaseline: non_empty_rate={mf['reply']['non_empty_reply_rate']}, avg_len={mf['reply']['average_reply_length']} chars")
    print(f"  KeywordBaseline:      non_empty_rate={kw['reply']['non_empty_reply_rate']}, avg_len={kw['reply']['average_reply_length']} chars")

    print("\nESCALATION POLICY METRICS (N=150):")
    print(f"  MostFrequentBaseline: auto_handle={mf['escalation']['auto_handle_rate']}, escalate={mf['escalation']['escalation_rate']}")
    print(f"  KeywordBaseline:      auto_handle={kw['escalation']['auto_handle_rate']}, escalate={kw['escalation']['escalation_rate']}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BASELINE_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {BASELINE_METRICS_JSON}")

    return results


if __name__ == "__main__":
    evaluate()
