"""Tests for the Phase 3 assisted golden annotation workflow."""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "assist_annotate_golden_set.py"


def load_assist_module():
    spec = importlib.util.spec_from_file_location("assist_annotate_golden_set", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_candidates_df():
    return pd.DataFrame(
        [
            {
                "example_id": "cand_reviewed",
                "pair_id": "pair_1_2",
                "conversation_id": "conv_1",
                "customer_message": "I was charged twice this month.",
                "context_before_customer": "",
                "brand_response": "Send us a DM and we can check.",
                "is_initial_inquiry": True,
                "existing_heuristic_intent": "billing_subscription_payment",
                "existing_candidate_intents": "billing_subscription_payment",
                "hard_case_category": "baseline",
                "response_type": "Redirect / DM",
                "sampling_stratum": "reviewed",
                "human_primary_intent": "billing_subscription_payment",
                "human_secondary_intents": "",
                "human_is_ambiguous": "False",
                "human_is_multi_intent": "False",
                "human_notes": "Already reviewed by human.",
                "annotation_status": "reviewed",
            },
            {
                "example_id": "cand_pending",
                "pair_id": "pair_3_4",
                "conversation_id": "conv_2",
                "customer_message": "My downloaded songs won't play offline.",
                "context_before_customer": "",
                "brand_response": "Try toggling offline mode.",
                "is_initial_inquiry": True,
                "existing_heuristic_intent": "offline_downloads_issue",
                "existing_candidate_intents": "offline_downloads_issue,playback_streaming_issue",
                "hard_case_category": "confusable",
                "response_type": "Troubleshooting / Actionable",
                "sampling_stratum": "pending",
                "human_primary_intent": "",
                "human_secondary_intents": "",
                "human_is_ambiguous": "",
                "human_is_multi_intent": "",
                "human_notes": "",
                "annotation_status": "pending",
            },
        ]
    )


def make_batch_df():
    rows = [
        {
            "example_id": "reviewed_keep",
            "pair_id": "pair_10_11",
            "conversation_id": "conv_10",
            "customer_message": "App crashes.",
            "context_before_customer": "",
            "brand_response": "Try reinstalling.",
            "is_initial_inquiry": True,
            "existing_heuristic_intent": "app_technical_device",
            "existing_candidate_intents": "app_technical_device",
            "hard_case_category": "baseline",
            "response_type": "Troubleshooting / Actionable",
            "sampling_stratum": "reviewed",
            "human_primary_intent": "app_technical_device",
            "human_secondary_intents": "",
            "human_is_ambiguous": "False",
            "human_is_multi_intent": "False",
            "human_notes": "Reviewed already.",
            "annotation_status": "reviewed",
        },
        {
            "example_id": "low_risk",
            "pair_id": "pair_20_21",
            "conversation_id": "conv_20",
            "customer_message": "I was charged twice and need a refund.",
            "context_before_customer": "",
            "brand_response": "Send us a DM.",
            "is_initial_inquiry": True,
            "existing_heuristic_intent": "billing_subscription_payment",
            "existing_candidate_intents": "billing_subscription_payment",
            "hard_case_category": "baseline",
            "response_type": "Redirect / DM",
            "sampling_stratum": "pending",
            "human_primary_intent": "",
            "human_secondary_intents": "",
            "human_is_ambiguous": "",
            "human_is_multi_intent": "",
            "human_notes": "",
            "annotation_status": "pending",
        },
        {
            "example_id": "high_risk",
            "pair_id": "pair_30_31",
            "conversation_id": "conv_30",
            "customer_message": "same issue lol ???",
            "context_before_customer": "Customer: I cannot log in and I was charged twice.",
            "brand_response": "Please DM us.",
            "is_initial_inquiry": False,
            "existing_heuristic_intent": "playback_streaming_issue",
            "existing_candidate_intents": "account_access_credentials,billing_subscription_payment",
            "hard_case_category": "context_dependent_followup",
            "response_type": "Redirect / DM",
            "sampling_stratum": "pending",
            "human_primary_intent": "",
            "human_secondary_intents": "",
            "human_is_ambiguous": "",
            "human_is_multi_intent": "",
            "human_notes": "",
            "annotation_status": "pending",
        },
    ]
    return pd.DataFrame(rows)


def test_compute_suggestion_uses_context_for_followup():
    mod = load_assist_module()

    suggestion = mod.compute_suggestion(
        customer_message="Same issue here",
        context_before_customer="Customer: Spotify crashes every time I open it.",
        brand_response="What device are you using?",
    )

    assert suggestion["suggested_primary_intent"] == "app_technical_device"
    assert suggestion["suggestion_confidence"] == "medium"
    assert "Context used" in suggestion["suggested_notes"]


def test_ensure_suggestions_only_populates_pending_rows_and_preserves_human_labels():
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_candidates_df())

    added = mod.ensure_suggestions_for_pending(df)

    assert added == 1
    assert df.loc[0, "human_primary_intent"] == "billing_subscription_payment"
    assert df.loc[0, "annotation_status"] == "reviewed"
    assert df.loc[0, "suggested_primary_intent"] == ""
    assert df.loc[1, "suggested_primary_intent"] == "offline_downloads_issue"
    assert df.loc[1, "human_primary_intent"] == ""
    assert df.loc[1, "annotation_status"] == "pending"


def test_apply_human_approved_suggestion_writes_human_fields_with_audit_method():
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_candidates_df())
    mod.ensure_suggestions_for_pending(df)

    mod.apply_human_approved_suggestion(df, 1)

    assert df.loc[1, "human_primary_intent"] == "offline_downloads_issue"
    assert df.loc[1, "annotation_status"] == "reviewed"
    assert df.loc[1, "annotation_method"] == "human_approved_ai_suggestion"
    assert df.loc[1, "existing_heuristic_intent"] == "offline_downloads_issue"


def test_apply_human_override_uses_distinct_audit_method():
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_candidates_df())

    mod.apply_human_override(
        df,
        1,
        primary="playback_streaming_issue",
        secondary="offline_downloads_issue",
        is_ambiguous="True",
        notes="Human correction during review.",
    )

    assert df.loc[1, "human_primary_intent"] == "playback_streaming_issue"
    assert df.loc[1, "human_secondary_intents"] == "offline_downloads_issue"
    assert df.loc[1, "human_is_ambiguous"] == "True"
    assert df.loc[1, "human_is_multi_intent"] == "True"
    assert df.loc[1, "annotation_method"] == "human_override"


def test_review_skip_does_not_auto_approve(tmp_path):
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_candidates_df().iloc[[1]].copy())
    csv_path = tmp_path / "candidates.csv"

    inputs = iter(["s"])
    mod.review_pending(df, csv_path, input_fn=lambda _: next(inputs))

    assert df.loc[1, "suggested_primary_intent"] == "offline_downloads_issue"
    assert df.loc[1, "human_primary_intent"] == ""
    assert df.loc[1, "annotation_status"] == "pending"
    assert not csv_path.exists()


def test_suggest_only_round_trip_writes_no_human_labels(tmp_path):
    mod = load_assist_module()
    csv_path = tmp_path / "candidates.csv"
    make_candidates_df().to_csv(csv_path, index=False)

    df = mod.load_candidates(csv_path)
    mod.ensure_suggestions_for_pending(df)
    mod.save_candidates(df, csv_path)
    saved = pd.read_csv(csv_path, dtype=str).fillna("")

    pending = saved[saved["example_id"] == "cand_pending"].iloc[0]
    reviewed = saved[saved["example_id"] == "cand_reviewed"].iloc[0]

    assert pending["suggested_primary_intent"] == "offline_downloads_issue"
    assert pending["human_primary_intent"] == ""
    assert pending["annotation_status"] == "pending"
    assert reviewed["human_primary_intent"] == "billing_subscription_payment"
    assert reviewed["annotation_status"] == "reviewed"


def test_review_priority_puts_high_risk_pending_first():
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_batch_df())
    mod.ensure_suggestions_for_pending(df)

    ranked = mod.ranked_pending_indices(df)

    assert df.loc[ranked[0], "example_id"] == "high_risk"
    assert "reviewed_keep" not in set(df.loc[ranked, "example_id"])


def test_batch_approve_requires_explicit_item_command(tmp_path):
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_candidates_df().iloc[[1]].copy())
    mod.ensure_suggestions_for_pending(df)
    csv_path = tmp_path / "candidates.csv"

    inputs = iter(["y 1"])
    mod.review_pending_batch(df, csv_path, batch_size=1, input_fn=lambda _: next(inputs))

    assert df.loc[1, "annotation_status"] == "reviewed"
    assert df.loc[1, "human_primary_intent"] == "offline_downloads_issue"
    assert df.loc[1, "annotation_method"] == "human_approved_ai_suggestion"
    saved = pd.read_csv(csv_path, dtype=str).fillna("")
    assert saved.loc[0, "annotation_status"] == "reviewed"


def test_batch_override_records_human_override(tmp_path):
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_candidates_df().iloc[[1]].copy())
    mod.ensure_suggestions_for_pending(df)
    csv_path = tmp_path / "candidates.csv"

    inputs = iter(["o 1 8", "", "y", "Human reviewed override."])
    mod.review_pending_batch(df, csv_path, batch_size=1, input_fn=lambda _: next(inputs))

    assert df.loc[1, "annotation_status"] == "reviewed"
    assert df.loc[1, "human_primary_intent"] == "playback_streaming_issue"
    assert df.loc[1, "human_is_ambiguous"] == "True"
    assert df.loc[1, "annotation_method"] == "human_override"


def test_batch_skip_keeps_pending_and_does_not_save(tmp_path):
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_candidates_df().iloc[[1]].copy())
    mod.ensure_suggestions_for_pending(df)
    csv_path = tmp_path / "candidates.csv"

    inputs = iter(["s 1"])
    mod.review_pending_batch(df, csv_path, batch_size=1, input_fn=lambda _: next(inputs))

    assert df.loc[1, "annotation_status"] == "pending"
    assert df.loc[1, "human_primary_intent"] == ""
    assert not csv_path.exists()


def test_batch_review_does_not_modify_reviewed_rows(tmp_path):
    mod = load_assist_module()
    df = mod.coerce_workflow_columns(make_batch_df())
    mod.ensure_suggestions_for_pending(df)
    csv_path = tmp_path / "candidates.csv"

    inputs = iter(["y 1", "q"])
    mod.review_pending_batch(df, csv_path, batch_size=1, input_fn=lambda _: next(inputs))

    reviewed = df[df["example_id"] == "reviewed_keep"].iloc[0]
    assert reviewed["annotation_status"] == "reviewed"
    assert reviewed["human_primary_intent"] == "app_technical_device"
    assert reviewed["human_notes"] == "Reviewed already."
