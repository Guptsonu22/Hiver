"""
Phase 2 pytest test suite — SpotifyCares conversation reconstruction & intent discovery.

Sections:
  1. Brand / Customer message classification
  2. Spotify ecosystem isolation
  3. Conversation reconstruction
  4. Pair extraction
  5. Intent taxonomy validation
  6. Response classifier
  7. Intent classifier
  8. Malformed data handling
"""
import sys
import re
import math
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.loader import (
    load_raw_data, is_brand_message, is_customer_message, isolate_spotify_ecosystem
)
from data.conversation_builder import reconstruct_conversations, build_customer_response_pairs
from data.response_classifier import classify_response_type, extract_response_features, RESPONSE_TYPES
from intents.taxonomy import (
    INTENT_TAXONOMY, INTENT_NAMES, IntentDefinition,
    detect_candidate_intents, classify_customer_intent, get_intent,
)

BRAND         = "SpotifyCares"
PARQUET_PATH  = ROOT / "data" / "processed" / "spotify_clean.parquet"
PAIRS_PATH    = ROOT / "data" / "processed" / "spotify_pairs.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_df(rows):
    """Build a minimal tweets DataFrame from a list of row dicts."""
    df = pd.DataFrame(rows)
    df["tweet_id"] = df["tweet_id"].astype("int64")
    df["inbound"]  = df["inbound"].astype(bool)
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    ).astype("Int64")
    if "created_at" not in df.columns:
        df["created_at"] = "2017-01-01"
    df["created_at"] = df["created_at"].astype(str)
    df["text"]       = df["text"].astype(str)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 1. Brand / Customer Classification
# ═════════════════════════════════════════════════════════════════════════════

class TestBrandCustomerClassification:

    def _r(self, author_id, inbound):
        return {"author_id": author_id, "inbound": inbound}

    def test_brand_outbound_is_brand(self):
        assert is_brand_message(self._r("SpotifyCares", False), "SpotifyCares") is True

    def test_brand_inbound_not_brand(self):
        """inbound=True marks a customer message even if author_id is the brand name."""
        assert is_brand_message(self._r("SpotifyCares", True), "SpotifyCares") is False

    def test_customer_inbound_is_customer(self):
        assert is_customer_message(self._r("115887", True)) is True

    def test_customer_outbound_not_customer(self):
        assert is_customer_message(self._r("115887", False)) is False

    def test_other_brand_outbound_not_spotifycares(self):
        assert is_brand_message(self._r("hulu_support", False), "SpotifyCares") is False

    def test_other_brand_outbound_not_customer(self):
        assert is_customer_message(self._r("hulu_support", False)) is False

    def test_case_sensitivity(self):
        """Brand match is case-sensitive: 'spotifycares' != 'SpotifyCares'."""
        assert is_brand_message(self._r("spotifycares", False), "SpotifyCares") is False

    def test_missing_author_id_not_brand(self):
        assert is_brand_message({"inbound": False}, "SpotifyCares") is False


# ═════════════════════════════════════════════════════════════════════════════
# 2. Spotify Ecosystem Isolation
# ═════════════════════════════════════════════════════════════════════════════

class TestEcosystemIsolation:

    def test_brand_tweet_included(self):
        df = make_df([
            {"tweet_id": 1, "author_id": "SpotifyCares", "inbound": False,
             "in_response_to_tweet_id": None, "created_at": "2017-01-01", "text": "Hi!"},
        ])
        eco = isolate_spotify_ecosystem(df, "SpotifyCares")
        assert 1 in eco["tweet_id"].values

    def test_parent_of_brand_reply_included(self):
        df = make_df([
            {"tweet_id": 100, "author_id": "cust123",      "inbound": True,
             "in_response_to_tweet_id": None, "created_at": "2017-01-01", "text": "Help!"},
            {"tweet_id": 101, "author_id": "SpotifyCares",  "inbound": False,
             "in_response_to_tweet_id": 100,  "created_at": "2017-01-02", "text": "Sure!"},
        ])
        eco = isolate_spotify_ecosystem(df, "SpotifyCares")
        assert set(eco["tweet_id"].values) == {100, 101}

    def test_unrelated_brand_excluded(self):
        df = make_df([
            {"tweet_id": 1,  "author_id": "AppleSupport", "inbound": False,
             "in_response_to_tweet_id": None, "created_at": "2017-01-01", "text": "Hi Apple"},
            {"tweet_id": 10, "author_id": "cust999",       "inbound": True,
             "in_response_to_tweet_id": None, "created_at": "2017-01-01", "text": "Help Spotify!"},
            {"tweet_id": 2,  "author_id": "SpotifyCares",  "inbound": False,
             "in_response_to_tweet_id": 10,   "created_at": "2017-01-01", "text": "SpotifyCares reply"},
        ])
        eco = isolate_spotify_ecosystem(df, "SpotifyCares")
        assert 1 not in eco["tweet_id"].values
        assert {2, 10}.issubset(set(eco["tweet_id"].values))

    def test_role_tagging_correct(self):
        df = make_df([
            {"tweet_id": 10, "author_id": "cust1",        "inbound": True,
             "in_response_to_tweet_id": None, "created_at": "2017-01-01", "text": "Issue!"},
            {"tweet_id": 11, "author_id": "SpotifyCares",  "inbound": False,
             "in_response_to_tweet_id": 10,   "created_at": "2017-01-01", "text": "On it!"},
        ])
        eco = isolate_spotify_ecosystem(df, "SpotifyCares")
        assert eco[eco["tweet_id"] == 11].iloc[0]["is_brand"] == True
        assert eco[eco["tweet_id"] == 10].iloc[0]["is_customer"] == True

    @pytest.mark.skipif(not PARQUET_PATH.exists(), reason="spotify_clean.parquet not found")
    def test_empirical_ecosystem_size(self):
        df = pd.read_parquet(PARQUET_PATH)
        assert len(df) == 91_889, f"Expected 91889, got {len(df)}"


# ═════════════════════════════════════════════════════════════════════════════
# 3. Conversation Reconstruction
# ═════════════════════════════════════════════════════════════════════════════

class TestConversationReconstruction:

    def test_root_detection(self):
        df = make_df([
            {"tweet_id": 1, "author_id": "cust",         "inbound": True,
             "in_response_to_tweet_id": None, "text": "Problem!"},
            {"tweet_id": 2, "author_id": "SpotifyCares",  "inbound": False,
             "in_response_to_tweet_id": 1,    "text": "On it!"},
        ])
        convs = reconstruct_conversations(df, "SpotifyCares")
        assert len(convs) == 1
        assert convs[0]["root_tweet_id"] == 1

    def test_bfs_root_is_first_message(self):
        df = make_df([
            {"tweet_id": 1, "author_id": "cust",        "inbound": True,
             "in_response_to_tweet_id": None, "text": "Root"},
            {"tweet_id": 2, "author_id": "SpotifyCares", "inbound": False,
             "in_response_to_tweet_id": 1,    "text": "Reply"},
            {"tweet_id": 3, "author_id": "cust",         "inbound": True,
             "in_response_to_tweet_id": 2,    "text": "Follow-up"},
        ])
        convs = reconstruct_conversations(df, "SpotifyCares")
        assert convs[0]["messages"][0]["tweet_id"] == 1

    def test_cycle_protection_no_infinite_loop(self):
        """Mutual parent cycle must not cause get_root() to loop forever."""
        df = make_df([
            {"tweet_id": 1, "author_id": "cust",        "inbound": True,
             "in_response_to_tweet_id": 2, "text": "A"},
            {"tweet_id": 2, "author_id": "SpotifyCares", "inbound": False,
             "in_response_to_tweet_id": 1, "text": "B"},
        ])
        convs = reconstruct_conversations(df, "SpotifyCares")
        assert isinstance(convs, list)  # must complete

    def test_incomplete_conversation_flagged(self):
        df = make_df([
            {"tweet_id": 1, "author_id": "cust1", "inbound": True,
             "in_response_to_tweet_id": None, "text": "Help!"},
            {"tweet_id": 2, "author_id": "cust1", "inbound": True,
             "in_response_to_tweet_id": 1,    "text": "Anyone?"},
        ])
        convs = reconstruct_conversations(df, "SpotifyCares")
        assert all(not c["is_complete"] for c in convs)

    def test_role_assignment(self):
        df = make_df([
            {"tweet_id": 1, "author_id": "cust",         "inbound": True,
             "in_response_to_tweet_id": None, "text": "Issue"},
            {"tweet_id": 2, "author_id": "SpotifyCares",  "inbound": False,
             "in_response_to_tweet_id": 1,    "text": "Noted"},
            {"tweet_id": 3, "author_id": "hulu_support",  "inbound": False,
             "in_response_to_tweet_id": 1,    "text": "Not us"},
        ])
        convs = reconstruct_conversations(df, "SpotifyCares")
        msg_map = {m["tweet_id"]: m["role"] for c in convs for m in c["messages"]}
        assert msg_map[1] == "customer"
        assert msg_map[2] == "brand"
        assert msg_map[3] == "other_brand"

    def test_null_text_produces_empty_string(self):
        df = make_df([
            {"tweet_id": 1, "author_id": "cust",        "inbound": True,
             "in_response_to_tweet_id": None, "text": float("nan")},
            {"tweet_id": 2, "author_id": "SpotifyCares", "inbound": False,
             "in_response_to_tweet_id": 1,    "text": "Fine!"},
        ])
        convs = reconstruct_conversations(df, "SpotifyCares")
        root_msg = next(m for m in convs[0]["messages"] if m["tweet_id"] == 1)
        assert isinstance(root_msg["text"], str)

    @pytest.mark.skipif(not PARQUET_PATH.exists(), reason="spotify_clean.parquet not found")
    def test_empirical_conversation_count(self):
        df = pd.read_parquet(PARQUET_PATH)
        convs = reconstruct_conversations(df, "SpotifyCares")
        assert len(convs) == 28_280, f"Expected 28280, got {len(convs)}"


# ═════════════════════════════════════════════════════════════════════════════
# 4. Pair Extraction
# ═════════════════════════════════════════════════════════════════════════════

def _simple_convs():
    return [{
        "conversation_id": 1,
        "messages": [
            {"tweet_id": 1, "role": "customer", "author_id": "c1",
             "created_at": "t", "text": "Help!", "in_response_to_tweet_id": None},
            {"tweet_id": 2, "role": "brand", "author_id": "SpotifyCares",
             "created_at": "t", "text": "On it!", "in_response_to_tweet_id": 1},
        ]
    }]


class TestPairExtraction:

    def test_pair_id_format(self):
        pairs_df = build_customer_response_pairs(_simple_convs())
        assert len(pairs_df) == 1
        assert pairs_df.iloc[0]["pair_id"] == "pair_1_2"

    def test_initial_inquiry_flag_true(self):
        pairs_df = build_customer_response_pairs(_simple_convs())
        assert pairs_df.iloc[0]["is_initial_inquiry"] == True

    def test_followup_pair_has_context(self):
        convs = [{
            "conversation_id": 1,
            "messages": [
                {"tweet_id": 1, "role": "customer", "author_id": "c", "created_at": "t",
                 "text": "Hi!", "in_response_to_tweet_id": None},
                {"tweet_id": 2, "role": "brand",    "author_id": "b", "created_at": "t",
                 "text": "What device?", "in_response_to_tweet_id": 1},
                {"tweet_id": 3, "role": "customer", "author_id": "c", "created_at": "t",
                 "text": "iPhone 7", "in_response_to_tweet_id": 2},
                {"tweet_id": 4, "role": "brand",    "author_id": "b", "created_at": "t",
                 "text": "Try this fix.", "in_response_to_tweet_id": 3},
            ]
        }]
        pairs_df = build_customer_response_pairs(convs)
        assert len(pairs_df) == 2
        followup = pairs_df[~pairs_df["is_initial_inquiry"]]
        if len(followup) > 0:
            assert followup.iloc[0]["context_before_customer"] != ""

    def test_empty_conversations(self):
        pairs_df = build_customer_response_pairs([])
        assert isinstance(pairs_df, pd.DataFrame)
        assert len(pairs_df) == 0

    def test_other_brand_reply_not_paired(self):
        convs = [{
            "conversation_id": 1,
            "messages": [
                {"tweet_id": 1, "role": "other_brand", "author_id": "hulu",
                 "created_at": "t", "text": "Hi", "in_response_to_tweet_id": None},
                {"tweet_id": 2, "role": "brand", "author_id": "SpotifyCares",
                 "created_at": "t", "text": "Thanks!", "in_response_to_tweet_id": 1},
            ]
        }]
        pairs_df = build_customer_response_pairs(convs)
        assert len(pairs_df) == 0

    @pytest.mark.skipif(not PAIRS_PATH.exists(), reason="spotify_pairs.parquet not found")
    def test_empirical_pair_count(self):
        df = pd.read_parquet(PAIRS_PATH)
        assert len(df) == 43_092, f"Expected 43092, got {len(df)}"

    @pytest.mark.skipif(not PAIRS_PATH.exists(), reason="spotify_pairs.parquet not found")
    def test_empirical_initial_inquiries(self):
        df = pd.read_parquet(PAIRS_PATH)
        assert int(df["is_initial_inquiry"].sum()) == 26_966


# ═════════════════════════════════════════════════════════════════════════════
# 5. Intent Taxonomy Validation
# ═════════════════════════════════════════════════════════════════════════════

EXPECTED_INTENTS = {
    "playback_streaming_issue", "offline_downloads_issue",
    "account_access_credentials", "billing_subscription_payment",
    "plan_management_discount", "content_catalog_licensing",
    "playlist_library_curation", "app_technical_device",
    "unclear_insufficient_context",
    "other_miscellaneous",
}


class TestTaxonomyValidation:

    def test_all_expected_intents_present(self):
        assert set(INTENT_TAXONOMY.keys()) == EXPECTED_INTENTS

    def test_no_extra_intents(self):
        extra = set(INTENT_TAXONOMY.keys()) - EXPECTED_INTENTS
        assert extra == set(), f"Extra intents: {extra}"

    def test_all_patterns_valid_regex(self):
        for intent_name, defn in INTENT_TAXONOMY.items():
            for p in defn.patterns:
                try:
                    re.compile(p, re.IGNORECASE)
                except re.error as e:
                    pytest.fail(f"Bad regex in {intent_name}: {p!r} — {e}")

    def test_required_fields_populated(self):
        for name, defn in INTENT_TAXONOMY.items():
            assert defn.name == name
            assert defn.display_name
            assert defn.description
            assert defn.actionability

    def test_representative_examples_present(self):
        for name, defn in INTENT_TAXONOMY.items():
            assert len(defn.representative_examples) >= 1, f"{name} has no examples"

    def test_escalation_flags_are_bool(self):
        for name, defn in INTENT_TAXONOMY.items():
            assert isinstance(defn.escalation_recommended, bool), \
                f"{name}.escalation_recommended is not bool"

    def test_difficulty_valid(self):
        valid = {"Easy", "Moderate", "Hard"}
        for name, defn in INTENT_TAXONOMY.items():
            assert defn.difficulty in valid, f"{name} has invalid difficulty {defn.difficulty!r}"

    def test_get_intent_returns_correct_object(self):
        defn = get_intent("billing_subscription_payment")
        assert defn is not None
        assert defn.name == "billing_subscription_payment"

    def test_get_intent_unknown_returns_none(self):
        assert get_intent("nonexistent_xyz") is None


# ═════════════════════════════════════════════════════════════════════════════
# 6. Response Classifier
# ═════════════════════════════════════════════════════════════════════════════

class TestResponseClassifier:

    def test_dm_redirect(self):
        assert classify_response_type("Hey! Send us a DM and we'll look into this.") == "Redirect / DM"

    def test_dm_redirect_backstage(self):
        assert classify_response_type("Drop us a message via backstage.") == "Redirect / DM"

    def test_troubleshooting_reinstall(self):
        assert classify_response_type("Try restarting your device and doing a clean reinstall.") == "Troubleshooting / Actionable"

    def test_troubleshooting_cache(self):
        assert classify_response_type("Log out, clear the cache, then log back in.") == "Troubleshooting / Actionable"

    def test_clarification(self):
        assert classify_response_type("Which device are you using and what version of the app?") == "Clarification / Request for Info"

    def test_info_explanation(self):
        assert classify_response_type("This depends on the rights holder and licensing in your region.") == "Information / Explanation"

    def test_acknowledgement(self):
        assert classify_response_type("You're welcome! Glad to hear it's working. Rock on!") == "Generic Acknowledgement"

    def test_other_fallback(self):
        assert classify_response_type("x") == "Other / Miscellaneous"

    def test_troubleshooting_beats_dm(self):
        """If response has both troubleshooting steps AND a DM redirect, troubleshooting wins."""
        text = "Try restarting the app, and if that doesn't help, send us a DM!"
        assert classify_response_type(text) == "Troubleshooting / Actionable"

    def test_feature_keys_complete(self):
        keys = set(extract_response_features("test").keys())
        expected = {
            "has_dm_redirect", "has_clarification", "has_troubleshooting",
            "has_external_redirect", "has_info_explanation", "has_acknowledgement", "has_link",
        }
        assert keys == expected

    def test_response_types_has_7_values(self):
        assert len(RESPONSE_TYPES) == 7

    def test_link_detected(self):
        assert extract_response_features("See https://t.co/ldfdzrinat")["has_link"] is True

    def test_no_link(self):
        assert extract_response_features("Just restart your device.")["has_link"] is False

    def test_empty_string_does_not_crash(self):
        result = classify_response_type("")
        assert result in RESPONSE_TYPES


# ═════════════════════════════════════════════════════════════════════════════
# 7. Intent Classifier
# ═════════════════════════════════════════════════════════════════════════════

class TestIntentClassifier:

    def test_playback_detected(self):
        assert "playback_streaming_issue" in detect_candidate_intents(
            "Spotify keeps stopping every few seconds when I try to listen!"
        )

    def test_billing_classified(self):
        assert classify_customer_intent("I was charged twice this month and need a refund!") == "billing_subscription_payment"

    def test_offline_beats_playback(self):
        assert classify_customer_intent("My downloaded songs won't play in offline mode on the train.") == "offline_downloads_issue"

    def test_account_access(self):
        assert classify_customer_intent("Can't log into my account, password reset email never arrives.") == "account_access_credentials"

    def test_unclear_vague(self):
        assert classify_customer_intent("it isn't working") == "unclear_insufficient_context"

    def test_unclear_very_short(self):
        assert classify_customer_intent("why") == "unclear_insufficient_context"

    def test_other_misc_unmatched(self):
        # Generic social praise with no technical Spotify support keywords whatsoever
        text = "Congratulations on winning the award tonight this is such great news and very well deserved"
        assert classify_customer_intent(text) == "other_miscellaneous"

    def test_playlist_detected(self):
        assert "playlist_library_curation" in detect_candidate_intents(
            "I accidentally deleted my favourite playlist. Can I restore it?"
        )

    def test_app_crash_detected(self):
        assert "app_technical_device" in detect_candidate_intents(
            "Spotify crashes immediately every time I tap the icon on iOS."
        )

    def test_content_catalog_detected(self):
        assert "content_catalog_licensing" in detect_candidate_intents(
            "Half of the songs on this album are greyed out and unplayable."
        )

    def test_billing_beats_plan_with_charged(self):
        text = "I was charged for the family plan but the invite link expired."
        assert classify_customer_intent(text) == "billing_subscription_payment"

    def test_all_returned_candidates_valid(self):
        text = "My downloaded playlists don't play offline and I can't log in and I was charged"
        for c in detect_candidate_intents(text):
            assert c in INTENT_TAXONOMY, f"Unknown intent: {c}"

    def test_context_fallback(self):
        """Very vague text with context should not crash."""
        result = classify_customer_intent("Same issue here", context="Spotify crashes on my phone.")
        assert isinstance(result, str)
        assert result in INTENT_TAXONOMY or result == "other_miscellaneous"

    @pytest.mark.skipif(not PAIRS_PATH.exists(), reason="spotify_pairs.parquet not found")
    def test_initial_inquiries_intent_sum_equals_population(self):
        """Verify mathematical integrity: intent sum across initial inquiries must strictly equal 26,966."""
        df = pd.read_parquet(PAIRS_PATH)
        inquiries = df[df["is_initial_inquiry"]]
        counts = inquiries["customer_primary_intent"].value_counts()
        assert counts.sum() == len(inquiries) == 26_966

    @pytest.mark.skipif(not PAIRS_PATH.exists(), reason="spotify_pairs.parquet not found")
    def test_all_pairs_intent_sum_equals_population(self):
        """Verify mathematical integrity: intent sum across all pairs must strictly equal 43,092."""
        df = pd.read_parquet(PAIRS_PATH)
        counts = df["customer_primary_intent"].value_counts()
        assert counts.sum() == len(df) == 43_092

    @pytest.mark.skipif(not PAIRS_PATH.exists(), reason="spotify_pairs.parquet not found")
    def test_response_type_sum_equals_population(self):
        """Verify mathematical integrity: response_type sum across all pairs must strictly equal 43,092 with 0 nulls."""
        df = pd.read_parquet(PAIRS_PATH)
        assert df["response_type"].isna().sum() == 0
        assert df["response_type"].value_counts().sum() == 43_092


# ═════════════════════════════════════════════════════════════════════════════
# 8. Malformed Data Handling
# ═════════════════════════════════════════════════════════════════════════════

class TestMalformedDataHandling:

    def test_nan_parent_id_treated_as_root(self):
        df = make_df([
            {"tweet_id": 1, "author_id": "cust",        "inbound": True,
             "in_response_to_tweet_id": None, "text": "Problem"},
            {"tweet_id": 2, "author_id": "SpotifyCares", "inbound": False,
             "in_response_to_tweet_id": 1,    "text": "Hi!"},
        ])
        convs = reconstruct_conversations(df, "SpotifyCares")
        assert len(convs) >= 1

    def test_float_nan_in_response_to_tweet_id(self, tmp_path):
        csv_text = (
            "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
            "1,cust1,True,2017-01-01,Help!,,\n"
            "2,SpotifyCares,False,2017-01-02,Sure!,,1.0\n"
        )
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_text)
        df = load_raw_data(str(csv_file))
        assert pd.isna(df.iloc[0]["in_response_to_tweet_id"])
        assert int(df.iloc[1]["in_response_to_tweet_id"]) == 1

    def test_empty_text_response_classifier(self):
        assert classify_response_type("") in RESPONSE_TYPES

    def test_nan_text_intent_classifier(self):
        result = classify_customer_intent(float("nan"))
        assert isinstance(result, str)

    def test_very_long_text_classifier(self):
        long_text = "issue " * 2000
        assert classify_response_type(long_text) in RESPONSE_TYPES

    def test_orphan_brand_reply_no_crash(self):
        """Brand tweet with no parent in dataset must not crash reconstruction."""
        df = make_df([
            {"tweet_id": 2, "author_id": "SpotifyCares", "inbound": False,
             "in_response_to_tweet_id": None, "text": "Hi!"},
        ])
        convs = reconstruct_conversations(df, "SpotifyCares")
        assert isinstance(convs, list)
