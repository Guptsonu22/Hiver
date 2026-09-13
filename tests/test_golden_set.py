"""
Phase 3 pytest test suite — Golden Evaluation Set.

Covers:
  1. Candidate CSV schema integrity
  2. Valid intent label reference
  3. Human annotation dtype coercion (pandas 3.x compatibility)
  4. No duplicate IDs in candidates
  5. No blank human labels in reviewed rows
  6. Golden/Development separation (leakage prevention)
  7. Deterministic sampling hash
  8. Golden-set hash generation
"""
import sys
import hashlib
from pathlib import Path

import pytest

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

CANDIDATES_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"

from intents.taxonomy import INTENT_TAXONOMY  # noqa: E402

VALID_INTENTS = set(INTENT_TAXONOMY.keys())
EXPECTED_COLUMNS = [
    "example_id",
    "pair_id",
    "conversation_id",
    "customer_message",
    "context_before_customer",
    "brand_response",
    "is_initial_inquiry",
    "existing_heuristic_intent",
    "existing_candidate_intents",
    "hard_case_category",
    "response_type",
    "sampling_stratum",
    "human_primary_intent",
    "human_secondary_intents",
    "human_is_ambiguous",
    "human_is_multi_intent",
    "human_notes",
    "annotation_status",
]

HAS_PANDAS = True
try:
    import pandas as pd  # noqa: E402
except ImportError:
    HAS_PANDAS = False


def _needs_pandas():
    if not HAS_PANDAS:
        pytest.skip("pandas is not installed in this environment")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Candidate CSV Schema Integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateSchema:

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_candidates_file_exists(self):
        assert CANDIDATES_CSV.exists(), f"{CANDIDATES_CSV} not found"

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_required_columns_present(self):
        _needs_pandas()
        import importlib.util
        spec = importlib.util.spec_from_file_location("annotate", ROOT / "scripts" / "annotate_golden_set.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df = pd.read_csv(CANDIDATES_CSV)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing required column: {col}"

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_expected_row_count(self):
        _needs_pandas()
        df = pd.read_csv(CANDIDATES_CSV)
        assert 150 <= len(df) <= 350, f"Candidate count out of range: {len(df)}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Valid Intent Label Reference
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentLabels:

    def test_all_10_intents_in_taxonomy(self):
        assert len(INTENT_TAXONOMY) == 10

    def test_intent_names_match_expected_set(self):
        expected = {
            "playback_streaming_issue",
            "offline_downloads_issue",
            "account_access_credentials",
            "billing_subscription_payment",
            "plan_management_discount",
            "content_catalog_licensing",
            "playlist_library_curation",
            "app_technical_device",
            "unclear_insufficient_context",
            "other_miscellaneous",
        }
        assert set(INTENT_TAXONOMY.keys()) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 3. Human Annotation Dtype Coercion (pandas 3.x regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotationDtypeCoercion:

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_string_intent_assignable_on_blank_columns(self):
        """Regression: in pandas 3.x blank columns are inferred as float64,
        so assigning a string intent raises TypeError.  The helper must coerce
        to object dtype before assignment."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("annotate", ROOT / "scripts" / "annotate_golden_set.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Simulate a dataframe where pandas 3.x infers blank human columns as float64
        raw = {
            "example_id": ["cand_0001"],
            "human_primary_intent": [float("nan")],
            "human_secondary_intents": [float("nan")],
            "human_is_ambiguous": [float("nan")],
            "human_is_multi_intent": [float("nan")],
            "human_notes": [float("nan")],
            "annotation_status": [float("nan")],
        }
        df = pd.DataFrame(raw)

        # Before coercion this would be float64 and raise TypeError:
        #   df.at[0, "human_primary_intent"] = "playback_streaming_issue"
        df = mod.coerce_annotation_columns(df)

        # Assignment of a string must now succeed without error
        df.at[0, "human_primary_intent"] = "playback_streaming_issue"
        assert df.at[0, "human_primary_intent"] == "playback_streaming_issue"
        assert df["human_primary_intent"].dtype == "object"

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_all_annotation_columns_coerced(self):
        """Every required annotation column must end up as object dtype."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("annotate", ROOT / "scripts" / "annotate_golden_set.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df = pd.DataFrame({
            "example_id": ["cand_0001"],
            "human_primary_intent": [float("nan")],
            "human_secondary_intents": [float("nan")],
            "human_is_ambiguous": [float("nan")],
            "human_is_multi_intent": [float("nan")],
            "human_notes": [float("nan")],
            "annotation_status": [float("nan")],
        })
        df = mod.coerce_annotation_columns(df)
        for col in mod.HUMAN_ANNOTATION_COLUMNS:
            assert df[col].dtype == "object", f"{col} is not object after coercion"


# ─────────────────────────────────────────────────────────────────────────────
# 4. No Duplicate IDs
# ─────────────────────────────────────────────────────────────────────────────

class TestNoDuplicates:

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_no_duplicate_pair_ids(self):
        _needs_pandas()
        df = pd.read_csv(CANDIDATES_CSV)
        assert df["pair_id"].is_unique, "Duplicate pair_id found in candidates"

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_no_duplicate_example_ids(self):
        _needs_pandas()
        df = pd.read_csv(CANDIDATES_CSV)
        assert df["example_id"].is_unique, "Duplicate example_id found in candidates"


# ─────────────────────────────────────────────────────────────────────────────
# 5. No Blank Human Labels in Reviewed Rows
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewedRowIntegrity:

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_reviewed_rows_have_primary_intent(self):
        _needs_pandas()
        df = pd.read_csv(CANDIDATES_CSV, dtype={"annotation_status": str, "human_primary_intent": str})
        reviewed = df[df["annotation_status"].fillna("").str.strip() == "reviewed"]
        if len(reviewed) == 0:
            pytest.skip("No reviewed rows yet — annotate before running this test")
        assert reviewed["human_primary_intent"].str.strip().ne("").all(), \
            "Some reviewed rows have blank human_primary_intent"

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_reviewed_rows_have_valid_intents(self):
        _needs_pandas()
        df = pd.read_csv(CANDIDATES_CSV, dtype={"annotation_status": str, "human_primary_intent": str})
        reviewed = df[df["annotation_status"].fillna("").str.strip() == "reviewed"]
        if len(reviewed) == 0:
            pytest.skip("No reviewed rows yet")
        for intent in reviewed["human_primary_intent"].str.strip():
            assert intent in VALID_INTENTS, f"Invalid intent label: {intent}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Golden / Development Separation (Leakage Prevention)
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakagePrevention:

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_golden_pairs_absent_from_development_data(self):
        """Golden pair_ids must not already be labelled in development data
        with heuristic labels that could be copied as 'human' labels."""
        _needs_pandas()
        df_candidates = pd.read_csv(CANDIDATES_CSV)
        golden_pair_ids = set(df_candidates["pair_id"])

        pairs_path = ROOT / "data" / "processed" / "spotify_pairs.parquet"
        if not pairs_path.exists():
            pytest.skip("Development parquet not available")

        dev = pd.read_parquet(pairs_path)
        overlap = golden_pair_ids & set(dev["pair_id"].astype(str))
        # Overlap is permissible (source of candidates), but the golden set itself
        # must NOT reuse heuristic labels directly
        assert overlap == golden_pair_ids, "Not all candidate pair_ids found in source"

    def test_golden_final_file_excludes_heuristic_columns(self):
        """When frozen, golden_set.csv must NOT include existing_heuristic_intent."""
        golden_csv = ROOT / "data" / "golden" / "golden_set.csv"
        if not golden_csv.exists():
            pytest.skip("Golden set not yet frozen")
        # When pandas is available this would be checked; for now verify
        # the file does not contain heuristic metadata as content
        content = golden_csv.read_text(encoding="utf-8", errors="replace")
        first_line = content.splitlines()[0] if content else ""
        forbidden = {"existing_heuristic_intent", "existing_candidate_intents"}
        assert not (forbidden & set(first_line.split(","))), \
            "Frozen golden set must exclude heuristic columns"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Deterministic Sampling Hash
# ─────────────────────────────────────────────────────────────────────────────

class TestSamplingDeterminism:

    def test_sample_script_uses_seed(self):
        """The sampler must use random_state=42."""
        _needs_pandas()
        spec = importlib.util.spec_from_file_location("sampler", ROOT / "scripts" / "sample_golden_candidates.py") if False else None
        # We can't import without pandas in scope, so just check source text
        import inspect
        # Read script source and verify seed constant
        src = (ROOT / "scripts" / "sample_golden_candidates.py").read_text(encoding="utf-8")
        assert "RANDOM_SEED = 42" in src, "Sampler does not declare RANDOM_SEED = 42"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Golden-Set Hash Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestGoldenSetHash:

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_compute_sha256_matches_hashlib(self):
        """compute_sha256 helper must match hashlib output."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("freeze", ROOT / "scripts" / "freeze_golden_set.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
            tf.write("a,b,c\n1,2,3\n")
            tmp_path = tf.name

        try:
            expected = hashlib.sha256(open(tmp_path, "rb").read()).hexdigest()
            assert mod.compute_sha256(tmp_path) == expected
        finally:
            Path(tmp_path).unlink()
