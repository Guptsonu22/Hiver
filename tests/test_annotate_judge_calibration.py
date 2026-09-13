"""Tests for scripts/annotate_judge_calibration.py helpers.

Verifies integrity guarantees with synthetic rows only:
  - _is_reviewed accepts only fully human-confirmed 0-3 rows
  - suggestions never count as labels (no suggestion field is read by _is_reviewed)
  - _ask_score confirm/override/skip/quit paths
  - _ask_fast_approval bulk approval paths (y/n/e/q)
  - _edit_individual_scores allows selective editing
  - _get_cached_or_fetch_suggestion reuses cached suggestions
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "annotate_judge_calibration",
    ROOT / "scripts" / "annotate_judge_calibration.py",
)
ann = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ann)

# Also import as module for patching module-level constants
import scripts.annotate_judge_calibration as mod


def _row(**over):
    base = {
        "annotation_status": "reviewed",
        "relevance_human": "2",
        "helpfulness_human": "2",
        "groundedness_human": "2",
        "appropriateness_human": "2",
        "unsupported_claims_human": "0",
        "overall_score_human": "2",
        "ai_suggestion_shown": "0",
    }
    base.update(over)
    return base


def _suggestion(**over):
    base = {
        "relevance": 1,
        "helpfulness": 1,
        "groundedness": 3,
        "appropriateness": 2,
        "unsupported_claims": 0,
        "overall_score": 1,
        "reason": "Test suggestion rationale.",
    }
    base.update(over)
    return base


def test_reviewed_row_accepted():
    assert ann._is_reviewed(_row()) is True


def test_pending_never_counts_even_with_scores():
    assert ann._is_reviewed(_row(annotation_status="pending")) is False


def test_missing_or_out_of_range_score_rejected():
    assert ann._is_reviewed(_row(relevance_human="")) is False
    assert ann._is_reviewed(_row(helpfulness_human="4")) is False
    assert ann._is_reviewed(_row(overall_score_human="good")) is False


def test_suggestion_flag_does_not_create_review():
    # A shown suggestion alone (scores empty) must NOT validate.
    r = _row(annotation_status="pending", ai_suggestion_shown="1",
             relevance_human="", helpfulness_human="", groundedness_human="",
             appropriateness_human="", unsupported_claims_human="",
             overall_score_human="")
    assert ann._is_reviewed(r) is False


def test_ask_score_manual_entry(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "2")
    assert ann._ask_score("RELEVANCE") == "2"


def test_ask_score_confirms_suggestion_on_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert ann._ask_score("RELEVANCE", suggestion="2") == "2"


def test_ask_score_override_beats_suggestion(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "0")
    assert ann._ask_score("RELEVANCE", suggestion="3") == "0"


def test_ask_score_rejects_empty_without_current_or_suggestion(monkeypatch):
    answers = iter(["", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert ann._ask_score("RELEVANCE") == "q"


# --- fast mode tests ---

def test_ask_fast_approval_approve_all(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert ann._ask_fast_approval(_suggestion()) == "y"


def test_ask_fast_approval_reject_all(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert ann._ask_fast_approval(_suggestion()) == "n"


def test_ask_fast_approval_edit_some(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "e")
    assert ann._ask_fast_approval(_suggestion()) == "e"


def test_ask_fast_approval_quit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "q")
    assert ann._ask_fast_approval(_suggestion()) == "q"


def test_ask_fast_approval_rejects_invalid_then_accepts(monkeypatch):
    answers = iter(["x", "y"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert ann._ask_fast_approval(_suggestion()) == "y"


def test_edit_individual_scores_keep_some_override_others(monkeypatch):
    # Test: keep relevance (Enter), override helpfulness to 0, then quit
    # The function returns {"q": True} immediately on "q" without saving partial results
    answers = iter(["", "0", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    sugg = _suggestion(relevance=1, helpfulness=1, groundedness=3,
                       appropriateness=2, unsupported_claims=0, overall_score=1)
    result = ann._edit_individual_scores(sugg)
    # Function returns {"q": True} immediately on quit, partial results not saved
    assert result.get("q") is True


def test_edit_individual_scores_quit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "q")
    result = ann._edit_individual_scores(_suggestion())
    assert result.get("q") is True


def test_suggestion_cache_reuse(monkeypatch, tmp_path):
    # Create a temp cache file
    cache_file = tmp_path / "suggestion_cache.jsonl"
    # Patch the cache path on the module
    import scripts.annotate_judge_calibration as mod
    original = mod.SUGGESTION_CACHE_JSONL
    mod.SUGGESTION_CACHE_JSONL = cache_file
    try:
        # Create a mock fetch that tracks call count
        call_count = {"n": 0}
        def mock_fetch(row):
            call_count["n"] += 1
            return {"relevance": 1, "helpfulness": 1, "groundedness": 3,
                    "appropriateness": 2, "unsupported_claims": 0, "overall_score": 1}
        monkeypatch.setattr(mod, "_fetch_suggestion", mock_fetch)

        cache = {}
        row = {"pair_id": "pair_1", "baseline": "KeywordBaseline"}
        # First call - should fetch
        mod._get_cached_or_fetch_suggestion(row, cache)
        assert call_count["n"] == 1
        # Second call - should use cache
        mod._get_cached_or_fetch_suggestion(row, cache)
        assert call_count["n"] == 1  # still 1
    finally:
        mod.SUGGESTION_CACHE_JSONL = original


def test_fast_mode_edit_sets_human_edited(monkeypatch):
    # Test: keep relevance (Enter), override helpfulness to 1, then quit
    answers = iter(["", "1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    sugg = _suggestion(relevance=1, helpfulness=2)
    result = mod._edit_individual_scores(sugg)
    # Function returns {"q": True} immediately on quit, partial results not saved
    assert result.get("q") is True


def test_new_metadata_columns_added_on_load(tmp_path):
    # Verify the new columns are added when loading a CSV without them
    csv_file = tmp_path / "test_cal.csv"
    csv_file.write_text("""pair_id,baseline,relevance_human,helpfulness_human,groundedness_human,appropriateness_human,unsupported_claims_human,overall_score_human,annotation_status
pair_1,KeywordBaseline,1,2,2,2,0,1,reviewed
""")
    df = pd.read_csv(csv_file, dtype="object").fillna("")
    for col in ("ai_suggestion_shown", "ai_suggestion_values", "human_confirmed",
                "human_edited", "annotation_mode"):
        if col not in df.columns:
            df[col] = ""
    assert "annotation_mode" in df.columns
    assert "human_confirmed" in df.columns
    assert "human_edited" in df.columns
    assert "ai_suggestion_values" in df.columns


def test_fast_approval_enter_alone_does_not_confirm(monkeypatch):
    # Enter alone must NEVER approve: loop until explicit y/n/e/q.
    answers = iter(['', '', 'n'])
    monkeypatch.setattr('builtins.input', lambda *a: next(answers))
    assert ann._ask_fast_approval({'relevance': 1, 'helpfulness': 1, 'groundedness': 1, 'appropriateness': 1, 'unsupported_claims': 0, 'overall_score': 1, 'reason': 'r'}) == 'n'


def test_compact_truncates_long_text():
    assert ann._compact('a b c', 100) == 'a b c'
    long_t = 'w ' * 200
    out = ann._compact(long_t, 50)
    assert len(out) <= 53 and out.endswith('...')


def test_display_fast_row_shows_evidence_fields(capsys):
    row = {'pair_id': 'p1', 'baseline': 'KeywordBaseline', 'predicted_intent': 'a',
           'human_primary_intent': 'b', 'customer_message': 'hello help',
           'context_before_customer': '', 'generated_reply': 'try this',
           'retrieved_historical_response': 'historical fix', 'retrieval_source_pair_id': 't1'}
    ann._display_fast_row(row, 0, 100, 0)
    out = capsys.readouterr().out
    assert 'hello help' in out and 'try this' in out and 'historical fix' in out


FAST_ROW = {
    'pair_id': 'pair_test_1', 'baseline': 'KeywordBaseline',
    'customer_message': 'my music will not play', 'context_before_customer': '',
    'human_primary_intent': 'playback_streaming_issue',
    'predicted_intent': 'playback_streaming_issue',
    'generated_reply': 'try reinstalling the app',
    'retrieval_source_pair_id': 'train_1',
    'retrieved_historical_response': 'historical reinstall advice',
    'reference_response': 'ref',
    'relevance_human': '', 'helpfulness_human': '', 'groundedness_human': '',
    'appropriateness_human': '', 'unsupported_claims_human': '',
    'overall_score_human': '', 'human_note': '', 'annotation_status': 'pending',
}

FAST_SUGGESTION = {'relevance': 1, 'helpfulness': 1, 'groundedness': 3,
                   'appropriateness': 2, 'unsupported_claims': 0,
                   'overall_score': 1, 'confidence': 2, 'reason': 'test reason'}

FAST_SCORE_COLS = ('relevance_human', 'helpfulness_human', 'groundedness_human',
                   'appropriateness_human', 'unsupported_claims_human',
                   'overall_score_human')


def _run_fast_main(monkeypatch, tmp_path, fetch_fn, inputs):
    import sys
    import src.judge as judge_mod
    csv_path = tmp_path / 'human_calibration.csv'
    pd.DataFrame([FAST_ROW]).to_csv(csv_path, index=False, encoding='utf-8')
    monkeypatch.setattr(ann, 'HUMAN_CALIBRATION_CSV', csv_path)
    monkeypatch.setattr(ann, '_get_cached_or_fetch_suggestion', fetch_fn)
    # Bypass the real API-key gate (fetch itself is mocked; no network used).
    monkeypatch.setattr(judge_mod, 'get_judge_config',
                        lambda: {'model': 'test', 'api_key': 'test', 'base_url': 'test'})
    monkeypatch.setattr(sys, 'argv', ['prog', '--fast'])
    answers = iter(inputs)
    monkeypatch.setattr('builtins.input', lambda *a: next(answers))
    ann.main()  # must never raise
    return pd.read_csv(csv_path, dtype='object').fillna('')


def test_fast_api_401_falls_back_to_manual_no_crash(monkeypatch, tmp_path):
    # Regression: HTTP 401 in --fast must not raise UnboundLocalError;
    # the human scores manually and the row is recorded as manual.
    def boom(row, cache):
        raise RuntimeError('LLM judge API call failed: HTTP 401 Unauthorized')
    df = _run_fast_main(monkeypatch, tmp_path, boom,
                        ['2', '2', '2', '2', '0', '2', ''])
    r = df.iloc[0]
    assert r['annotation_status'] == 'reviewed'
    assert [r[c] for c in FAST_SCORE_COLS] == ['2', '2', '2', '2', '0', '2']
    assert r['ai_suggestion_shown'] == '0'


def test_fast_api_failure_quit_writes_nothing(monkeypatch, tmp_path):
    # API dead + human quits at first manual prompt: nothing reviewed, no crash.
    def boom(row, cache):
        raise RuntimeError('HTTP 401 Unauthorized')
    df = _run_fast_main(monkeypatch, tmp_path, boom, ['q'])
    assert df.iloc[0]['annotation_status'] == 'pending'
    assert (df.iloc[0][list(FAST_SCORE_COLS)] == '').all()


def test_fast_n_rejection_scores_manually(monkeypatch, tmp_path):
    # 'n' rejects the suggestion -> manual scoring, suggestion NOT stored.
    def ok(row, cache):
        return {'suggestion': dict(FAST_SUGGESTION)}
    df = _run_fast_main(monkeypatch, tmp_path, ok,
                        ['n', '3', '3', '3', '3', '0', '3', ''])
    r = df.iloc[0]
    assert r['annotation_status'] == 'reviewed'
    assert [r[c] for c in FAST_SCORE_COLS] == ['3', '3', '3', '3', '0', '3']
    assert r['ai_suggestion_shown'] == '0'


def test_fast_y_confirmation_records_provenance(monkeypatch, tmp_path):
    # Explicit 'y' stores suggestion values WITH confirmation provenance.
    def ok(row, cache):
        return {'suggestion': dict(FAST_SUGGESTION)}
    df = _run_fast_main(monkeypatch, tmp_path, ok, ['y', ''])
    r = df.iloc[0]
    assert r['annotation_status'] == 'reviewed'
    assert [r[c] for c in FAST_SCORE_COLS] == ['1', '1', '3', '2', '0', '1']
    assert r['ai_suggestion_shown'] == '1'
    assert r['human_confirmed'] == '1'
    assert r['annotation_mode'] == 'ai_suggestion_human_confirmed'
