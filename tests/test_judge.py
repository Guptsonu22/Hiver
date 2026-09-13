"""Phase 4 tests: LLM-as-judge + human agreement evaluation.

Covers STEP 15 requirements using toy/mock data only (never fake real results):
  schema validation, score ranges, malformed JSON, deterministic sampling,
  calibration size/membership, leakage freedom, evidence presence,
  no hardcoded API key, cache loading, agreement + kappa correctness,
  full population size.

Real-data tests are skipped unless data files exist.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.judge import (
    CALIBRATION_SEED,
    CALIBRATION_SIZE,
    EVALUATION_SIZE,
    JUDGE_DIMENSIONS,
    build_judge_prompt,
    exact_agreement_rate,
    get_judge_config,
    load_cached_outputs,
    parse_judge_json,
    sample_calibration_set,
    validate_judge_score,
    weighted_kappa,
    compute_agreement,
    find_retrieval_source_pair_id,
    build_evaluation_frame,
)

GOLDEN_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"
PAIRS_PARQUET = ROOT / "data" / "processed" / "spotify_pairs.parquet"


def valid_score(**overrides):
    base = {
        "relevance": 2,
        "helpfulness": 2,
        "groundedness": 2,
        "appropriateness": 3,
        "unsupported_claims": 0,
        "overall_score": 2,
        "confidence": 2,
        "reason": "Reply addresses the issue and matches evidence.",
    }
    base.update(overrides)
    return base


# --- schema tests ---

def test_valid_judge_schema_accepted():
    assert validate_judge_score(valid_score()) == valid_score()


def test_score_range_0_to_3_enforced():
    for dim in JUDGE_DIMENSIONS + ["confidence"]:
        with pytest.raises(ValueError):
            validate_judge_score(valid_score(**{dim: 4}))
        with pytest.raises(ValueError):
            validate_judge_score(valid_score(**{dim: -1}))
        with pytest.raises(ValueError):
            validate_judge_score(valid_score(**{dim: "2"}))
    # Boundaries 0 and 3 are legal.
    s = valid_score()
    for dim in JUDGE_DIMENSIONS + ["confidence"]:
        s[dim] = 0
    validate_judge_score(s)
    for dim in JUDGE_DIMENSIONS + ["confidence"]:
        s[dim] = 3
    validate_judge_score(s)


def test_missing_fields_rejected():
    for field in ["relevance", "helpfulness", "groundedness", "appropriateness",
                  "unsupported_claims", "overall_score", "confidence", "reason"]:
        bad = valid_score()
        del bad[field]
        with pytest.raises(ValueError):
            validate_judge_score(bad)


def test_malformed_json_rejected():
    with pytest.raises(ValueError):
        parse_judge_json("{not valid json")
    with pytest.raises(ValueError):
        parse_judge_json("")
    with pytest.raises(ValueError):
        parse_judge_json(json.dumps({"relevance": 2}))  # missing fields
    parsed = parse_judge_json(json.dumps(valid_score()))
    assert parsed["overall_score"] == 2


# --- sampling tests (toy reviewed frame) ---

def _toy_reviewed(n=150):
    return pd.DataFrame([{
        "pair_id": f"pair_{i}",
        "customer_message": f"msg {i}",
        "context_before_customer": "",
        "brand_response": f"ref {i}",
        "human_primary_intent": "playback_streaming_issue",
        "annotation_status": "reviewed" if i < 150 else "pending",
    } for i in range(n)])


def test_deterministic_calibration_sampling():
    df = _toy_reviewed(150)
    a = sample_calibration_set(df, n=50, seed=CALIBRATION_SEED)
    b = sample_calibration_set(df, n=50, seed=CALIBRATION_SEED)
    assert a["pair_id"].tolist() == b["pair_id"].tolist()


def test_exactly_50_calibration_examples():
    df = _toy_reviewed(150)
    calib = sample_calibration_set(df, n=50, seed=CALIBRATION_SEED)
    assert len(calib) == CALIBRATION_SIZE == 50


def test_calibration_examples_all_belong_to_reviewed_150():
    df = _toy_reviewed(150)
    calib = sample_calibration_set(df, n=50, seed=CALIBRATION_SEED)
    assert set(calib["pair_id"]) <= set(df["pair_id"])
    assert (calib["annotation_status"] == "reviewed").all()


def test_no_pending_examples_in_calibration():
    df = _toy_reviewed(150)
    # Inject a pending row to prove the sampler rejects mixed sources.
    mixed = pd.concat([df.iloc[:149],
                       pd.DataFrame([{"pair_id": "pair_pend", "customer_message": "x",
                                      "context_before_customer": "",
                                      "brand_response": "y",
                                      "human_primary_intent": "other_miscellaneous",
                                      "annotation_status": "pending"}])],
                      ignore_index=True)
    assert len(mixed) == 150
    with pytest.raises(ValueError):
        sample_calibration_set(mixed, n=50, seed=CALIBRATION_SEED)


def test_full_evaluation_population_is_exactly_150():
    df = _toy_reviewed(150)
    assert len(df) == EVALUATION_SIZE == 150
    with pytest.raises(ValueError):
        sample_calibration_set(_toy_reviewed(149), n=50, seed=CALIBRATION_SEED)


# --- leakage / evidence tests ---

def _toy_pool_and_baseline():
    from src.baselines import KeywordBaseline
    pool = pd.DataFrame([
        {"pair_id": f"train_{i}", "customer_message": "m", "context_before_customer": "",
         "brand_response": "Evidence reply A", "customer_primary_intent": "playback_streaming_issue"}
        for i in range(3)
    ] + [
        {"pair_id": "train_b", "customer_message": "m", "context_before_customer": "",
         "brand_response": "Evidence reply B", "customer_primary_intent": "billing_subscription_payment"}
    ])
    golden_ids = {"pair_gold_1", "pair_gold_2"}
    base = KeywordBaseline(training_pool=pool, golden_pair_ids=golden_ids)
    return pool, base, golden_ids


def test_no_golden_retrieval_pair_used():
    pool, base, golden_ids = _toy_pool_and_baseline()
    src = find_retrieval_source_pair_id(pool, "playback_streaming_issue")
    assert src not in golden_ids
    assert src in set(pool["pair_id"])


def test_judge_input_contains_evidence():
    prompt = build_judge_prompt(
        customer_message="My music won't play",
        context_before_customer="",
        generated_reply="Try reinstalling.",
        retrieved_evidence="Historical: try a clean reinstall.",
        predicted_intent="playback_streaming_issue",
    )
    assert "Historical" in prompt and "Try reinstalling." in prompt
    with pytest.raises(ValueError):
        build_judge_prompt("msg", "", "reply", "")  # empty evidence rejected
    with pytest.raises(ValueError):
        build_judge_prompt("msg", "", "", "evidence")  # empty reply rejected


def test_evaluation_frame_asserts_no_golden_retrieval():
    pool, base, golden_ids = _toy_pool_and_baseline()
    reviewed = pd.DataFrame([{
        "pair_id": f"pair_{i}", "customer_message": "song stopped playing buffering",
        "context_before_customer": "", "brand_response": "ref",
        "human_primary_intent": "playback_streaming_issue", "annotation_status": "reviewed",
    } for i in range(150)])
    frame = build_evaluation_frame(reviewed, pool, golden_ids, "KeywordBaseline", base)
    assert len(frame) == 150
    assert set(frame["retrieval_source_pair_id"]) & golden_ids == set()
    assert (frame["retrieved_historical_response"].astype(str).str.len() > 0).all()
    assert (frame["generated_reply"].astype(str).str.len() > 0).all()


# --- API key / cache tests ---

def test_api_key_is_not_hardcoded(monkeypatch):
    import src.judge as jm
    src_text = Path(jm.__file__).read_text(encoding="utf-8")
    assert "sk-" not in src_text  # no embedded secret
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_judge_config()


def test_cached_output_loading(tmp_path):
    p = tmp_path / "cache.jsonl"
    rec = {"pair_id": "pair_1", "baseline": "KeywordBaseline", "overall_score": 2}
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    cache = load_cached_outputs(p)
    assert cache["KeywordBaseline::pair_1"]["overall_score"] == 2
    assert load_cached_outputs(tmp_path / "missing.jsonl") == {}


# --- agreement tests on known toy data ---

def test_agreement_calculation_correctness_on_known_toy_data():
    human = [0, 1, 2, 3, 2]
    judge = [0, 1, 2, 3, 2]
    assert exact_agreement_rate(human, judge) == 1.0
    assert weighted_kappa(human, judge) == pytest.approx(1.0)
    human2 = [0, 0, 0, 0]
    judge2 = [0, 1, 2, 3]
    assert exact_agreement_rate(human2, judge2) == pytest.approx(0.25)


def test_weighted_kappa_partial_credit():
    # Close misses should score higher than far misses under linear weighting.
    human = [2, 2, 2, 2]
    close = [1, 1, 3, 3]  # all off by 1
    far = [0, 0, 0, 0]    # all off by 2
    assert weighted_kappa(human, close) > weighted_kappa(human, far)


def test_compute_agreement_keys_must_match():
    h = pd.DataFrame([{"pair_id": "p1", "baseline": "KeywordBaseline",
                       **{d + "_human": 2 for d in JUDGE_DIMENSIONS}}])
    j = pd.DataFrame([{"pair_id": "p2", "baseline": "KeywordBaseline",
                       **{d + "_judge": 2 for d in JUDGE_DIMENSIONS}}])
    with pytest.raises(ValueError):
        compute_agreement(h, j)


def test_compute_agreement_perfect_toy():
    rows_h, rows_j = [], []
    for i in range(4):
        rows_h.append({"pair_id": f"p{i}", "baseline": "KeywordBaseline",
                       **{d + "_human": 2 for d in JUDGE_DIMENSIONS}})
        rows_j.append({"pair_id": f"p{i}", "baseline": "KeywordBaseline",
                       **{d + "_judge": 2 for d in JUDGE_DIMENSIONS}})
    res = compute_agreement(pd.DataFrame(rows_h), pd.DataFrame(rows_j))
    assert res["overall_exact_agreement"] == 1.0
    assert res["overall_weighted_kappa"] == pytest.approx(1.0)


# --- real-data population guards (skip without data files) ---

@pytest.mark.skipif(not GOLDEN_CSV.exists(), reason="golden CSV not found")
def test_real_golden_reviewed_is_150_and_pending_excluded():
    df = pd.read_csv(GOLDEN_CSV, dtype={"pair_id": str})
    reviewed = df[df["annotation_status"] == "reviewed"]
    pending = df[df["annotation_status"] == "pending"]
    assert len(reviewed) == 150
    assert len(pending) == 172
    calib_ids = set(reviewed.sample(n=50, random_state=CALIBRATION_SEED)["pair_id"])
    assert calib_ids <= set(reviewed["pair_id"])
    assert calib_ids.isdisjoint(set(pending["pair_id"]))


@pytest.mark.skipif(not (GOLDEN_CSV.exists() and PAIRS_PARQUET.exists()),
                    reason="data files not found")
def test_real_full_inputs_would_be_150():
    from src.evaluate import load_data, prepare_leakage_free_split
    golden, pairs = load_data(GOLDEN_CSV, PAIRS_PARQUET)
    _, test_df, pending_df, v = prepare_leakage_free_split(golden, pairs)
    assert len(test_df) == 150
    assert len(pending_df) == 172
    assert v["overlap_after"] == 0
