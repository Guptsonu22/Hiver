"""Tests for the MACHINE (non-human) calibration path.

Integrity guarantees (synthetic data where possible, read-only real data):
  - machine labels never enter human_calibration.csv
  - no human_confirmed/human_* markers on machine labels
  - machine outputs clearly marked (method/source/machine_* prefix)
  - no leakage vs 322 golden IDs; pairs subset of reviewed 150
  - scorer accepts no golden/human labels (signature + synthetic-only + determinism)
  - official run_judge gate still requires genuine 100/100
  - machine mode never writes judge_metrics.json/judge_outputs.jsonl
"""
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.machine_score import (
    MACHINE_DIMS,
    METHOD,
    heuristic_reply_scores,
    score_inputs_frame,
    summarize_machine_baseline,
)

GOLDEN_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"
HUMAN_CSV = ROOT / "data" / "judge" / "human_calibration.csv"
MACHINE_CSV = ROOT / "data" / "judge" / "machine_calibration_100.csv"
JUDGE_INPUTS = ROOT / "data" / "judge" / "judge_inputs_full150.jsonl"

_rj_spec = importlib.util.spec_from_file_location(
    "run_judge_mod", ROOT / "scripts" / "run_judge.py")
rj = importlib.util.module_from_spec(_rj_spec)
_rj_spec.loader.exec_module(rj)


def test_scorer_signature_accepts_no_human_or_golden_labels():
    params = set(inspect.signature(heuristic_reply_scores).parameters)
    assert params == {"customer_message", "generated_reply", "retrieved_evidence"}
    assert not any("human" in p or "golden" in p or "label" in p for p in params)


def test_scorer_deterministic_and_bounded():
    kw = dict(customer_message="my music will not play at all",
              generated_reply="try reinstalling the app now",
              retrieved_evidence="try reinstalling the app now please")
    a = heuristic_reply_scores(**kw)
    b = heuristic_reply_scores(**kw)
    assert a == b
    for dim in MACHINE_DIMS:
        assert isinstance(a[dim], int) and 0 <= a[dim] <= 3
    assert isinstance(a["machine_reason"], str) and a["machine_reason"].strip()


def test_scorer_groundedness_rewards_verbatim_evidence():
    # Extractive replies match evidence by construction -> documented consequence.
    s = heuristic_reply_scores("help", "try reinstalling the app",
                               "try reinstalling the app")
    assert s["machine_groundedness"] == 3


def test_scorer_flags_unsupported_url():
    s = heuristic_reply_scores("help", "visit http://evil.example now please sir",
                               "please send us a dm for help")
    assert s["machine_unsupported_claims"] == 2
    clean = heuristic_reply_scores("help", "please send us a dm for help",
                                   "please send us a dm for help")
    assert clean["machine_unsupported_claims"] == 0


def test_machine_file_schema_and_marking():
    m = pd.read_csv(MACHINE_CSV, dtype="object").fillna("")
    assert len(m) == 100
    assert not any(c.startswith("human_") for c in m.columns)
    assert "human_confirmed" not in m.columns and "annotation_status" not in m.columns
    assert (m["method"] == METHOD).all()
    assert m["source"].astype(str).str.len().gt(0).all()
    assert m["machine_reason"].astype(str).str.len().gt(0).all()
    for c in MACHINE_DIMS:
        assert m[c].astype(str).str.strip().isin(["0", "1", "2", "3"]).all()
    assert m["pair_id"].nunique() == 50
    assert set(m["baseline"].unique()) == {"MostFrequentBaseline", "KeywordBaseline"}


def test_machine_pairs_subset_reviewed_no_leakage():
    g = pd.read_csv(GOLDEN_CSV, dtype={"pair_id": str})
    golden_ids = set(g["pair_id"])
    reviewed = set(g[g["annotation_status"] == "reviewed"]["pair_id"])
    pending = set(g[g["annotation_status"] == "pending"]["pair_id"])
    m = pd.read_csv(MACHINE_CSV, dtype="object")
    assert set(m["pair_id"].astype(str)) <= reviewed
    assert set(m["pair_id"].astype(str)).isdisjoint(pending)
    assert set(m["retrieval_source_pair_id"].astype(str)).isdisjoint(golden_ids)


def test_generator_never_touches_human_csv(tmp_path, monkeypatch):
    import scripts.generate_machine_calibration as gen  # noqa: import-outside-toplevel
    before = hashlib.sha256(HUMAN_CSV.read_bytes()).hexdigest()
    out_csv = tmp_path / "machine_calibration_100.csv"
    out_cache = tmp_path / "machine_llm_cache.jsonl"
    monkeypatch.setattr(gen, "MACHINE_CSV", out_csv)
    monkeypatch.setattr(gen, "MACHINE_LLM_CACHE", out_cache)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    gen.main()
    after = hashlib.sha256(HUMAN_CSV.read_bytes()).hexdigest()
    assert before == after
    out = pd.read_csv(out_csv, dtype="object").fillna("")
    assert len(out) == 100
    assert (out["source"].str.contains("deterministic_heuristic_v1")).all()


def test_official_gate_still_requires_genuine_human_calibration():
    # Real human file is 0/100 -> official path must STOP.
    with pytest.raises(SystemExit):
        rj.verify_human_calibration()


def test_machine_mode_never_writes_official_outputs(tmp_path, monkeypatch):
    fake_results = tmp_path / "results"
    fake_results.mkdir()
    monkeypatch.setattr(rj, "MACHINE_METRICS_JSON", fake_results / "machine_metrics.json")
    monkeypatch.setattr(rj, "MACHINE_CSV", MACHINE_CSV)  # real machine file, read-only
    rj.run_machine_mode()
    assert (fake_results / "machine_metrics.json").exists()
    assert not (fake_results / "judge_metrics.json").exists()
    assert not (fake_results / "judge_outputs.jsonl").exists()
    import json
    metrics = json.loads((fake_results / "machine_metrics.json").read_text(encoding="utf-8"))
    assert metrics["metadata"]["kind"].startswith("MACHINE-GENERATED DIAGNOSTIC")
    assert "MostFrequentBaseline" in metrics and "KeywordBaseline" in metrics


def test_score_frame_and_summarize_helpers():
    df = pd.DataFrame([
        {"pair_id": "p1", "baseline": "KeywordBaseline", "customer_message": "help play",
         "generated_reply": "try reinstalling now please", "retrieved_historical_response": "try reinstalling now please"},
        {"pair_id": "p2", "baseline": "KeywordBaseline", "customer_message": "help play",
         "generated_reply": "ok", "retrieved_historical_response": "try reinstalling now please"},
    ])
    scored = score_inputs_frame(df)
    assert len(scored) == 2
    assert (scored["method"] == METHOD).all()
    summ = summarize_machine_baseline(scored, "KeywordBaseline")
    # Row 1 is verbatim evidence -> groundedness 3; row 2 ("ok") shares no
    # content words with evidence -> groundedness 0.
    assert summ["machine_groundedness"]["n"] == 2
    assert summ["machine_groundedness"]["distribution"] == {"0": 1, "1": 0, "2": 0, "3": 1}
    assert summ["machine_groundedness"]["mean"] == pytest.approx(1.5)
