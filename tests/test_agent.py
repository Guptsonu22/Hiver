"""Unit tests for the leakage-safe AI Support Agent (src/agent.py).

Uses small synthetic pools only — never real golden data, never fabricated
production results. Real-data guards assert leakage freedom on repo data.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import (
    CLARIFICATION_REPLY,
    SupportAgent,
    TfidfRetriever,
)
from src.intents.taxonomy import INTENT_NAMES

GOLDEN_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"
PAIRS_PARQUET = ROOT / "data" / "processed" / "spotify_pairs.parquet"


@pytest.fixture
def toy_pool() -> pd.DataFrame:
    rows = []
    for i in range(4):
        rows.append({
            "pair_id": f"train_play_{i}",
            "customer_message": "my music keeps buffering and pausing every few seconds",
            "context_before_customer": "",
            "brand_response": "Hi! Try clearing the app cache and restarting playback.",
            "customer_primary_intent": "playback_streaming_issue",
            "has_dm_redirect": False,
        })
    for i in range(3):
        rows.append({
            "pair_id": f"train_bill_{i}",
            "customer_message": "i was charged twice on my credit card this month refund",
            "context_before_customer": "",
            "brand_response": "Hey! Please send us a DM with your account email.",
            "customer_primary_intent": "billing_subscription_payment",
            "has_dm_redirect": True,
        })
    return pd.DataFrame(rows)


GOLDEN_IDS = {"pair_gold_1", "pair_gold_2"}


def test_retriever_is_deterministic(toy_pool):
    r = TfidfRetriever(toy_pool, golden_pair_ids=GOLDEN_IDS)
    a = r.retrieve("music keeps pausing buffering")
    b = r.retrieve("music keeps pausing buffering")
    assert [e.pair_id for e in a] == [e.pair_id for e in b]
    assert all(a[i].similarity >= a[i + 1].similarity for i in range(len(a) - 1))


def test_retriever_never_returns_golden_ids(toy_pool):
    r = TfidfRetriever(toy_pool, golden_pair_ids=GOLDEN_IDS)
    ev = r.retrieve("charged twice refund", top_k=5)
    assert {e.pair_id for e in ev}.isdisjoint(GOLDEN_IDS)
    assert len(ev) == 5  # requested top_k honored (pool has 7 rows)


def test_retriever_fit_rejects_golden_pool(toy_pool):
    dirty = toy_pool.copy()
    dirty.loc[0, "pair_id"] = "pair_gold_1"
    with pytest.raises(AssertionError, match="LEAKAGE"):
        TfidfRetriever(dirty, golden_pair_ids=GOLDEN_IDS)


def test_evidence_schema(toy_pool):
    agent = SupportAgent(toy_pool, golden_pair_ids=GOLDEN_IDS)
    out = agent.handle("my music keeps buffering and pausing")
    assert out["leakage_assertion"].startswith("PASSED")
    for e in out["retrieved_evidence"]:
        for k in ("pair_id", "similarity", "customer_message", "brand_response",
                  "customer_primary_intent", "has_dm_redirect"):
            assert k in e
    assert all(e["pair_id"] not in GOLDEN_IDS for e in out["retrieved_evidence"])


def test_structured_output_schema(toy_pool):
    agent = SupportAgent(toy_pool, golden_pair_ids=GOLDEN_IDS)
    out = agent.handle("charged twice on my card")
    assert out["predicted_intent"] in INTENT_NAMES
    assert isinstance(out["generated_reply"], str) and out["generated_reply"].strip()
    esc = out["escalation"]
    assert esc["decision"] in ("auto_handle", "escalate")
    assert isinstance(esc["reason"], str) and esc["reason"].strip()
    assert 0.0 <= esc["confidence"] <= 1.0


def test_security_financial_escalates(toy_pool):
    agent = SupportAgent(toy_pool, golden_pair_ids=GOLDEN_IDS)
    out = agent.handle("i was charged twice on my credit card, need refund")
    assert out["predicted_intent"] == "billing_subscription_payment"
    assert out["escalation"]["decision"] == "escalate"


def test_low_confidence_escalates_with_clarification(toy_pool):
    agent = SupportAgent(toy_pool, golden_pair_ids=GOLDEN_IDS)
    out = agent.handle("zebra quantum pineapple unrelated words here")
    assert out["escalation"]["decision"] == "escalate"
    assert out["generated_reply"] == CLARIFICATION_REPLY


def test_extractive_reply_is_grounded(toy_pool):
    agent = SupportAgent(toy_pool, golden_pair_ids=GOLDEN_IDS)
    out = agent.handle("my music keeps buffering and pausing every few seconds")
    assert out["generated_reply"] in set(toy_pool["brand_response"])


def test_llm_generator_fail_closed_without_key(toy_pool, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = SupportAgent(toy_pool, golden_pair_ids=GOLDEN_IDS, use_llm=True)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        agent.handle("my music keeps buffering")


def test_no_escalation_accuracy_claim_possible():
    # Guard: agent module must not define escalation accuracy helpers
    # (no human escalation ground truth exists).
    import src.agent as am
    src = Path(am.__file__).read_text(encoding="utf-8").lower()
    assert "escalation_accuracy" not in src


@pytest.mark.skipif(not (GOLDEN_CSV.exists() and PAIRS_PARQUET.exists()),
                    reason="data files not found")
def test_real_pool_retrieval_is_leakage_free():
    from src.evaluate import load_data, prepare_leakage_free_split
    golden, pairs = load_data(GOLDEN_CSV, PAIRS_PARQUET)
    train_df, test_df, _, _ = prepare_leakage_free_split(golden, pairs)
    golden_ids = set(golden["pair_id"].astype(str))
    agent = SupportAgent(train_df, golden_pair_ids=golden_ids)
    assert len(test_df) == 150
    for _, row in test_df.head(5).iterrows():
        out = agent.handle(str(row["customer_message"]),
                           "" if pd.isna(row["context_before_customer"]) else str(row["context_before_customer"]))
        assert {e["pair_id"] for e in out["retrieved_evidence"]}.isdisjoint(golden_ids)
