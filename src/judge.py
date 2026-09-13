"""Phase 4: LLM-as-judge reply-quality evaluation with human agreement.

Frozen constraints (DO NOT violate):
  - Golden intent annotations in data/golden/golden_candidates.csv are FROZEN.
  - Evaluation population is EXACTLY the 150 reviewed golden examples.
  - 172 pending examples are NEVER evaluated.
  - Retrieval MUST be leakage-free: no retrieval_source_pair_id may belong
    to the 322 golden pair IDs. Runtime assertions enforce this.
  - Human calibration labels and judge scores MUST NEVER be fabricated.
    Without OPENAI_API_KEY the judge fails clearly instead of inventing scores.

Rubric (RUBRIC_VERSION v1):
  relevance, helpfulness, groundedness, appropriateness: 0=poor..3=strong
  unsupported_claims: 0=no meaningful unsupported claim (best),
                      3=severe/fabricated claim (worst)
  overall_score: 0-3 (holistic reply quality)
  confidence: 0-3 (judge self-confidence)

Note on unsupported_claims direction: unlike the first four dimensions where
higher is better, for unsupported_claims LOWER is better (0 = clean).
Agreement statistics are computed per dimension regardless of direction.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"
PAIRS_PARQUET = ROOT / "data" / "processed" / "spotify_pairs.parquet"
JUDGE_DIR = ROOT / "data" / "judge"
RESULTS_DIR = ROOT / "results"
JUDGE_INPUTS_JSONL = JUDGE_DIR / "judge_inputs_full150.jsonl"
CALIBRATION_CSV = JUDGE_DIR / "calibration_sample.csv"
HUMAN_CALIBRATION_CSV = JUDGE_DIR / "human_calibration.csv"
JUDGE_OUTPUTS_JSONL = RESULTS_DIR / "judge_outputs.jsonl"
JUDGE_METRICS_JSON = RESULTS_DIR / "judge_metrics.json"

RUBRIC_VERSION = "v1"
JUDGE_PROMPT_VERSION = "judge-prompt-v1"
CALIBRATION_SEED = 42
CALIBRATION_SIZE = 50
EVALUATION_SIZE = 150

JUDGE_DIMENSIONS = [
    "relevance",
    "helpfulness",
    "groundedness",
    "appropriateness",
    "unsupported_claims",
    "overall_score",
]

REQUIRED_JUDGE_FIELDS = [
    "relevance",
    "helpfulness",
    "groundedness",
    "appropriateness",
    "unsupported_claims",
    "overall_score",
    "confidence",
    "reason",
]

RUBRIC_TEXT = """Reply-quality rubric (RUBRIC v1). Score each dimension 0-3.

1. RELEVANCE (0=poor .. 3=strong): Does the reply address the customer's actual issue?
2. HELPFULNESS / ACTIONABILITY (0=poor .. 3=strong): Useful next steps, explanation, or resolution?
3. GROUNDEDNESS (0=poor .. 3=strong): Consistent with evidence from historical SpotifyCares responses?
4. APPROPRIATENESS (0=poor .. 3=strong): Tone and content appropriate for customer support?
5. UNSUPPORTED_CLAIMS (INVERTED: 0=best/no meaningful unsupported claim, 1=minor, 2=significant, 3=severe/fabricated):
   Does the reply invent policies, facts, guarantees, links, or actions not supported by evidence?
OVERALL_SCORE (0=poor .. 3=strong): Holistic reply quality.
CONFIDENCE (0=low .. 3=high): Judge self-confidence.
"""


# ---------------------------------------------------------------------------
# Schema validation / parsing
# ---------------------------------------------------------------------------

def validate_judge_score(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a judge score dict. Raises ValueError/TypeError on any violation."""
    if not isinstance(obj, dict):
        raise TypeError("Judge score must be a dict.")
    for field in REQUIRED_JUDGE_FIELDS:
        if field not in obj:
            raise ValueError(f"Missing required judge field: '{field}'.")
    for dim in JUDGE_DIMENSIONS + ["confidence"]:
        val = obj[dim]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ValueError(f"Field '{dim}' must be an int 0-3, got {val!r}.")
        if val < 0 or val > 3:
            raise ValueError(f"Field '{dim}' out of range 0-3: {val}.")
    reason = obj["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Field 'reason' must be a non-empty string.")
    return obj


def parse_judge_json(raw: str) -> Dict[str, Any]:
    """Strictly parse raw LLM output into a validated judge score dict.

    Raises ValueError on malformed JSON or schema violations. Never fabricates.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Empty judge output.")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed judge JSON: {e}") from e
    return validate_judge_score(obj)


# ---------------------------------------------------------------------------
# Model / API configuration (never hardcode keys)
# ---------------------------------------------------------------------------

def get_judge_config() -> Dict[str, str]:
    """Return judge model config from environment.

    Raises RuntimeError with an actionable message if OPENAI_API_KEY is missing.
    Never returns fabricated scores.
    """
    # Fail-closed: refuse to invent scores when no key is configured.
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your environment "
            "(e.g. $env:OPENAI_API_KEY='...' on Windows) before running the "
            "LLM judge. Refusing to fabricate judge scores."
        )
    model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    return {"model": model, "api_key": api_key, "base_url": base_url}


def build_judge_prompt(
    customer_message: str,
    context_before_customer: str,
    generated_reply: str,
    retrieved_evidence: str,
    reference_response: Optional[str] = None,
    predicted_intent: Optional[str] = None,
) -> str:
    """Build the judge prompt. Evidence is always included (fail-closed otherwise)."""
    if not str(generated_reply or "").strip():
        raise ValueError("generated_reply must be non-empty for judging.")
    if not str(retrieved_evidence or "").strip():
        raise ValueError("retrieved_evidence must be non-empty for judging.")
    parts = [
        "You are an impartial customer-support reply-quality judge for SpotifyCares.",
        RUBRIC_TEXT,
        "Return ONLY strict JSON with keys: relevance, helpfulness, groundedness, "
        "appropriateness, unsupported_claims, overall_score, confidence, reason. "
        "All scores are integers 0-3. 'reason' is one concise evidence-based sentence. "
        "Do not include chain-of-thought.",
        f"Predicted intent: {predicted_intent}" if predicted_intent else "Predicted intent: (not provided)",
        f"Conversation context:\n{(context_before_customer or '').strip() or '(none)'}",
        f"Customer message:\n{str(customer_message).strip()}",
        f"Generated reply (EVALUATE THIS):\n{str(generated_reply).strip()}",
        f"Historical evidence (leakage-free retrieved SpotifyCares response):\n{str(retrieved_evidence).strip()}",
    ]
    if reference_response and str(reference_response).strip():
        parts.append(
            "Historical golden reference response (for context only, NOT retrieval data):\n"
            + str(reference_response).strip()
        )
    return "\n\n---\n\n".join(parts)


def call_llm_judge(prompt: str, model: Optional[str] = None, timeout_s: int = 60) -> Dict[str, Any]:
    """Call the configured OpenAI-compatible chat API and return a validated score.

    Uses only the stdlib (urllib) so no extra dependency is required.
    Raises RuntimeError if no API key; ValueError on malformed model output.
    """
    import urllib.request

    cfg = get_judge_config()
    use_model = (model or cfg["model"]).strip()
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    body = {
        "model": use_model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only reply-quality judge."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"LLM judge API call failed (model={use_model}): {e}") from e
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected judge API response shape: {e}") from e
    return parse_judge_json(content)


# ---------------------------------------------------------------------------
# Leakage-free evidence construction
# ---------------------------------------------------------------------------

def find_retrieval_source_pair_id(training_pool: pd.DataFrame, intent: str) -> str:
    """Deterministically locate the source pair_id behind most_frequent_response().

    Among training rows with customer_primary_intent == intent whose brand_response
    equals the deterministic most-frequent response, return the smallest pair_id
    (lexical order). Raises if none found.
    """
    from src.baselines import most_frequent_response

    top = most_frequent_response(training_pool, intent)
    subset = training_pool[training_pool["customer_primary_intent"] == intent]
    cands = subset[subset["brand_response"].astype(str).str.strip() == top]
    if cands.empty:
        raise ValueError(f"No retrieval source found for intent '{intent}'.")
    return sorted(cands["pair_id"].astype(str).tolist())[0]


def build_evaluation_frame(
    golden_reviewed_df: pd.DataFrame,
    training_pool: pd.DataFrame,
    golden_pair_ids: Set[str],
    baseline_name: str,
    baseline_obj: Any,
) -> pd.DataFrame:
    """Build one row per reviewed example with generated reply + grounding evidence.

    Asserts: exactly 150 rows, all reviewed, no pending, every
    retrieval_source_pair_id NOT in golden_pair_ids.
    """
    if len(golden_reviewed_df) != EVALUATION_SIZE:
        raise ValueError(f"Evaluation population must be exactly 150, got {len(golden_reviewed_df)}.")
    if not (golden_reviewed_df["annotation_status"] == "reviewed").all():
        raise ValueError("Evaluation frame must contain only reviewed examples.")

    messages = golden_reviewed_df["customer_message"].astype(str).tolist()
    contexts = golden_reviewed_df["context_before_customer"].fillna("").astype(str).tolist()
    pred_intents = baseline_obj.predict(messages, contexts)
    replies = baseline_obj.reply(messages, contexts)
    assert len(pred_intents) == len(replies) == EVALUATION_SIZE

    rows: List[Dict[str, Any]] = []
    for (_, grow), pred, reply in zip(golden_reviewed_df.iterrows(), pred_intents, replies):
        source_id = find_retrieval_source_pair_id(training_pool, pred)
        # Runtime leakage assertion (STEP 3 requirement).
        assert source_id not in golden_pair_ids, (
            f"LEAKAGE DETECTED: retrieval_source_pair_id {source_id} belongs to golden set!"
        )
        ev_row = training_pool[training_pool["pair_id"].astype(str) == source_id]
        retrieved_resp = str(ev_row["brand_response"].iloc[0]) if not ev_row.empty else str(reply)
        rows.append(
            {
                "pair_id": str(grow["pair_id"]),
                "baseline": baseline_name,
                "customer_message": str(grow["customer_message"]),
                "context_before_customer": "" if pd.isna(grow["context_before_customer"]) else str(grow["context_before_customer"]),
                "human_primary_intent": str(grow["human_primary_intent"]),
                "predicted_intent": str(pred),
                "generated_reply": str(reply),
                "retrieval_source_pair_id": str(source_id),
                "retrieved_historical_response": str(retrieved_resp),
                "reference_response": str(grow["brand_response"]),
            }
        )
    frame = pd.DataFrame(rows)
    overlap = set(frame["retrieval_source_pair_id"].astype(str)) & set(golden_pair_ids)
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} retrieval sources in golden set!"
    return frame


# ---------------------------------------------------------------------------
# Deterministic calibration sampling
# ---------------------------------------------------------------------------

def sample_calibration_set(
    reviewed_df: pd.DataFrame,
    n: int = CALIBRATION_SIZE,
    seed: int = CALIBRATION_SEED,
) -> pd.DataFrame:
    """Deterministically sample n examples from the 150 reviewed set.

    Raises if reviewed_df is not exactly 150 reviewed rows or contains pending.
    """
    if len(reviewed_df) != EVALUATION_SIZE:
        raise ValueError(f"Calibration source must be exactly 150 reviewed rows, got {len(reviewed_df)}.")
    if not (reviewed_df["annotation_status"] == "reviewed").all():
        raise ValueError("Calibration source must contain only reviewed examples.")
    if n != CALIBRATION_SIZE:
        raise ValueError(f"Calibration size must be exactly {CALIBRATION_SIZE}, got {n}.")
    return reviewed_df.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------

def exact_agreement_rate(human: List[int], judge: List[int]) -> float:
    """Fraction of exact matches on 0-3 ordinal ratings."""
    if len(human) != len(judge) or len(human) == 0:
        raise ValueError("human and judge lists must be non-empty and equal length.")
    return sum(1 for h, j in zip(human, judge) if int(h) == int(j)) / len(human)


def weighted_kappa(human: List[int], judge: List[int], n_categories: int = 4) -> float:
    """Linear-weighted Cohen's kappa for ordinal 0-3 ratings.

    Implements the standard definition with linear disagreement weights
    w[i][j] = |i - j| / (K - 1). Returns 0.0 when chance agreement is 1
    (degenerate), 1.0 on perfect agreement. Pure-python/numpy, no sklearn.
    """
    h = [int(x) for x in human]
    j = [int(x) for x in judge]
    if len(h) != len(j) or len(h) == 0:
        raise ValueError("human and judge lists must be non-empty and equal length.")
    K = int(n_categories)
    for v in h + j:
        if v < 0 or v >= K:
            raise ValueError(f"Rating {v} outside 0-{K - 1}.")
    n = len(h)
    # Observed confusion matrix.
    O = np.zeros((K, K), dtype=float)
    for a, b in zip(h, j):
        O[a, b] += 1
    O /= n
    r = O.sum(axis=1)
    c = O.sum(axis=0)
    E = np.outer(r, c)
    W = np.fromfunction(lambda i, jj: np.abs(i - jj) / (K - 1), (K, K))
    po = float((W * O).sum())
    pe = float((W * E).sum())
    if pe >= 1.0 - 1e-12:
        return 1.0 if po <= 1e-12 else 0.0
    return float(1.0 - po / (1.0 - pe)) if (1.0 - pe) != 0 else 0.0


def compute_agreement(
    human_scores: pd.DataFrame,
    judge_scores: pd.DataFrame,
    join_key: str = "pair_id",
    baseline_col: str = "baseline",
) -> Dict[str, Any]:
    """Compute per-dimension exact agreement + linear-weighted kappa.

    Both frames must cover the same (pair_id, baseline) keys. Returns dict with
    per-dimension stats, overall means, and weakest dimension by kappa.
    """
    h = human_scores.copy()
    j = judge_scores.copy()
    h_keys = set(zip(h[join_key].astype(str), h[baseline_col].astype(str)))
    j_keys = set(zip(j[join_key].astype(str), j[baseline_col].astype(str)))
    if h_keys != j_keys:
        raise ValueError(
            f"Human/judge key mismatch: human={len(h_keys)}, judge={len(j_keys)}, "
            f"overlap={len(h_keys & j_keys)}."
        )
    merged = h.merge(j, on=[join_key, baseline_col], suffixes=("_human", "_judge"))
    per_dim: Dict[str, Dict[str, float]] = {}
    for dim in JUDGE_DIMENSIONS:
        hl = merged[f"{dim}_human"].astype(int).tolist()
        jl = merged[f"{dim}_judge"].astype(int).tolist()
        per_dim[dim] = {
            "n": len(hl),
            "exact_agreement": round(exact_agreement_rate(hl, jl), 4),
            "weighted_kappa": round(weighted_kappa(hl, jl), 4),
        }
    overall_exact = round(float(np.mean([v["exact_agreement"] for v in per_dim.values()])), 4)
    overall_kappa = round(float(np.mean([v["weighted_kappa"] for v in per_dim.values()])), 4)
    weakest = min(per_dim, key=lambda d: per_dim[d]["weighted_kappa"])
    return {
        "n": len(merged),
        "per_dimension": per_dim,
        "overall_exact_agreement": overall_exact,
        "overall_weighted_kappa": overall_kappa,
        "weakest_dimension_by_kappa": weakest,
    }


# ---------------------------------------------------------------------------
# Cache helpers (never silently overwrite)
# ---------------------------------------------------------------------------

def load_cached_outputs(path: Path = JUDGE_OUTPUTS_JSONL) -> Dict[str, Dict[str, Any]]:
    """Load cached judge outputs keyed by 'baseline::pair_id'. Returns {} if absent."""
    if not Path(path).exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[f"{obj['baseline']}::{obj['pair_id']}"] = obj
    return cache


def append_cached_output(obj: Dict[str, Any], path: Path = JUDGE_OUTPUTS_JSONL) -> None:
    """Append one judge output record to the JSONL cache."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
