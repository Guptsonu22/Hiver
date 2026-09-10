"""
Phase 3: Stratified Golden Evaluation Set Candidate Sampler.

Samples ~320 candidate customer-brand dialogue pairs from data/processed/spotify_pairs.parquet
across balanced strata (10 intents, hard cases, multi-turn follow-ups, message lengths)
for human annotation by the project owner.

Deterministic Random Seed: 42
Output: data/golden/golden_candidates.csv
"""
import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

PAIRS_PATH = "data/processed/spotify_pairs.parquet"
HARD_CASES_PATH = "data/processed/hard_cases.csv"
OUTPUT_DIR = "data/golden"
OUTPUT_CANDIDATES = os.path.join(OUTPUT_DIR, "golden_candidates.csv")
RANDOM_SEED = 42


def sample_candidates():
    print("=" * 70)
    print("PHASE 3: STRATIFIED GOLDEN SET CANDIDATE SAMPLING")
    print("=" * 70)

    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(f"Missing {PAIRS_PATH}. Run scripts/prepare_spotify_data.py first.")

    pairs_df = pd.read_parquet(PAIRS_PATH)
    hard_df = pd.read_csv(HARD_CASES_PATH) if os.path.exists(HARD_CASES_PATH) else pd.DataFrame()

    print(f"Loaded {len(pairs_df):,} dialogue pairs.")

    # Map hard cases
    if not hard_df.empty:
        hard_map = dict(zip(hard_df["pair_id"], hard_df["reason_difficult"]))
        pairs_df["hard_case_category"] = pairs_df["pair_id"].map(hard_map).fillna("normal")
    else:
        pairs_df["hard_case_category"] = "normal"

    rng = np.random.RandomState(RANDOM_SEED)
    sampled_pair_ids = set()
    candidate_records = []

    # -------------------------------------------------------------------------
    # Stratum 1: Standard Intent Coverage (Initial Inquiries, Normal Difficulty)
    # Target: ~18 per intent across all 10 heuristic intents = 180
    # -------------------------------------------------------------------------
    intents = sorted(pairs_df["customer_primary_intent"].unique())
    normal_initial = pairs_df[
        (pairs_df["is_initial_inquiry"]) & 
        (pairs_df["hard_case_category"] == "normal")
    ]

    for intent in intents:
        subset = normal_initial[normal_initial["customer_primary_intent"] == intent]
        n_sample = min(18, len(subset))
        chosen = subset.sample(n=n_sample, random_state=rng)
        for _, row in chosen.iterrows():
            sampled_pair_ids.add(row["pair_id"])
            candidate_records.append((row["pair_id"], f"intent_coverage_{intent}"))

    print(f"[Stratum 1] Standard Intent Coverage: {len(candidate_records)} candidates")

    # -------------------------------------------------------------------------
    # Stratum 2: Hard Cases (Ambiguous, Multi-intent, Hostile, Short, Noisy)
    # Target: 112 candidates
    # -------------------------------------------------------------------------
    hard_targets = [
        ("multi_intent", 35),
        ("context_dependent_followup", 35),
        ("insufficient_context", 20),
        ("sarcasm_slang_hostile", 20),
        ("noisy_or_attachment_only", 2)
    ]
    stratum_2_count = 0
    for reason, target in hard_targets:
        subset = pairs_df[
            (pairs_df["hard_case_category"] == reason) & 
            (~pairs_df["pair_id"].isin(sampled_pair_ids))
        ]
        n_sample = min(target, len(subset))
        chosen = subset.sample(n=n_sample, random_state=rng)
        for _, row in chosen.iterrows():
            sampled_pair_ids.add(row["pair_id"])
            candidate_records.append((row["pair_id"], f"hard_case_{reason}"))
            stratum_2_count += 1

    print(f"[Stratum 2] Hard / Edge Cases: {stratum_2_count} candidates")

    # -------------------------------------------------------------------------
    # Stratum 3: Multi-turn Follow-up Conversations
    # Target: 30 candidates (normal multi-turn dialogues with prior context)
    # -------------------------------------------------------------------------
    followups = pairs_df[
        (~pairs_df["is_initial_inquiry"]) & 
        (~pairs_df["pair_id"].isin(sampled_pair_ids))
    ]
    chosen_followups = followups.sample(n=30, random_state=rng)
    for _, row in chosen_followups.iterrows():
        sampled_pair_ids.add(row["pair_id"])
        candidate_records.append((row["pair_id"], "multi_turn_followup"))

    print(f"[Stratum 3] Multi-turn Follow-ups: {len(chosen_followups)} candidates")

    # -------------------------------------------------------------------------
    # Build Final Candidate DataFrame
    # -------------------------------------------------------------------------
    pair_meta_map = dict(candidate_records)
    candidate_df = pairs_df[pairs_df["pair_id"].isin(sampled_pair_ids)].copy()
    candidate_df["sampling_stratum"] = candidate_df["pair_id"].map(pair_meta_map)

    # Sort deterministically by conversation_id, then turn_index
    candidate_df.sort_values(by=["conversation_id", "turn_index"], inplace=True)
    candidate_df.reset_index(drop=True, inplace=True)

    # Format output fields
    candidate_df["example_id"] = [f"cand_{i+1:04d}" for i in range(len(candidate_df))]
    candidate_df["existing_heuristic_intent"] = candidate_df["customer_primary_intent"]
    candidate_df["existing_candidate_intents"] = candidate_df["customer_candidate_intents"]

    # Mandatory blank human annotation columns (MUST NOT BE PRE-POPULATED)
    candidate_df["human_primary_intent"] = ""
    candidate_df["human_secondary_intents"] = ""
    candidate_df["human_is_ambiguous"] = ""
    candidate_df["human_is_multi_intent"] = ""
    candidate_df["human_notes"] = ""
    candidate_df["annotation_status"] = "pending"

    output_columns = [
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
        "annotation_status"
    ]

    final_csv = candidate_df[output_columns]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_csv.to_csv(OUTPUT_CANDIDATES, index=False, encoding="utf-8")

    print(f"\nSuccessfully sampled {len(final_csv)} candidates to {OUTPUT_CANDIDATES}")
    print(f"Unique conversation trees represented: {final_csv['conversation_id'].nunique()}")
    print(f"Initial inquiries: {(final_csv['is_initial_inquiry']).sum()}")
    print(f"Follow-up turns: {(~final_csv['is_initial_inquiry']).sum()}")
    print("\nSampling Strata Breakdown:")
    print(final_csv["sampling_stratum"].value_counts())
    print("\nHard Case Breakdown:")
    print(final_csv["hard_case_category"].value_counts())
    print("=" * 70)


if __name__ == "__main__":
    sample_candidates()
