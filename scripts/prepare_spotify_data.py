"""Step 2 - 6 & 18: Prepare clean SpotifyCares dataset, reconstruct conversations, and extract pairs."""
import os
import sys
import json
import re
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')

from src.data.loader import load_raw_data, isolate_spotify_ecosystem, is_brand_message, is_customer_message
from src.data.conversation_builder import reconstruct_conversations, build_customer_response_pairs
from src.data.response_classifier import classify_response_type, extract_response_features
from src.intents.taxonomy import detect_candidate_intents, classify_customer_intent


def main():
    print("=" * 70)
    print("PHASE 2: PREPARING SPOTIFYCARES CONVERSATION DATASET & PAIRS")
    print("=" * 70)

    raw_csv = "data/raw/twcs.csv"
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw dataset not found at {raw_csv}")

    # 1. Load Raw Dataset
    print("\n[Step 1/6] Loading raw dataset from data/raw/twcs.csv...")
    raw_df = load_raw_data(raw_csv)
    total_tweets = len(raw_df)
    print(f"Loaded {total_tweets:,} total tweets from twcs.csv.")

    # 2. Isolate SpotifyCares Conversation Ecosystem
    print("\n[Step 2/6] Isolating SpotifyCares conversation ecosystem...")
    spotify_df = isolate_spotify_ecosystem(raw_df, brand_name="SpotifyCares")
    print(f"Extracted {len(spotify_df):,} tweets in connected SpotifyCares conversation trees.")

    # Validate Customer vs Brand classification
    customer_count = spotify_df["is_customer"].sum()
    brand_count = spotify_df["is_brand"].sum()
    other_brand_count = len(spotify_df) - customer_count - brand_count
    print(f"  - Customer inbound tweets : {customer_count:,}")
    print(f"  - SpotifyCares brand tweets: {brand_count:,}")
    print(f"  - Co-tagged other brands   : {other_brand_count:,}")

    # Save isolated clean tweets
    os.makedirs("data/processed", exist_ok=True)
    clean_path = "data/processed/spotify_clean.parquet"
    spotify_df.to_parquet(clean_path, index=False)
    print(f"Saved isolated clean tweets to {clean_path}")

    # 3. Reconstruct Conversation Threads
    print("\n[Step 3/6] Reconstructing conversation threads...")
    conversations = reconstruct_conversations(spotify_df, brand_name="SpotifyCares")
    print(f"Reconstructed {len(conversations):,} distinct conversation trees.")

    # Save conversations as JSON Lines for easy inspection
    convs_jsonl_path = "data/processed/spotify_conversations.jsonl"
    with open(convs_jsonl_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")
    print(f"Saved conversation trees to {convs_jsonl_path}")

    # 4. Extract Customer -> Brand Response Pairs
    print("\n[Step 4/6] Extracting Customer -> SpotifyCares response pairs...")
    pairs_df = build_customer_response_pairs(conversations, brand_name="SpotifyCares")
    print(f"Extracted {len(pairs_df):,} direct customer-brand interaction pairs.")

    # 5. Classify Response Types and Extract Features
    print("\n[Step 5/6] Classifying brand response types and extracting features...")
    pairs_df["response_type"] = pairs_df["brand_response"].apply(classify_response_type)
    
    # Feature flags
    feature_dicts = pairs_df["brand_response"].apply(extract_response_features).tolist()
    feat_df = pd.DataFrame(feature_dicts)
    for col in feat_df.columns:
        pairs_df[col] = feat_df[col]

    # Assign intent heuristically to customer inquiry
    pairs_df["customer_candidate_intents"] = pairs_df["customer_message"].apply(
        lambda t: ",".join(detect_candidate_intents(t))
    )
    pairs_df["customer_primary_intent"] = pairs_df.apply(
        lambda r: classify_customer_intent(r["customer_message"], r["context_before_customer"]),
        axis=1
    )

    # Save pairs to Parquet
    pairs_parquet_path = "data/processed/spotify_pairs.parquet"
    pairs_df.to_parquet(pairs_parquet_path, index=False)
    print(f"Saved {len(pairs_df):,} pairs to {pairs_parquet_path}")

    # Save 100 sample rows to CSV for human inspection
    sample_csv_path = "data/processed/spotify_pairs_sample.csv"
    pairs_sample = pairs_df.head(100)
    pairs_sample.to_csv(sample_csv_path, index=False, encoding="utf-8")
    print(f"Saved 100 sample pairs to {sample_csv_path}")

    # 6. Extract Hard Cases Dataset (Step 18)
    print("\n[Step 6/6] Identifying hard, ambiguous, and multi-intent cases...")
    hard_cases = []

    for idx, row in pairs_df.iterrows():
        c_msg = str(row["customer_message"]).strip()
        candidates = detect_candidate_intents(c_msg)
        words = c_msg.split()
        has_context = bool(row["context_before_customer"])

        reason = None
        # Rule 1: Multi-intent
        if len(candidates) >= 2:
            reason = "multi_intent"
        # Rule 2: Context-dependent follow-up
        elif not row["is_initial_inquiry"] and len(words) <= 7 and has_context:
            reason = "context_dependent_followup"
        # Rule 3: Insufficient context / ultra-short complaint
        elif len(words) <= 5 and not candidates:
            reason = "insufficient_context"
        # Rule 4: Sarcasm / slang / rhetorical
        elif re.search(r'\b(wtf|smh|fuck|pissed|bruh|sucks|garbage|trash|fix your shit)\b', c_msg, re.IGNORECASE) and not candidates:
            reason = "sarcasm_slang_hostile"
        # Rule 5: Noisy / URL-heavy with minimal text
        elif len(re.sub(r'https?://\S+|@\S+', '', c_msg).strip()) <= 10 and not candidates:
            reason = "noisy_or_attachment_only"

        if reason is not None:
            hard_cases.append({
                "example_id": f"hard_{len(hard_cases)+1:04d}",
                "pair_id": row["pair_id"],
                "conversation_id": row["conversation_id"],
                "turn_index": row["turn_index"],
                "customer_message": c_msg,
                "reason_difficult": reason,
                "candidate_intents": "|".join(candidates) if candidates else "none",
                "context_available": row["context_before_customer"][:200] if has_context else "None"
            })

    hard_cases_df = pd.DataFrame(hard_cases)
    # Deduplicate and sample a representative distribution
    hard_cases_path = "data/processed/hard_cases.csv"
    hard_cases_df.to_csv(hard_cases_path, index=False, encoding="utf-8")
    print(f"Extracted {len(hard_cases_df):,} hard cases saved to {hard_cases_path}")
    print("\nHard case breakdown by reason:")
    print(hard_cases_df["reason_difficult"].value_counts())

    print("\n" + "=" * 70)
    print("PHASE 2 DATA PREPARATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
