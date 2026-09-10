"""
Phase 3: Freeze Human-Annotated Golden Evaluation Set.

Takes the reviewed examples from data/golden/golden_candidates.csv,
validates all human labels against taxonomy constraints, checks completeness,
and exports the frozen data/golden/golden_set.csv and data/golden/golden_set.sha256.

Usage:
    python scripts/freeze_golden_set.py
"""
import os
import sys
import hashlib
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

CANDIDATES_CSV = "data/golden/golden_candidates.csv"
GOLDEN_SET_CSV = "data/golden/golden_set.csv"
GOLDEN_SET_HASH = "data/golden/golden_set.sha256"
GOLDEN_README = "data/golden/README.md"

VALID_INTENTS = {
    "other_miscellaneous",
    "unclear_insufficient_context",
    "billing_subscription_payment",
    "playlist_library_curation",
    "account_access_credentials",
    "app_technical_device",
    "plan_management_discount",
    "playback_streaming_issue",
    "offline_downloads_issue",
    "content_catalog_licensing"
}


def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def freeze_golden_set():
    print("=" * 75)
    print("PHASE 3: FREEZING HUMAN-ANNOTATED GOLDEN EVALUATION SET")
    print("=" * 75)

    if not os.path.exists(CANDIDATES_CSV):
        print(f"Error: {CANDIDATES_CSV} not found.")
        return

    df = pd.read_csv(CANDIDATES_CSV)
    reviewed = df[df["annotation_status"] == "reviewed"].copy()

    print(f"Total candidates in pool: {len(df)}")
    print(f"Total reviewed candidates: {len(reviewed)}")

    if len(reviewed) < 150:
        print(f"\n[Validation Error] Minimum required golden set size is 150 examples.")
        print(f"Currently only {len(reviewed)} examples have annotation_status == 'reviewed'.")
        print("Please complete human annotation first via scripts/annotate_golden_set.py or Excel.")
        return

    # Cap at target ~200 if reviewed > 200, or use all reviewed up to 250
    if len(reviewed) > 250:
        print(f"Selecting first 200-250 reviewed examples (target 200)...")
        final_set = reviewed.head(200).copy()
    else:
        final_set = reviewed.copy()

    # Reset IDs
    final_set["example_id"] = [f"gold_{i+1:04d}" for i in range(len(final_set))]

    # Strictly validate every label
    for idx, row in final_set.iterrows():
        intent = str(row["human_primary_intent"]).strip()
        if intent not in VALID_INTENTS:
            raise ValueError(f"Invalid human_primary_intent '{intent}' in {row['example_id']}")

    # Export columns (do NOT include existing_heuristic_intent to prevent leakage)
    golden_cols = [
        "example_id",
        "pair_id",
        "conversation_id",
        "customer_message",
        "context_before_customer",
        "brand_response",
        "is_initial_inquiry",
        "human_primary_intent",
        "human_secondary_intents",
        "human_is_ambiguous",
        "human_is_multi_intent",
        "human_notes"
    ]

    export_df = final_set[golden_cols]
    export_df.to_csv(GOLDEN_SET_CSV, index=False, encoding="utf-8")

    # Compute hash
    hash_val = compute_sha256(GOLDEN_SET_CSV)
    with open(GOLDEN_SET_HASH, "w", encoding="utf-8") as f:
        f.write(f"{hash_val}  golden_set.csv\n")

    print(f"\n[Success] Frozen golden evaluation set exported to {GOLDEN_SET_CSV}")
    print(f"Total frozen examples: {len(export_df)}")
    print(f"SHA-256 Hash: {hash_val}")
    print(f"Saved hash to {GOLDEN_SET_HASH}")
    print("=" * 75)


if __name__ == "__main__":
    freeze_golden_set()
