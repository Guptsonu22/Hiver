"""
Phase 3: Interactive CLI Helper for Human Annotation of Golden Set Candidates.

Allows the project owner (Sonu) to inspect customer dialogues one by one,
review context, assign primary/secondary intents, record ambiguity/multi-intent flags,
and persist progress safely across sessions.

Usage:
    python scripts/annotate_golden_set.py
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Columns that must hold string/object values for human annotation.
# Explicit coercion prevents pandas 3.x from inferring blank columns as float64,
# which would otherwise raise TypeError when a string intent value is assigned
# via df.at[idx, "human_primary_intent"].
HUMAN_ANNOTATION_COLUMNS = [
    "human_primary_intent",
    "human_secondary_intents",
    "human_is_ambiguous",
    "human_is_multi_intent",
    "human_notes",
    "annotation_status",
]


def coerce_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force every human‑annotation column to `object` dtype so that string
    labels can be assigned without a pandas 3.x float64 dtype error."""
    for col in HUMAN_ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
        else:
            df[col] = df[col].astype("object")
    return df

CANDIDATES_CSV = "data/golden/golden_candidates.csv"

INTENTS = [
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
]


def print_banner():
    print("=" * 75)
    print("SPOTIFYCARES GOLDEN EVALUATION SET — HUMAN ANNOTATION HELPER")
    print("Annotator: Project Owner (Sonu)")
    print("Target: Review candidates and assign ground-truth intent labels.")
    print("=" * 75)


def save_progress(df):
    df.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8")
    reviewed_count = (df["annotation_status"] == "reviewed").sum()
    print(f"\n[Progress Saved] {reviewed_count} / {len(df)} candidates reviewed.")


def main():
    print_banner()

    if not os.path.exists(CANDIDATES_CSV):
        print(f"Error: {CANDIDATES_CSV} does not exist. Run scripts/sample_golden_candidates.py first.")
        return

    df = pd.read_csv(CANDIDATES_CSV)
    # Ensure all human-annotation columns are object-compatible so that
    # string values (e.g. an intent name) can be assigned via df.at[]
    # without raising TypeError in pandas 3.x.
    df = coerce_annotation_columns(df)

    pending_mask = df["annotation_status"] != "reviewed"
    pending_indices = df[pending_mask].index.tolist()

    reviewed_count = (df["annotation_status"] == "reviewed").sum()
    print(f"Total candidates: {len(df)}")
    print(f"Already reviewed: {reviewed_count}")
    print(f"Pending review  : {len(pending_indices)}")

    if not pending_indices:
        print("\nAll candidates have been reviewed! You can now freeze the golden set.")
        return

    print("\nCommands:")
    print("  [1-10] Choose primary intent by number")
    print("  's'    Skip current example")
    print("  'q'    Save progress and quit")
    print("-" * 75)

    for idx in pending_indices:
        row = df.loc[idx]

        print("\n" + "=" * 75)
        print(f"EXAMPLE [{row['example_id']}] ({idx + 1} of {len(df)}) | Turn: {'Initial' if row['is_initial_inquiry'] else 'Follow-up'}")
        print(f"Hard Case Category: {row['hard_case_category']}")
        print(f"Reference Heuristic: {row['existing_heuristic_intent']}")
        print("-" * 75)

        if pd.notna(row["context_before_customer"]) and str(row["context_before_customer"]).strip():
            print("PRECEDING CONVERSATION CONTEXT:")
            for line in str(row["context_before_customer"]).splitlines():
                print(f"  {line}")
            print("-" * 75)

        print(f"CUSTOMER MESSAGE:\n  \"{row['customer_message']}\"")
        print("-" * 75)
        print(f"BRAND RESPONSE:\n  \"{row['brand_response']}\"")
        print("-" * 75)

        print("\nAvailable Intents:")
        for i, name in enumerate(INTENTS, 1):
            print(f"  {i:2d}. {name}")

        while True:
            choice = input("\nEnter primary intent [1-10], 's' to skip, or 'q' to quit: ").strip().lower()

            if choice == "q":
                save_progress(df)
                print("Exiting annotation session. Goodbye!")
                return
            elif choice == "s":
                print("Skipped.")
                break
            elif choice.isdigit() and 1 <= int(choice) <= 10:
                primary_intent = INTENTS[int(choice) - 1]

                # Secondary intents
                sec_input = input("Secondary intent(s) if multi-intent (comma-separated numbers or Enter for none): ").strip()
                sec_intents = []
                if sec_input:
                    for part in sec_input.split(","):
                        part = part.strip()
                        if part.isdigit() and 1 <= int(part) <= 10:
                            sec_intents.append(INTENTS[int(part) - 1])

                # Ambiguous flag
                amb_input = input("Is this example ambiguous / underspecified? [y/N]: ").strip().lower()
                is_amb = "True" if amb_input == "y" else "False"

                # Multi-intent flag
                is_multi = "True" if len(sec_intents) > 0 else "False"

                # Notes
                notes = input("Annotation notes / rationale (optional, press Enter to skip): ").strip()

                # Update row
                df.at[idx, "human_primary_intent"] = primary_intent
                df.at[idx, "human_secondary_intents"] = ",".join(sec_intents)
                df.at[idx, "human_is_ambiguous"] = is_amb
                df.at[idx, "human_is_multi_intent"] = is_multi
                df.at[idx, "human_notes"] = notes
                df.at[idx, "annotation_status"] = "reviewed"

                print(f"\n-> Assigned: {primary_intent} (Ambiguous: {is_amb}, Multi: {is_multi})")
                save_progress(df)
                break
            else:
                print("Invalid input. Please enter a number between 1 and 10, 's', or 'q'.")

    print("\nSession completed!")
    save_progress(df)


if __name__ == "__main__":
    main()
