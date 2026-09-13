"""
Phase 3: assisted human review for golden candidate annotation.

This workflow generates AI/heuristic suggestions for pending candidates, then
presents one example at a time for explicit human approval or correction. It
never copies suggestions into human annotation fields unless the reviewer
accepts or overrides during the interactive review.

Usage:
    python scripts/assist_annotate_golden_set.py
    python scripts/assist_annotate_golden_set.py --batch
    python scripts/assist_annotate_golden_set.py --suggest-only
    python scripts/assist_annotate_golden_set.py --csv path/to/candidates.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intents.taxonomy import (  # noqa: E402
    INTENT_TAXONOMY,
    classify_customer_intent,
    detect_candidate_intents,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

CANDIDATES_CSV = ROOT / "data" / "golden" / "golden_candidates.csv"
TAXONOMY_DOC = ROOT / "docs" / "intent_taxonomy.md"
GUIDELINES_DOC = ROOT / "docs" / "golden_set_annotation_guidelines.md"

INTENTS_ORDERED = [
    "other_miscellaneous",
    "unclear_insufficient_context",
    "billing_subscription_payment",
    "playlist_library_curation",
    "account_access_credentials",
    "app_technical_device",
    "plan_management_discount",
    "playback_streaming_issue",
    "offline_downloads_issue",
    "content_catalog_licensing",
]

INTENT_DISPLAY = {name: INTENT_TAXONOMY[name].display_name for name in INTENTS_ORDERED}

HUMAN_ANNOTATION_COLUMNS = [
    "human_primary_intent",
    "human_secondary_intents",
    "human_is_ambiguous",
    "human_is_multi_intent",
    "human_notes",
    "annotation_status",
    "annotation_method",
]

SUGGESTION_COLUMNS = [
    "suggested_primary_intent",
    "suggested_secondary_intents",
    "suggested_is_ambiguous",
    "suggested_is_multi_intent",
    "suggested_notes",
    "suggestion_confidence",
    "suggestion_reasoning",
]

DIFFICULT_LANGUAGE_PATTERNS = [
    r"\b(lol|lmao|wtf|smh|ugh|bruh|pls|plz|po|unli|promo)\b",
    r"\?\?\?+",
    r"!!!+",
]


def clean_cell(value: object) -> str:
    """Return a stable string for CSV cells, treating NaN as blank."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def coerce_workflow_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure human annotation and suggestion columns can hold strings."""
    for col in HUMAN_ANNOTATION_COLUMNS + SUGGESTION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")
    return df


def pending_mask(df: pd.DataFrame) -> pd.Series:
    """Select rows awaiting human review."""
    return df["annotation_status"].fillna("").astype(str).str.strip().str.lower().eq("pending")


def load_candidates(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return coerce_workflow_columns(df)


def save_candidates(df: pd.DataFrame, csv_path: Path) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8")


def validate_reference_docs() -> None:
    missing = [path for path in (TAXONOMY_DOC, GUIDELINES_DOC) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required annotation reference document(s) missing: {joined}")


def _matched_intents(customer_message: str, context: str) -> tuple[list[str], bool]:
    message_matches = detect_candidate_intents(customer_message)
    if message_matches:
        return message_matches, False

    context_matches = detect_candidate_intents(context)
    if context_matches:
        return context_matches, True

    return [], False


def compute_suggestion(
    customer_message: object,
    context_before_customer: object = "",
    brand_response: object = "",
) -> dict[str, str]:
    """
    Produce a suggestion using the fixed taxonomy and annotation guidelines.

    The customer message is primary evidence; context is used for follow-up
    disambiguation when the message itself is underspecified. Brand response is
    included as reviewer context, but final intent remains grounded in the
    customer's support need.
    """
    message = clean_cell(customer_message)
    context = clean_cell(context_before_customer)
    response = clean_cell(brand_response)
    combined_evidence = "\n".join(part for part in [message, context] if part)

    matches, used_context = _matched_intents(message, context)
    primary = classify_customer_intent(message, context=context)

    if not matches and primary in INTENT_TAXONOMY and primary not in {"unclear_insufficient_context", "other_miscellaneous"}:
        matches = [primary]

    secondary = [intent for intent in matches if intent != primary]
    message_word_count = len(message.split())
    has_context = bool(context)
    has_brand_response = bool(response)
    has_specific_match = primary not in {"unclear_insufficient_context", "other_miscellaneous"}

    is_multi = bool(secondary)
    is_ambiguous = primary == "unclear_insufficient_context" or (
        is_multi and primary not in {"billing_subscription_payment", "account_access_credentials"}
    )

    if primary == "unclear_insufficient_context":
        confidence = "low"
        reasoning = (
            "Customer message and available context lack enough diagnostic detail; "
            "guidelines route this to unclear_insufficient_context."
        )
    elif is_multi:
        confidence = "medium"
        reasoning = (
            f"Multiple taxonomy patterns matched ({', '.join(matches)}); "
            f"guideline priority rules select {primary}."
        )
    elif used_context:
        confidence = "medium"
        reasoning = f"Customer follow-up is underspecified, but preceding context matches {primary}."
    elif has_specific_match:
        confidence = "high"
        reasoning = f"Customer message contains specific evidence matching {primary} in the documented taxonomy."
    else:
        confidence = "low"
        reasoning = (
            "No specific support-domain evidence matched; guidelines keep this as "
            "other_miscellaneous rather than inventing a label."
        )

    notes = []
    if used_context:
        notes.append("Context used because the customer message alone was underspecified.")
    if has_brand_response:
        notes.append("Brand response shown for reviewer orientation only.")
    if message_word_count <= 6 and not has_specific_match and not has_context:
        notes.append("Very short message with no disambiguating context.")

    if combined_evidence and primary == "billing_subscription_payment":
        notes.append("Financial-support category requires human review before final annotation.")
    elif combined_evidence and primary == "account_access_credentials":
        notes.append("Account/security category requires human review before final annotation.")

    return {
        "suggested_primary_intent": primary,
        "suggested_secondary_intents": ",".join(secondary),
        "suggested_is_ambiguous": str(is_ambiguous),
        "suggested_is_multi_intent": str(is_multi),
        "suggested_notes": " ".join(notes),
        "suggestion_confidence": confidence,
        "suggestion_reasoning": reasoning,
    }


def ensure_suggestions_for_pending(df: pd.DataFrame) -> int:
    """Populate missing suggestion fields for pending rows only."""
    count = 0
    for idx, row in df[pending_mask(df)].iterrows():
        if clean_cell(row.get("suggested_primary_intent")):
            continue

        suggestion = compute_suggestion(
            row.get("customer_message", ""),
            row.get("context_before_customer", ""),
            row.get("brand_response", ""),
        )
        for col, value in suggestion.items():
            df.at[idx, col] = value
        count += 1
    return count


def calculate_review_priority(row: pd.Series) -> dict[str, object]:
    """Score how much human attention a pending candidate likely needs."""
    score = 0
    reasons = []

    confidence = clean_cell(row.get("suggestion_confidence")).lower()
    if confidence == "low":
        score += 30
        reasons.append("low confidence")
    elif confidence == "medium":
        score += 12
        reasons.append("medium confidence")

    context = clean_cell(row.get("context_before_customer"))
    hard_case = clean_cell(row.get("hard_case_category")).lower()
    is_initial = clean_cell(row.get("is_initial_inquiry")).lower() == "true"
    suggestion_notes = clean_cell(row.get("suggested_notes")).lower()
    if context and (not is_initial or "context" in hard_case or "context used" in suggestion_notes):
        score += 18
        reasons.append("context-dependent follow-up")

    if clean_cell(row.get("suggested_is_multi_intent")).lower() == "true":
        score += 20
        reasons.append("multi-intent")

    if clean_cell(row.get("suggested_is_ambiguous")).lower() == "true":
        score += 18
        reasons.append("ambiguous/underspecified")

    heuristic = clean_cell(row.get("existing_heuristic_intent"))
    suggested = clean_cell(row.get("suggested_primary_intent"))
    if heuristic and suggested and heuristic != suggested:
        score += 18
        reasons.append("heuristic/suggestion disagreement")

    candidate_intents = [
        part.strip()
        for part in clean_cell(row.get("existing_candidate_intents")).split(",")
        if part.strip()
    ]
    reasoning = clean_cell(row.get("suggestion_reasoning")).lower()
    if len(candidate_intents) > 1 or "no specific" in reasoning or suggested in {
        "unclear_insufficient_context",
        "other_miscellaneous",
    }:
        score += 10
        reasons.append("weak or conflicting evidence")

    text_blob = " ".join(
        [
            clean_cell(row.get("customer_message")),
            context,
            clean_cell(row.get("brand_response")),
        ]
    ).lower()
    if any(re.search(pattern, text_blob) for pattern in DIFFICULT_LANGUAGE_PATTERNS):
        score += 8
        reasons.append("difficult language/slang")

    if "unusual" in hard_case or "edge" in hard_case or len(clean_cell(row.get("customer_message")).split()) <= 3:
        score += 8
        reasons.append("unusual case")

    return {
        "review_risk_score": score,
        "review_risk_reasons": "; ".join(reasons) if reasons else "clear/high-confidence",
    }


def ranked_pending_indices(df: pd.DataFrame) -> list[int]:
    """Return pending row indices sorted high-risk first."""
    scored = []
    for idx, row in df[pending_mask(df)].iterrows():
        priority = calculate_review_priority(row)
        scored.append((idx, int(priority["review_risk_score"])))
    return [idx for idx, _ in sorted(scored, key=lambda item: (-item[1], item[0]))]


def apply_human_approved_suggestion(df: pd.DataFrame, idx: int) -> None:
    """Copy a displayed suggestion into human fields after explicit approval."""
    df.at[idx, "human_primary_intent"] = clean_cell(df.at[idx, "suggested_primary_intent"])
    df.at[idx, "human_secondary_intents"] = clean_cell(df.at[idx, "suggested_secondary_intents"])
    df.at[idx, "human_is_ambiguous"] = clean_cell(df.at[idx, "suggested_is_ambiguous"])
    df.at[idx, "human_is_multi_intent"] = clean_cell(df.at[idx, "suggested_is_multi_intent"])
    df.at[idx, "human_notes"] = clean_cell(df.at[idx, "suggested_notes"])
    df.at[idx, "annotation_status"] = "reviewed"
    df.at[idx, "annotation_method"] = "human_approved_ai_suggestion"


def apply_human_override(
    df: pd.DataFrame,
    idx: int,
    primary: str,
    secondary: str = "",
    is_ambiguous: str = "False",
    notes: str = "",
) -> None:
    """Write reviewer-entered labels after explicit correction."""
    df.at[idx, "human_primary_intent"] = primary
    df.at[idx, "human_secondary_intents"] = secondary
    df.at[idx, "human_is_ambiguous"] = is_ambiguous
    df.at[idx, "human_is_multi_intent"] = str(bool(secondary))
    df.at[idx, "human_notes"] = notes
    df.at[idx, "annotation_status"] = "reviewed"
    df.at[idx, "annotation_method"] = "human_override"


def print_banner(csv_path: Path) -> None:
    print("=" * 75)
    print("SPOTIFYCARES GOLDEN SET - ASSISTED HUMAN ANNOTATION")
    print("Mode: suggestions only until explicit human approval or correction.")
    print(f"Candidates: {csv_path}")
    print(f"Taxonomy:   {TAXONOMY_DOC}")
    print(f"Guidelines: {GUIDELINES_DOC}")
    print("=" * 75)


def print_intent_menu() -> None:
    print("\nAvailable intents:")
    for i, name in enumerate(INTENTS_ORDERED, 1):
        print(f"  {i:2d}. {name} ({INTENT_DISPLAY[name]})")


def print_candidate(row: pd.Series, position: int, total: int) -> None:
    print("\n" + "=" * 75)
    print(f"EXAMPLE [{row.get('example_id', position)}] ({position} of {total})")
    print(f"Turn: {'Initial' if str(row.get('is_initial_inquiry')) == 'True' else 'Follow-up'}")
    print(f"Hard Case Category: {clean_cell(row.get('hard_case_category'))}")
    print(f"Reference Heuristic: {clean_cell(row.get('existing_heuristic_intent'))}")
    print("-" * 75)

    context = clean_cell(row.get("context_before_customer"))
    if context:
        print("PRECEDING CONVERSATION CONTEXT:")
        for line in context.splitlines():
            print(f"  {line}")
        print("-" * 75)

    print(f"CUSTOMER MESSAGE:\n  \"{clean_cell(row.get('customer_message'))}\"")
    print("-" * 75)
    print(f"BRAND RESPONSE:\n  \"{clean_cell(row.get('brand_response'))}\"")
    print("-" * 75)


def print_suggestion(row: pd.Series) -> None:
    primary = clean_cell(row.get("suggested_primary_intent"))
    secondary = clean_cell(row.get("suggested_secondary_intents"))
    notes = clean_cell(row.get("suggested_notes"))

    print("\nSUGGESTION, NOT A HUMAN LABEL:")
    print(f"  Suggested primary:      {primary} ({INTENT_DISPLAY.get(primary, '')})")
    print(f"  Suggested secondary:    {secondary}")
    print(f"  Suggested ambiguous:    {clean_cell(row.get('suggested_is_ambiguous'))}")
    print(f"  Suggested multi-intent: {clean_cell(row.get('suggested_is_multi_intent'))}")
    print(f"  Confidence:             {clean_cell(row.get('suggestion_confidence'))}")
    print(f"  Reasoning:              {clean_cell(row.get('suggestion_reasoning'))}")
    if notes:
        print(f"  Suggested notes:        {notes}")
    print("-" * 75)


def save_progress(df: pd.DataFrame, csv_path: Path) -> None:
    save_candidates(df, csv_path)
    reviewed = df["annotation_status"].fillna("").astype(str).str.strip().eq("reviewed").sum()
    pending = pending_mask(df).sum()
    print(f"\n[Progress saved] reviewed={reviewed}, pending={pending}, total={len(df)}")


def _parse_secondary_numbers(raw: str) -> str:
    secondary = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(INTENTS_ORDERED):
            secondary.append(INTENTS_ORDERED[int(part) - 1])
    return ",".join(secondary)


def print_batch_candidate(row: pd.Series, batch_number: int, position: int, total: int) -> None:
    priority = calculate_review_priority(row)
    primary = clean_cell(row.get("suggested_primary_intent"))

    print("\n" + "=" * 75)
    print(
        f"BATCH {batch_number} ITEM {position}/{total} | "
        f"{clean_cell(row.get('example_id'))} | risk={priority['review_risk_score']}"
    )
    print(f"Risk reasons: {priority['review_risk_reasons']}")
    print(f"Reference heuristic: {clean_cell(row.get('existing_heuristic_intent'))}")
    print("-" * 75)

    context = clean_cell(row.get("context_before_customer"))
    if context:
        print("CONTEXT:")
        for line in context.splitlines():
            print(f"  {line}")
        print("-" * 75)

    print(f"CUSTOMER:\n  \"{clean_cell(row.get('customer_message'))}\"")
    print("-" * 75)
    print(f"BRAND RESPONSE:\n  \"{clean_cell(row.get('brand_response'))}\"")
    print("-" * 75)
    print("SUGGESTION, NOT A HUMAN LABEL:")
    print(f"  Suggested primary:      {primary} ({INTENT_DISPLAY.get(primary, '')})")
    print(f"  Suggested secondary:    {clean_cell(row.get('suggested_secondary_intents'))}")
    print(f"  Suggested ambiguous:    {clean_cell(row.get('suggested_is_ambiguous'))}")
    print(f"  Suggested multi-intent: {clean_cell(row.get('suggested_is_multi_intent'))}")
    print(f"  Confidence:             {clean_cell(row.get('suggestion_confidence'))}")
    print(f"  Reasoning:              {clean_cell(row.get('suggestion_reasoning'))}")


def print_batch_commands(batch_size: int) -> None:
    print("\nBatch commands:")
    print("  y list        approve suggested labels, e.g. y 1,2,5")
    print("  o item intent override primary intent, e.g. o 3 7")
    print("  s list        skip items, e.g. s 4,6")
    print("  menu          show intent numbers")
    print("  n             save handled items and move to next batch")
    print("  q             save and quit")
    print(f"Visible item numbers are 1-{batch_size}. Every item needs y, o, s, n, or q.")


def _parse_item_numbers(raw: str, max_item: int) -> list[int]:
    items = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= max_item:
            items.append(int(part))
    return items


def _prompt_override_details(input_fn: Callable[[str], str]) -> tuple[str, str, str]:
    secondary = _parse_secondary_numbers(
        input_fn("Secondary intent number(s), comma-separated, or Enter for none: ").strip()
    )
    is_ambiguous = "True" if input_fn("Ambiguous / underspecified? [y/N]: ").strip().lower() == "y" else "False"
    notes = input_fn("Human notes (optional): ").strip()
    return secondary, is_ambiguous, notes


def review_pending_batch(
    df: pd.DataFrame,
    csv_path: Path,
    batch_size: int = 25,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Review pending candidates in high-risk batches without auto-approval."""
    batch_size = max(1, min(batch_size, 30))
    deferred_this_session = set()
    batch_number = 0

    while True:
        indices = [idx for idx in ranked_pending_indices(df) if idx not in deferred_this_session]
        if not indices:
            if pending_mask(df).sum() > 0:
                print("\nAll remaining pending candidates were skipped or deferred in this session.")
                return
            print("\nNo pending candidates remain. Do not freeze unless Phase 3 criteria are complete.")
            return

        batch = indices[:batch_size]
        batch_number += 1
        print_intent_menu()
        for position, idx in enumerate(batch, 1):
            print_batch_candidate(df.loc[idx], batch_number=batch_number, position=position, total=len(batch))
        print_batch_commands(len(batch))

        handled = set()
        while len(handled) < len(batch):
            raw = input_fn("\nBatch command: ").strip().lower()
            if not raw:
                continue

            if raw == "q":
                save_progress(df, csv_path)
                print("Exiting batch annotation session.")
                return

            if raw == "menu":
                print_intent_menu()
                continue

            if raw == "n":
                for item, idx in enumerate(batch, 1):
                    if item not in handled:
                        deferred_this_session.add(idx)
                save_progress(df, csv_path)
                break

            command, _, rest = raw.partition(" ")
            if command == "y":
                items = _parse_item_numbers(rest, len(batch))
                if not items:
                    print("Enter item numbers to approve, for example: y 1,2,5")
                    continue
                for item in items:
                    if item in handled:
                        print(f"Item {item} was already handled in this batch.")
                        continue
                    apply_human_approved_suggestion(df, batch[item - 1])
                    handled.add(item)
                    print(f"Approved item {item}: {df.at[batch[item - 1], 'human_primary_intent']}")
                save_progress(df, csv_path)
                continue

            if command == "s":
                items = _parse_item_numbers(rest, len(batch))
                if not items:
                    print("Enter item numbers to skip, for example: s 4,6")
                    continue
                for item in items:
                    if item in handled:
                        print(f"Item {item} was already handled in this batch.")
                        continue
                    deferred_this_session.add(batch[item - 1])
                    handled.add(item)
                    print(f"Skipped item {item}. It remains pending.")
                continue

            if command == "o":
                parts = rest.split()
                if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                    print("Use: o <item-number> <intent-number>, for example: o 3 7")
                    continue
                item = int(parts[0])
                intent_number = int(parts[1])
                if not (1 <= item <= len(batch)) or not (1 <= intent_number <= len(INTENTS_ORDERED)):
                    print("Item or intent number out of range.")
                    continue
                if item in handled:
                    print(f"Item {item} was already handled in this batch.")
                    continue
                secondary, is_ambiguous, notes = _prompt_override_details(input_fn)
                primary = INTENTS_ORDERED[intent_number - 1]
                apply_human_override(df, batch[item - 1], primary, secondary, is_ambiguous, notes)
                handled.add(item)
                save_progress(df, csv_path)
                print(f"Override recorded for item {item}: {primary}")
                continue

            print("Unknown command. Use y, o, s, menu, n, or q.")


def review_pending(
    df: pd.DataFrame,
    csv_path: Path,
    input_fn: Callable[[str], str] = input,
) -> None:
    indices = df[pending_mask(df)].index.tolist()
    total_pending = len(indices)
    if not indices:
        print("\nNo pending candidates remain. Do not freeze unless Phase 3 criteria are complete.")
        return

    print("\nCommands:")
    print("  y      accept suggestion")
    print("  1-10   override primary intent")
    print("  s      skip")
    print("  q      save and quit")
    print_intent_menu()

    for offset, idx in enumerate(indices, 1):
        if not clean_cell(df.at[idx, "suggested_primary_intent"]):
            suggestion = compute_suggestion(
                df.at[idx, "customer_message"],
                df.at[idx, "context_before_customer"],
                df.at[idx, "brand_response"],
            )
            for col, value in suggestion.items():
                df.at[idx, col] = value

        print_candidate(df.loc[idx], offset, total_pending)
        print_suggestion(df.loc[idx])

        while True:
            choice = input_fn("\n[y] Accept  [1-10] Override primary  [s] Skip  [q] Save+quit: ").strip().lower()

            if choice == "q":
                save_progress(df, csv_path)
                print("Exiting annotation session.")
                return

            if choice == "s":
                print("Skipped. No human annotation fields changed.")
                break

            if choice == "y":
                apply_human_approved_suggestion(df, idx)
                save_progress(df, csv_path)
                print(f"Human-approved suggestion recorded: {df.at[idx, 'human_primary_intent']}")
                break

            if choice.isdigit() and 1 <= int(choice) <= len(INTENTS_ORDERED):
                primary = INTENTS_ORDERED[int(choice) - 1]
                secondary = _parse_secondary_numbers(
                    input_fn("Secondary intent number(s), comma-separated, or Enter for none: ").strip()
                )
                amb = "True" if input_fn("Ambiguous / underspecified? [y/N]: ").strip().lower() == "y" else "False"
                notes = input_fn("Human notes (optional): ").strip()
                apply_human_override(df, idx, primary, secondary, amb, notes)
                save_progress(df, csv_path)
                print(f"Human override recorded: {primary}")
                break

            print("Invalid input. Enter y, a number from 1-10, s, or q.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assisted Phase 3 golden candidate annotation.")
    parser.add_argument("--csv", type=Path, default=CANDIDATES_CSV, help="Candidate CSV path.")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Review pending candidates in high-risk batches instead of one at a time.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of pending candidates to display per batch, capped at 30.",
    )
    parser.add_argument(
        "--suggest-only",
        action="store_true",
        help="Populate missing suggested_* fields for pending rows, without reviewing or writing human_* fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()

    validate_reference_docs()
    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate CSV does not exist: {csv_path}")

    df = load_candidates(csv_path)
    print_banner(csv_path)
    print(f"Total candidates: {len(df)}")
    print(f"Already reviewed: {df['annotation_status'].fillna('').astype(str).str.strip().eq('reviewed').sum()}")
    print(f"Pending review:   {pending_mask(df).sum()}")

    added = ensure_suggestions_for_pending(df)
    if args.suggest_only:
        save_candidates(df, csv_path)
        print(f"Suggestion-only pass complete. Added suggestions for {added} pending candidate(s).")
        print("No human_* labels were written and no pending candidate was auto-approved.")
        return

    if added:
        save_candidates(df, csv_path)
        print(f"Prepared suggestions for {added} pending candidate(s).")

    if args.batch:
        review_pending_batch(df, csv_path, batch_size=args.batch_size)
    else:
        review_pending(df, csv_path)


if __name__ == "__main__":
    main()
