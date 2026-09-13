"""Phase 4 interactive human calibration annotator (reply quality).

Scores the GENERATED REPLY (never the golden reference) against the retrieved
historical evidence on the deterministic 50-example calibration subset
(100 rows = 50 examples x 2 baselines).

For each row the annotator sees:
  customer message, conversation context, generated reply (with baseline +
  predicted intent), retrieved historical evidence, golden reference (context only).

Scores per row:
  relevance, helpfulness, groundedness, appropriateness: 0=poor..3=strong
  unsupported_claims (INVERTED): 0=no meaningful unsupported claim (best),
      1=minor, 2=significant, 3=severe/fabricated (worst)
  overall_score: 0-3 holistic
  optional human_note

Workflow guarantees:
  - Loads data/judge/human_calibration.csv (created by build_judge_inputs.py).
  - Saves progress after every annotation; skips already-reviewed rows.
  - Never overwrites completed annotations accidentally ('q' saves + quits,
    completed rows are only re-edited via explicit 'edit' command).
  - Resume-safe: re-running continues at the first pending row.
  - Default mode is 100% manual: no model output is ever shown or stored.
  - Optional `--suggest` mode calls the LLM judge ONCE per row and shows the
    result ONLY as "AI SUGGESTION — NOT HUMAN LABEL". A suggestion is stored
    as a human label ONLY if the human explicitly confirms (Enter) or edits
    (0-3) that exact field. The `ai_suggestion_shown` column records which rows
    displayed a suggestion, so any anchoring effect on agreement is auditable.
  - Optional `--fast` mode shows a single AI suggestion for all six scores and
    asks for a single bulk approval [y/n/e/q]. The suggestion NEVER becomes a
    human label without explicit human confirmation. Metadata columns track
    the provenance of every reviewed row.
  - `--status` prints validation counts without requiring interaction.

Usage:
    python scripts/annotate_judge_calibration.py [--status] [--suggest] [--fast]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HUMAN_CALIBRATION_CSV = ROOT / "data" / "judge" / "human_calibration.csv"
SUGGESTION_CACHE_JSONL = ROOT / "data" / "judge" / "suggestion_cache.jsonl"

SCORE_COLS = [
    ("relevance_human", "RELEVANCE (0=poor,1=weak,2=acceptable,3=strong)"),
    ("helpfulness_human", "HELPFULNESS (0=poor,1=weak,2=acceptable,3=strong)"),
    ("groundedness_human", "GROUNDEDNESS (0=poor,1=weak,2=acceptable,3=strong)"),
    ("appropriateness_human", "APPROPRIATENESS (0=poor,1=weak,2=acceptable,3=strong)"),
    ("unsupported_claims_human",
     "UNSUPPORTED_CLAIMS INVERTED (0=none/best,1=minor,2=significant,3=severe/fabricated)"),
    ("overall_score_human", "OVERALL (0=poor,1=weak,2=acceptable,3=strong)"),
]


# score column -> LLM judge field (for optional --suggest display only)
SUGGEST_MAP = {
    "relevance_human": "relevance",
    "helpfulness_human": "helpfulness",
    "groundedness_human": "groundedness",
    "appropriateness_human": "appropriateness",
    "unsupported_claims_human": "unsupported_claims",
    "overall_score_human": "overall_score",
}


def _load_suggestion_cache() -> dict:
    """Load cached AI suggestions keyed by 'baseline::pair_id'."""
    cache = {}
    if SUGGESTION_CACHE_JSONL.exists():
        with open(SUGGESTION_CACHE_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = f"{obj['baseline']}::{obj['pair_id']}"
                    cache[key] = obj
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def _save_suggestion_cache(key: str, suggestion: dict) -> None:
    """Append a suggestion to the cache (never overwrites)."""
    SUGGESTION_CACHE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(SUGGESTION_CACHE_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(suggestion, ensure_ascii=False) + "\n")


def _is_reviewed(row) -> bool:
    if str(row.get("annotation_status", "")).strip().lower() != "reviewed":
        return False
    for col, _ in SCORE_COLS:
        v = str(row.get(col, "")).strip()
        if v == "" or v.lower() == "nan":
            return False
        try:
            iv = int(float(v))
        except ValueError:
            return False
        if iv < 0 or iv > 3:
            return False
    return True


def _ask_score(label: str, current: str = "", suggestion: str = "") -> str:
    while True:
        if suggestion != "":
            prompt = (f"  {label} [AI SUGGESTION = {suggestion} — NOT A LABEL] "
                      f"Enter to CONFIRM, or type 0-3 to override, 's'=skip, 'q'=save+quit: ")
        else:
            prompt = (f"  {label} [0-3]{f' (current={current})' if current else ''}, "
                      f"'s'=skip row, 'q'=save+quit: ")
        raw = input(prompt).strip().lower()
        if raw in ("q", "s"):
            return raw
        if raw == "" and (current != "" or suggestion != ""):
            return suggestion if suggestion != "" else current
        if raw in ("0", "1", "2", "3"):
            return raw
        print("    Invalid. Enter 0, 1, 2, 3, 's', or 'q'.")


def _compact(text: str, limit: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[:limit] + "..."


def _display_fast_row(row, idx: int, total: int, done_count: int) -> None:
    """Concise fast-mode row display: suggestion-relevant facts only.

    The human MUST see customer message, generated reply, and retrieved evidence
    before confirming. Context shown only when non-empty (truncated). The golden
    reference is intentionally omitted in fast mode (it is never scored).
    """
    pct = 100.0 * done_count / total if total else 0.0
    print("\n" + "=" * 75)
    print(f"ROW {idx + 1}/{total} ({pct:.0f}% done) | pair={row['pair_id']} | "
          f"{row['baseline']} | pred={row.get('predicted_intent', '')} "
          f"| true={row.get('human_primary_intent', '')}")
    print(f"CUSTOMER: {_compact(row.get('customer_message', ''), 300)}")
    ctx = _compact(row.get("context_before_customer", ""), 200)
    if ctx and ctx.lower() not in ("nan", "none", "(none)"):
        print(f"CONTEXT: {ctx}")
    print(f"REPLY (score this): {_compact(row.get('generated_reply', ''), 400)}")
    print(f"EVIDENCE [{row.get('retrieval_source_pair_id', '')}]: "
          f"{_compact(row.get('retrieved_historical_response', ''), 400)}")


def _ask_fast_approval(suggestion: dict) -> str:
    """Display AI suggestion and ask for bulk approval y/n/e/q.

    Returns 'y' (confirm all), 'n' (reject all, go manual), 'e' (edit individual),
    'q' (quit).
    """
    scores = [
        suggestion.get("relevance"),
        suggestion.get("helpfulness"),
        suggestion.get("groundedness"),
        suggestion.get("appropriateness"),
        suggestion.get("unsupported_claims"),
        suggestion.get("overall_score"),
    ]
    reason = suggestion.get("reason", "")
    print("-" * 75)
    print("AI SUGGESTION — NOT HUMAN LABEL")
    print("Suggested scores:")
    print(f"  1. Relevance:           {scores[0]}")
    print(f"  2. Helpfulness:         {scores[1]}")
    print(f"  3. Groundedness:        {scores[2]}")
    print(f"  4. Appropriateness:     {scores[3]}")
    print(f"  5. Unsupported claims:  {scores[4]} (INVERTED: 0=best)")
    print(f"  6. Overall:             {scores[5]}")
    if reason:
        print(f"Rationale: {reason[:200]}")
    print("-" * 75)
    while True:
        raw = input("Approve these scores? [y/n/e/q]: ").strip().lower()
        if raw in ("y", "n", "e", "q"):
            return raw
        print("    Enter y (approve all), n (reject, go manual), e (edit some), or q (quit).")


def _edit_individual_scores(suggestion: dict) -> dict:
    """Allow human to edit individual scores from the suggestion."""
    print("Edit individual scores (Enter keeps AI suggestion, type 0-3 to override):")
    final = {}
    for col, label in SCORE_COLS:
        key = SUGGEST_MAP[col]
        sugg_val = suggestion.get(key, "")
        while True:
            raw = input(f"  {label} [AI={sugg_val}] → Enter to keep, 0-3 to change, q to quit: ").strip().lower()
            if raw == "q":
                return {"q": True}
            if raw == "" and sugg_val != "":
                final[key] = str(sugg_val)
                break
            if raw in ("0", "1", "2", "3"):
                final[key] = raw
                break
            print("    Invalid. Enter 0, 1, 2, 3, or just Enter to keep AI suggestion.")
    return final


def _fetch_suggestion(row) -> dict:
    """Call the LLM judge for a display-only suggestion. Raises on API problems."""
    from src.judge import build_judge_prompt, call_llm_judge, get_judge_config
    get_judge_config()  # fail-closed without OPENAI_API_KEY
    prompt = build_judge_prompt(
        customer_message=str(row.get("customer_message", "")),
        context_before_customer=str(row.get("context_before_customer", "") or ""),
        generated_reply=str(row.get("generated_reply", "")),
        retrieved_evidence=str(row.get("retrieved_historical_response", "")),
        reference_response=str(row.get("reference_response", "") or ""),
        predicted_intent=str(row.get("predicted_intent", "") or ""),
    )
    return call_llm_judge(prompt)


def _get_cached_or_fetch_suggestion(row, cache: dict) -> dict:
    """Get cached suggestion or fetch new one, save to cache."""
    key = f"{row['baseline']}::{row['pair_id']}"
    if key in cache:
        return cache[key]
    suggestion = _fetch_suggestion(row)
    suggestion_with_meta = {
        "pair_id": str(row["pair_id"]),
        "baseline": str(row["baseline"]),
        "suggestion": suggestion,
    }
    _save_suggestion_cache(key, suggestion_with_meta)
    cache[key] = suggestion_with_meta
    return suggestion


def _print_status(df) -> None:
    total = len(df)
    reviewed = int(sum(1 for _, r in df.iterrows() if _is_reviewed(r)))
    print(f"HUMAN_REVIEWED = {reviewed} / {total}")
    print(f"HUMAN_PENDING = {total - reviewed}")
    for col, _ in SCORE_COLS:
        filled = int(((df[col].astype(str).str.strip() != "")
                       & (df["annotation_status"].astype(str).str.lower() == "reviewed")).sum())
        print(f"  {col}: {filled} reviewed-valid")
    if "ai_suggestion_shown" in df.columns:
        n_sugg = int((df["ai_suggestion_shown"].astype(str) == "1").sum())
        print(f"  rows shown an AI suggestion (audited, NOT labels): {n_sugg}")
    dupes = int(df.duplicated(subset=["pair_id", "baseline"]).sum())
    print(f"  duplicate (pair_id, baseline) rows: {dupes}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Phase 4 human reply-quality calibration.")
    ap.add_argument("--status", action="store_true",
                    help="Print validation counts and exit (no interaction).")
    ap.add_argument("--suggest", action="store_true",
                    help="Show per-row LLM suggestions labeled 'AI SUGGESTION - NOT HUMAN LABEL'. "
                         "Requires OPENAI_API_KEY. Suggestions are stored ONLY on explicit "
                         "per-field human confirmation.")
    ap.add_argument("--fast", action="store_true",
                    help="Fast mode: show single AI suggestion for all six scores, "
                         "ask for bulk approval [y/n/e/q]. AI suggestion NEVER becomes "
                         "a human label without explicit confirmation. Requires OPENAI_API_KEY.")
    args = ap.parse_args()

    if args.fast and args.suggest:
        print("ERROR: --fast and --suggest are mutually exclusive. Choose one.")
        return

    print("=" * 75)
    print("PHASE 4 — HUMAN REPLY-QUALITY CALIBRATION (50 examples x 2 baselines)")
    print("You are scoring the GENERATED REPLY against the retrieved evidence.")
    print("Unsupported_claims is INVERTED: 0 = clean/best, 3 = severe fabrication.")
    if args.suggest:
        print("SUGGEST MODE: AI SUGGESTIONs are NOT human labels until YOU confirm each field.")
    if args.fast:
        print("FAST MODE: Single AI suggestion per row, approve with [y] or edit with [e].")
        print("Rubric: R/H/G/A 0=poor..3=strong; U INVERTED 0=clean..3=fabricated; O=overall.")
    print("=" * 75)
    if not HUMAN_CALIBRATION_CSV.exists():
        print(f"Missing {HUMAN_CALIBRATION_CSV}. Run scripts/build_judge_inputs.py first.")
        return
    df = pd.read_csv(HUMAN_CALIBRATION_CSV, dtype="object").fillna("")
    # Ensure all metadata columns exist
    for col in ("ai_suggestion_shown", "ai_suggestion_values", "human_confirmed",
                "human_edited", "annotation_mode"):
        if col not in df.columns:
            df[col] = ""
    if args.suggest or args.fast:
        try:
            from src.judge import get_judge_config
            get_judge_config()
        except RuntimeError as e:
            print(f"STOP: --suggest/--fast needs an LLM API key: {e}")
            print("Re-run without --suggest/--fast for fully manual annotation.")
            return
    total = len(df)
    reviewed = int(sum(1 for _, r in df.iterrows() if _is_reviewed(r)))
    print(f"Rows: {total} | Reviewed: {reviewed} | Pending: {total - reviewed}")
    if args.status:
        _print_status(df)
        return

    suggestion_cache = _load_suggestion_cache()
    mode_desc = ("Commands per prompt: 0-3 score | Enter keeps current | s skip | q save+quit"
                 if not args.fast else "Approve AI suggestion: y=confirm all, n=reject+manual, e=edit some, q=quit")
    print(mode_desc)
    print("-" * 75)

    for idx in df.index.tolist():
        row = df.loc[idx]
        if _is_reviewed(row):
            continue
        done_count = int(sum(1 for _, r in df.iterrows() if _is_reviewed(r)))
        if args.fast:
            _display_fast_row(row, idx, total, done_count)
        else:
            pct = 100.0 * done_count / total if total else 0.0
            print("\n" + "=" * 75)
            print(f"ROW {idx + 1}/{total} ({pct:.0f}% done) | pair={row['pair_id']} | baseline={row['baseline']} "
                  f"| predicted_intent={row.get('predicted_intent', '')} "
                  f"| human_intent={row.get('human_primary_intent', '')}")
            print("-" * 75)
            ctx = str(row.get("context_before_customer", "")).strip()
            print(f"CONTEXT:\n  {(ctx if ctx else '(none)')}")
            print("-" * 75)
            print(f"CUSTOMER:\n  {row.get('customer_message', '')}")
            print("-" * 75)
            print(f"GENERATED REPLY [{row['baseline']}] (SCORE THIS):\n  {row.get('generated_reply', '')}")
            print("-" * 75)
            print(f"RETRIEVED EVIDENCE (source={row.get('retrieval_source_pair_id', '')}):\n"
                  f"  {row.get('retrieved_historical_response', '')}")
            print("-" * 75)
            ref = str(row.get("reference_response", "")).strip()
            if ref:
                print(f"GOLDEN REFERENCE (context only, do NOT score this):\n  {ref[:800]}")
                print("-" * 75)

        new_vals = {}
        suggestion = {}
        suggestion_shown = False
        human_confirmed = False
        human_edited = False
        annotation_mode = "manual"
        # Always bound: "" means "no fast decision yet". Any exception path
        # below must leave these safe so line 438 can never raise UnboundLocalError.
        approval = ""
        fast_manual = False

        if args.fast:
            try:
                cached = _get_cached_or_fetch_suggestion(row, suggestion_cache)
                suggestion = cached["suggestion"]
                suggestion_shown = True
                annotation_mode = "ai_suggestion_human_confirmed"
                approval = _ask_fast_approval(suggestion)
                if approval == "q":
                    df.to_csv(HUMAN_CALIBRATION_CSV, index=False, encoding="utf-8")
                    print(f"\n[Saved] progress written to {HUMAN_CALIBRATION_CSV}. Goodbye!")
                    return
                if approval == "y":
                    # Human explicitly confirms all AI suggestions
                    for col, _ in SCORE_COLS:
                        key = SUGGEST_MAP[col]
                        new_vals[col] = str(suggestion.get(key, ""))
                    human_confirmed = True
                elif approval == "e":
                    # Human edits some fields
                    edit_result = _edit_individual_scores(suggestion)
                    if edit_result.get("q"):
                        df.to_csv(HUMAN_CALIBRATION_CSV, index=False, encoding="utf-8")
                        print(f"\n[Saved] progress written to {HUMAN_CALIBRATION_CSV}. Goodbye!")
                        return
                    for col, _ in SCORE_COLS:
                        key = SUGGEST_MAP[col]
                        if key in edit_result:
                            new_vals[col] = edit_result[key]
                            if edit_result[key] != str(suggestion.get(key, "")):
                                human_edited = True
                            else:
                                human_confirmed = True
                    annotation_mode = "ai_suggestion_human_edited"
                elif approval == "n":
                    # Reject suggestion: fall through to manual entry below.
                    annotation_mode = "manual"
                    suggestion_shown = False
                    suggestion = {}
                    fast_manual = True
                else:
                    # Should not happen
                    suggestion_shown = False
                    suggestion = {}
            except Exception as e:
                print(f"  [AI suggestion unavailable: {e} — falling back to manual]")
                suggestion_shown = False
                suggestion = {}
                annotation_mode = "manual"
                fast_manual = True

        if not args.fast and args.suggest:
            try:
                suggestion = _fetch_suggestion(row)
                suggestion_shown = True
                print("-" * 75)
                print("AI SUGGESTION — NOT HUMAN LABEL (confirm or override each field): "
                      + ", ".join(f"{SUGGEST_MAP[c]}={suggestion[SUGGEST_MAP[c]]}"
                                  for c, _ in SCORE_COLS)
                      + f" | judge reason: {suggestion.get('reason', '')[:160]}")
            except Exception as e:
                print(f"  [suggestion unavailable: {e} — continuing fully manual]")
                suggestion = {}
                suggestion_shown = False

        # Manual scoring runs in manual/--suggest modes, and in --fast mode only
        # when the human rejected the suggestion (n) or the API failed (fallback).
        # After y/e, new_vals is already filled so this block is skipped.
        if not new_vals and (not args.fast or fast_manual):
            if fast_manual:
                print("  Manual scoring (type 0-3 per dimension):")
            for col, label in SCORE_COLS:
                cur = str(row.get(col, "")).strip()
                sugg = str(suggestion.get(SUGGEST_MAP[col], "")) if suggestion else ""
                ans = _ask_score(label, current=cur, suggestion=sugg)
                if ans == "q":
                    df.to_csv(HUMAN_CALIBRATION_CSV, index=False, encoding="utf-8")
                    print(f"\n[Saved] progress written to {HUMAN_CALIBRATION_CSV}. Goodbye!")
                    return
                if ans == "s":
                    print("  Skipped (left pending).")
                    new_vals = {}
                    break
                new_vals[col] = ans
            if not new_vals:
                continue

        if new_vals or (args.fast and approval in ("y", "e")):
            note = input("  Optional human note (Enter to skip, 'q' to save+quit): ")
            if note.strip().lower() == "q":
                for col, val in new_vals.items():
                    df.at[idx, col] = val
                df.at[idx, "annotation_status"] = "reviewed"
                df.at[idx, "ai_suggestion_shown"] = "1" if suggestion_shown else "0"
                if suggestion_shown:
                    df.at[idx, "ai_suggestion_values"] = json.dumps({
                        k: suggestion.get(SUGGEST_MAP[k], "") for k, _ in SCORE_COLS
                    })
                    df.at[idx, "human_confirmed"] = "1" if human_confirmed else "0"
                    df.at[idx, "human_edited"] = "1" if human_edited else "0"
                    df.at[idx, "annotation_mode"] = annotation_mode
                df.to_csv(HUMAN_CALIBRATION_CSV, index=False, encoding="utf-8")
                print(f"\n[Saved] progress written to {HUMAN_CALIBRATION_CSV}. Goodbye!")
                return
            for col, val in new_vals.items():
                df.at[idx, col] = val
            df.at[idx, "human_note"] = note.strip()
            df.at[idx, "annotation_status"] = "reviewed"
            df.at[idx, "ai_suggestion_shown"] = "1" if suggestion_shown else "0"
            if suggestion_shown:
                df.at[idx, "ai_suggestion_values"] = json.dumps({
                    k: suggestion.get(SUGGEST_MAP[k], "") for k, _ in SCORE_COLS
                })
                df.at[idx, "human_confirmed"] = "1" if human_confirmed else "0"
                df.at[idx, "human_edited"] = "1" if human_edited else "0"
                df.at[idx, "annotation_mode"] = annotation_mode
            df.to_csv(HUMAN_CALIBRATION_CSV, index=False, encoding="utf-8")
            reviewed_now = int(sum(1 for _, r in df.iterrows() if _is_reviewed(r)))
            print(f"  -> Recorded. Reviewed {reviewed_now}/{total}.")


if __name__ == "__main__":
    main()
