# Decision Log — Redirect Notice

The engineering decision log has been consolidated into a single document.

**All 13 decisions are in [`DECISIONS.md`](../DECISIONS.md) (repo root).**

Decisions cover:
- D1: Brand selection (SpotifyCares vs alternatives)
- D2: Ten-intent taxonomy design
- D3: `pair_id` leakage gate
- D4: All 322 golden IDs excluded (not just 150 reviewed)
- D5: Language strategy (retain all, evaluate English-only)
- D6: TF-IDF kNN agent vs keyword baseline separation
- D7: Extractive reply as default (zero hallucination by construction)
- D8: Escalation as policy output (no "escalation accuracy")
- D9: Weak training labels explicitly distinguished from ground truth
- D10: 0–3 rubric with inverted `unsupported_claims`; 50-example calibration
- D11: Fail-closed judge (no fallback heuristic)
- D12: Resume-safe annotator (no auto-confirm)
- D13: Vectorized `lexsort` retrieval
