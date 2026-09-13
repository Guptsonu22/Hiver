# Machine Calibration — Diagnostic Path (NOT Human Calibration)

## What it is
A time-saving diagnostic artifact for when genuine human calibration (0/100) and/or
the LLM API (HTTP 401) are unavailable: `data/judge/machine_calibration_100.csv`
(100 rows = 50 examples x 2 baselines) plus `results/machine_metrics.json`
(full-300 diagnostic means per baseline).

## What it is NOT
- NOT human labels. No `human_*` columns, no `human_confirmed`, no agreement.
- NOT LLM judge scores (unless a row's `source` says `llm:<model>`, meaning the real
  API produced it — still machine-generated, still not human).
- NOT a replacement for the assignment's human-label requirement.
- Human-vs-LLM agreement CANNOT be computed from it and never will be.

## Method
Per row, the generator (`scripts/generate_machine_calibration.py`) first tries the
real LLM judge (existing rubric logic, cached in `data/judge/machine_llm_cache.jsonl`
with no key material). On ANY failure it falls back to `deterministic_heuristic_v1`
(`src/machine_score.py`): crude, fully documented word-overlap/length/keyword bins on
the 0-3 rubric scales. Every row records `method`, `source`, `machine_reason`
(which bin rule fired), and `generated_utc`.

## Guarantees (tested in `tests/test_machine_calibration.py`)
- `human_calibration.csv` is hashed before/after generation and must be byte-identical.
- Machine pair_ids are a subset of the reviewed 150, disjoint from the pending 172.
- Retrieval evidence is disjoint from all 322 golden IDs (same leakage assertion).
- Official `run_judge.py` (no flags) still STOPs without genuine 100/100.
- `--machine-calibration` writes ONLY `results/machine_metrics.json`; it cannot
  create `judge_metrics.json` / `judge_outputs.jsonl`.

## Commands
```powershell
.\.venv\Scripts\python.exe scripts/generate_machine_calibration.py
.\.venv\Scripts\python.exe scripts/run_judge.py --machine-calibration
```

## Current state
100/100 machine rows, all `deterministic_heuristic_v1` (API unavailable at generation
time). Diagnostic means: both baselines overall ~2.0, groundedness 3.0 (extractive
replies match evidence by construction), relevance low (terse Twitter text) — crude
heuristic behavior, reported as such, not as quality findings.
