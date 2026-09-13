"""Programmatic leakage verification script.

Verifies:
- Total pairs count
- Golden set counts (reviewed/pending)
- All 322 golden IDs present in spotify_pairs.parquet
- Overlap after exclusion == 0
- Training/retrieval pool size == 42,770

Run: .venv/Scripts/python.exe scripts/verify_leakage.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

pairs = pd.read_parquet(ROOT / "data" / "processed" / "spotify_pairs.parquet")
golden = pd.read_csv(ROOT / "data" / "golden" / "golden_candidates.csv")

golden_ids = set(golden["pair_id"].tolist())
pool = pairs[~pairs["pair_id"].isin(golden_ids)]
golden_in_pairs = len(pairs[pairs["pair_id"].isin(golden_ids)])
unmatched = len(golden_ids) - golden_in_pairs
overlap_after = len(pool[pool["pair_id"].isin(golden_ids)])

reviewed = golden[golden["human_primary_intent"].notna()]
pending = golden[golden["human_primary_intent"].isna()]

print(f"Total pairs:          {len(pairs)}")
print(f"Golden total:         {len(golden)}")
print(f"Golden in pairs:      {golden_in_pairs}")
print(f"Unmatched golden:     {unmatched}")
print(f"Leakage-free pool:    {len(pool)}")
print(f"Overlap after excl:   {overlap_after}")
print(f"Reviewed:             {len(reviewed)}")
print(f"Pending:              {len(pending)}")

# Intent distribution in reviewed set
print("\nIntent distribution (reviewed 150):")
for intent, count in reviewed["human_primary_intent"].value_counts().items():
    print(f"  {intent}: {count}")

# Assertions
assert len(pairs) == 43092, f"Expected 43092 pairs, got {len(pairs)}"
assert len(golden) == 322, f"Expected 322 golden, got {len(golden)}"
assert golden["pair_id"].duplicated().sum() == 0, "Duplicate pair_ids in golden set!"
assert golden_in_pairs == 322, f"Expected all 322 golden IDs in pairs, got {golden_in_pairs}"
assert unmatched == 0, f"Expected 0 unmatched golden IDs, got {unmatched}"
assert len(pool) == 42770, f"Expected 42770 in pool, got {len(pool)}"
assert overlap_after == 0, f"Expected 0 overlap after exclusion, got {overlap_after}"
assert len(reviewed) == 150, f"Expected 150 reviewed, got {len(reviewed)}"
assert len(pending) == 172, f"Expected 172 pending, got {len(pending)}"
assert len(reviewed["human_primary_intent"].unique()) == 10, "Not all 10 intents represented!"

print("\nALL LEAKAGE ASSERTIONS PASSED.")
