"""Check that build_frame_incomplete reproduces an existing complete origin table exactly (python verify_builder.py PROJ_ROOT 2014-05-18)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import origin_builder as ob  # noqa: E402

root, origin = Path(sys.argv[1]).resolve(), pd.Timestamp(sys.argv[2])
ep = ob._load_pipeline(root)
frames = ep.load_raw(ep.Paths(root))
mine = ob.build_frame_incomplete(ep, frames["train"], frames["train"], frames["metadata"], origin)
ref = pd.read_parquet(root / f"data/processed/training_origins/origin_{origin.date()}.parquet")
print("shape", mine.shape, ref.shape)
assert list(mine.columns) == list(ref.columns), "column mismatch"
bad = []
for c in ref.columns:
    a, b = mine[c], ref[c]
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        ok = np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=1e-9, atol=1e-9)
    else:
        ok = a.fillna('').astype(str).equals(b.fillna('').astype(str))  # NaN vs None only differ after a parquet round trip
    if not ok:
        bad.append(c)
print("MISMATCHED COLUMNS:", bad)
raise SystemExit(1 if bad else 0)
