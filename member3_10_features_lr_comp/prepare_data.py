import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
a = ap.parse_args()
os.makedirs("data", exist_ok=True)
rd = lambda ch, s: pd.read_csv(os.path.join(a.src, ch, f"{s}.csv"))
for s in ("train", "test"):
    A, C = rd("A", s), rd("C", s)
    p = os.path.join(a.src, "B", f"{s}.csv")
    if os.path.exists(p):
        B = pd.read_csv(p)
    else:
        B = pd.concat([A[A.label == 0], C[C.label == 1]]).sample(frac=1, random_state=0)
        print(f"B/{s}.csv not found -> rebuilt ({len(B)} rows)")
    for ch, d in (("A", A), ("B", B), ("C", C)):
        d.to_csv(f"data/{ch}_{s}.csv", index=False)
