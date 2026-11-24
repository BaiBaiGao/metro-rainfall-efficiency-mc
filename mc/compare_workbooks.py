"""Compare every sheet of a new MonteCarlo workbook with a reference workbook (same cell, Run_1..Run_k).

usage: compare_workbooks.py NEW_DIR REF_DIR PATTERN RUNS TOL
PATTERN is matched in both directories, e.g. 'MonteCarlo_Results_P100_slow5_stop10_*.xlsx'.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

new_dir, ref_dir, pattern, runs, tol = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])


def one(directory):
    files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    if len(files) != 1:
        sys.exit(f"expected one workbook for {pattern} in {directory}, found {files}")
    return files[0]


new_path, ref_path = one(new_dir), one(ref_dir)
new, ref = pd.read_excel(new_path, sheet_name=None), pd.read_excel(ref_path, sheet_name=None)
cols = [f"Run_{i}" for i in range(1, runs + 1)]
report = {"new": os.path.relpath(new_path, new_dir), "reference": os.path.relpath(ref_path, ref_dir), "runs": runs}
for sheet in ref:
    a = ref[sheet][cols].to_numpy(float)
    b = new[sheet][cols].to_numpy(float)
    report[sheet] = float(np.nanmax(np.abs(a - b)))
worst = max(v for k, v in report.items() if k not in ("new", "reference", "runs"))
report["tolerance"] = tol
report["passed"] = bool(worst <= tol)
print(json.dumps(report, ensure_ascii=False, indent=2))
with open(os.path.join(new_dir, "regression_v2_report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
sys.exit(0 if report["passed"] else 1)
