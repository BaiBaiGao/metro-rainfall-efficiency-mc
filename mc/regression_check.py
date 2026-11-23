"""Compare a legacy-mode cloud run (P100, slow5/stop10, Run_1-4) with the archived Windows workbook."""
import glob
import json
import sys

import numpy as np
import pandas as pd

out_dir, ref_csv = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
files = glob.glob(f"{out_dir}/MonteCarlo_Results_P100_slow5_stop10_*.xlsx")
if len(files) != 1:
    sys.exit(f"expected one workbook, found {files}")
new = pd.read_excel(files[0], sheet_name=None)
ref = pd.read_csv(ref_csv)
report = {}
for sheet, frame in ref.groupby("sheet"):
    a = frame[["Run_1", "Run_2", "Run_3", "Run_4"]].to_numpy(float)
    b = new[sheet][["Run_1", "Run_2", "Run_3", "Run_4"]].to_numpy(float)
    report[sheet] = float(np.nanmax(np.abs(a - b)))
worst = max(report.values())
report["tolerance"] = tol
report["passed"] = bool(worst <= tol)
print(json.dumps(report, indent=2))
with open(f"{out_dir}/regression_report.json", "w") as handle:
    json.dump(report, handle, indent=2)
sys.exit(0 if report["passed"] else 1)
