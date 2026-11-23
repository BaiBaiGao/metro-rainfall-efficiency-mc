"""Export the Efficiency sheet of every downloaded workbook as gzip CSV (small enough to commit)."""
import glob
import os
import sys

import pandas as pd

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for path in glob.glob(os.path.join(root, "**", "MonteCarlo_Results_*.xlsx"), recursive=True):
    job = os.path.basename(os.path.dirname(path))
    name = os.path.splitext(os.path.basename(path))[0]
    pd.read_excel(path, sheet_name="Efficiency").to_csv(
        os.path.join(out, f"{job}__{name}.csv.gz"), index=False, float_format="%.17g")
print(len(os.listdir(out)), "efficiency tables")
