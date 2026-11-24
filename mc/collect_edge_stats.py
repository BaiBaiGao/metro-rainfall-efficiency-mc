"""Collect <job>/edge_stats/*.json (one per run) into <job>/edge_stats.csv (the artifact keeps only the CSV)."""
import glob
import json
import os
import sys

import pandas as pd

job = sys.argv[1]
files = sorted(glob.glob(os.path.join(job, "edge_stats", "*.json")))
if not files:
    print("no edge stats")
    sys.exit(0)
rows = [json.load(open(path, encoding="utf-8")) for path in files]
meta_path = os.path.join(job, "job_meta.json")
meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
frame = pd.DataFrame(rows).assign(alpha=meta.get("speed_retention_alpha"))
frame.sort_values(["period", "slow", "stop", "run"]).to_csv(os.path.join(job, "edge_stats.csv"), index=False)
print(f"{len(rows)} runs -> edge_stats.csv")
