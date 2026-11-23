"""Summarise downloaded job artifacts into one CSV per (wait model, period, threshold).

Metrics per run (240 one-minute steps): mean relative efficiency (= normalised resilience Re/240),
resilience Re = sum of Ef over minutes, minimum Ef, minutes with Ef < 0.9.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, "**", "MonteCarlo_Results_*.xlsx"), recursive=True):
    match = re.search(r"MonteCarlo_Results_P(\d+)_slow([\dp.]+)_stop([\dp.]+)_", os.path.basename(path))
    meta_path = os.path.join(os.path.dirname(path), "job_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    eff = pd.read_excel(path, sheet_name="Efficiency")
    runs = eff[[c for c in eff.columns if str(c).startswith("Run_")]].to_numpy(float)
    re_sum = runs.sum(axis=0)
    rows.append({
        "wait_model": meta.get("wait_model"), "n_door": meta.get("n_door"),
        "period": int(match.group(1)), "slow": float(match.group(2).replace("p", ".")),
        "stop": float(match.group(3).replace("p", ".")), "runs": runs.shape[1],
        "Ef_mean": float(runs.mean()), "Re_mean": float(re_sum.mean()), "Re_sd": float(re_sum.std(ddof=1)) if runs.shape[1] > 1 else np.nan,
        "minEf_mean": float(runs.min(axis=0).mean()),
        "minutes_below_0p9_mean": float((runs < 0.9).sum(axis=0).mean()),
        "share_rel_gt1": float((runs > 1 + 1e-12).mean()), "max_rel": float(runs.max()),
        "file": os.path.relpath(path, root),
    })
summary = pd.DataFrame(rows).sort_values(["wait_model", "n_door", "period", "slow", "stop"])
summary.to_csv(os.path.join(root, "summary_cells.csv"), index=False, encoding="utf-8-sig")

rank_rows = []
for (model, n_door, period), group in summary.groupby(["wait_model", "n_door", "period"], dropna=False):
    ordered = group.sort_values("Re_mean", ascending=False)
    rank_rows.append({
        "wait_model": model, "n_door": n_door, "period": period, "cells": len(group),
        "Ef_mean_range": f"{group.Ef_mean.min():.4f}-{group.Ef_mean.max():.4f}",
        "best_threshold": f"{ordered.iloc[0].slow:g}/{ordered.iloc[0].stop:g}",
        "ranking_by_Re": " > ".join(f"{r.slow:g}/{r.stop:g}" for r in ordered.itertuples()),
    })
pd.DataFrame(rank_rows).to_csv(os.path.join(root, "summary_ranking.csv"), index=False, encoding="utf-8-sig")

diags = []
for path in glob.glob(os.path.join(root, "**", "wait_diag_*.json"), recursive=True):
    d = json.load(open(path, encoding="utf-8"))
    diags.append({"file": os.path.relpath(path, root), **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.items()}})
if diags:
    pd.DataFrame(diags).to_csv(os.path.join(root, "summary_wait_diag.csv"), index=False, encoding="utf-8-sig")
print(summary.to_string(index=False))
