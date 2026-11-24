"""Paired summary of a sensitivity experiment against the v2 baseline (same seeds, Run_1..Run_N).

usage: summarize_sens.py BASELINE_DIR SENS_DIR OUT_DIR N PARAMETER

BASELINE_DIR : downloaded artifacts of the ning:12:v2 run (variant value 0.5 for speed_retention)
SENS_DIR     : downloaded artifacts of the sensitivity run(s)
Outputs (OUT_DIR):
  sens_cells.csv    one row per (parameter, variant, period, slow, stop)
  sens_ranking.csv  one row per (parameter, variant, period, threshold set) — 21 original pairs and all 23
  sens_edge.csv     limited / stopped edge-minutes per (variant, period, slow, stop)
  sens_qa.json      file counts, run alignment, edge-count invariance across variants
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

base_dir, sens_dir, out_dir, n_runs, parameter = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
os.makedirs(out_dir, exist_ok=True)
ORIGINAL = {5.0, 10.0, 20.0, 35.0, 50.0, 70.0, 100.0}
cols = [f"Run_{i}" for i in range(1, n_runs + 1)]
qa = {"n_runs": n_runs, "parameter": parameter}


def variant_of(meta):
    if parameter == "speed_retention":
        return float(meta.get("speed_retention_alpha", 0.5))
    raise SystemExit(f"unknown parameter {parameter}")


def load(root, label):
    out = {}
    files = glob.glob(os.path.join(root, "**", "MonteCarlo_Results_*.xlsx"), recursive=True)
    for path in files:
        m = re.search(r"MonteCarlo_Results_P(\d+)_slow([\dp.]+)_stop([\dp.]+)_", os.path.basename(path))
        meta_path = os.path.join(os.path.dirname(path), "job_meta.json")
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        if not meta.get("model_flags", {}).get("alight"):
            continue  # skip legacy regression workbooks
        key = (variant_of(meta), int(m.group(1)), float(m.group(2).replace("p", ".")), float(m.group(3).replace("p", ".")))
        eff = pd.read_excel(path, sheet_name="Efficiency")
        missing = [c for c in cols if c not in eff.columns]
        if missing:
            raise SystemExit(f"{path}: missing {missing[:3]}")
        if key in out:
            raise SystemExit(f"duplicate cell {key}")
        out[key] = eff[cols].to_numpy(float)
    qa[f"{label}_workbooks"] = len(out)
    return out


base = load(base_dir, "baseline")
sens = load(sens_dir, "sensitivity")
cells = {**base, **sens}

rows = []
for (variant, period, slow, stop), runs in sorted(cells.items()):
    ref = base.get((0.5 if parameter == "speed_retention" else None, period, slow, stop))
    re_run = runs.sum(axis=0)
    row = {
        "parameter": parameter, "variant": variant, "period": period, "slow": slow, "stop": stop,
        "original_21": slow in ORIGINAL and stop in ORIGINAL, "runs": runs.shape[1],
        "Re_mean": re_run.mean(), "Re_sd": re_run.std(ddof=1),
        "Re_p05": np.quantile(re_run, 0.05), "Re_p95": np.quantile(re_run, 0.95),
        "Ef_mean": runs.mean(), "minEf_mean": runs.min(axis=0).mean(),
        "dur_below_0p9_mean": (runs < 0.9).sum(axis=0).mean(),
        "share_rel_gt1": (runs > 1 + 1e-12).mean(), "max_rel": runs.max(),
    }
    if ref is not None:
        d = re_run - ref.sum(axis=0)
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if d.std(ddof=1) > 0 else 0.0
        loss_ref = 240.0 - ref.sum(axis=0).mean()
        row.update({
            "dRe_mean": d.mean(), "dRe_ci95_low": d.mean() - half, "dRe_ci95_high": d.mean() + half,
            "dRe_pct_of_Re": 100 * d.mean() / ref.sum(axis=0).mean(),
            "loss_ratio": (240.0 - re_run.mean()) / loss_ref if loss_ref > 0 else np.nan,
            "share_runs_lower": float((d < -1e-12).mean()), "share_runs_higher": float((d > 1e-12).mean()),
        })
    rows.append(row)
cell_df = pd.DataFrame(rows)

edge_frames = [pd.read_csv(p) for p in glob.glob(os.path.join(sens_dir, "**", "edge_stats.csv"), recursive=True)]
edge_df = pd.DataFrame()
if edge_frames:
    raw = pd.concat(edge_frames, ignore_index=True)
    raw = raw[raw.run <= n_runs]
    # limited/stopped classification depends only on rain and thresholds, so it must not change with alpha
    key = ["period", "slow", "stop", "run"]
    spread = raw.groupby(key)[["riding_limited_edge_min", "riding_stopped_edge_min"]].agg(lambda s: s.max() - s.min())
    qa["edge_counts_identical_across_variants"] = bool((spread.to_numpy() == 0).all())
    raw["limited_share_of_above"] = raw.riding_limited_edge_min / raw.riding_above_edge_min
    raw["stopped_share_of_above"] = raw.riding_stopped_edge_min / raw.riding_above_edge_min
    agg = raw.groupby(["alpha", "period", "slow", "stop"]).agg(
        runs=("run", "count"),
        limited_edge_min_mean=("riding_limited_edge_min", "mean"),
        stopped_edge_min_mean=("riding_stopped_edge_min", "mean"),
        limited_share_of_above_mean=("limited_share_of_above", "mean"),
        stopped_share_of_above_mean=("stopped_share_of_above", "mean"),
        minutes_any_limited_mean=("minutes_any_limited", "mean"),
        minutes_any_stopped_mean=("minutes_any_stopped", "mean"),
        boarding_ge7200_edge_min_mean=("boarding_ge7200_edge_min", "mean"),
        boarding_closed_edge_min_mean=("boarding_closed_edge_min", "mean"),
    ).reset_index().rename(columns={"alpha": "variant"})
    edge_df = agg
    cell_df = cell_df.merge(agg.drop(columns="runs"), on=["variant", "period", "slow", "stop"], how="left")
cell_df.to_csv(os.path.join(out_dir, "sens_cells.csv"), index=False, encoding="utf-8-sig")
if not edge_df.empty:
    edge_df.to_csv(os.path.join(out_dir, "sens_edge.csv"), index=False, encoding="utf-8-sig")

rank_rows = []
baseline_variant = 0.5
for (variant, period), group in cell_df.groupby(["variant", "period"]):
    for set_name, sub in (("original_21", group[group.original_21]), ("all_23", group)):
        ref = cell_df[(cell_df.variant == baseline_variant) & (cell_df.period == period)]
        if set_name == "original_21":
            ref = ref[ref.original_21]
        merged = sub.merge(ref[["slow", "stop", "Re_mean"]], on=["slow", "stop"], suffixes=("", "_ref"))
        if len(merged) < 2:
            continue
        ordered = merged.sort_values("Re_mean", ascending=False)
        ordered_ref = merged.sort_values("Re_mean_ref", ascending=False)
        label = lambda r: f"{r.slow:g}/{r.stop:g}"
        top3 = {label(r) for r in ordered.head(3).itertuples()}
        top3_ref = {label(r) for r in ordered_ref.head(3).itertuples()}
        worst = merged.loc[(merged.Re_mean - merged.Re_mean_ref).abs().idxmax()]
        rank_rows.append({
            "parameter": parameter, "variant": variant, "period": period, "set": set_name, "cells": len(merged),
            "spearman": stats.spearmanr(merged.Re_mean, merged.Re_mean_ref).correlation,
            "kendall": stats.kendalltau(merged.Re_mean, merged.Re_mean_ref).correlation,
            "top3_overlap": len(top3 & top3_ref),
            "best": label(ordered.iloc[0]), "best_ref": label(ordered_ref.iloc[0]),
            "best_same": label(ordered.iloc[0]) == label(ordered_ref.iloc[0]),
            "worst": label(ordered.iloc[-1]), "worst_ref": label(ordered_ref.iloc[-1]),
            "max_abs_dRe": float((merged.Re_mean - merged.Re_mean_ref).abs().max()),
            "max_abs_dRe_at": f"{worst.slow:g}/{worst.stop:g}",
            "ranking_by_Re": " > ".join(label(r) for r in ordered.itertuples()),
        })
pd.DataFrame(rank_rows).to_csv(os.path.join(out_dir, "sens_ranking.csv"), index=False, encoding="utf-8-sig")

qa["cells_per_variant"] = cell_df.groupby("variant").size().to_dict()
qa["cells_without_baseline"] = int(cell_df.dRe_mean.isna().sum()) if "dRe_mean" in cell_df else None
with open(os.path.join(out_dir, "sens_qa.json"), "w", encoding="utf-8") as handle:
    json.dump(qa, handle, ensure_ascii=False, indent=2, default=str)
print(json.dumps(qa, ensure_ascii=False, indent=2, default=str))
if rank_rows:
    print(pd.DataFrame(rank_rows)[["variant", "period", "set", "spearman", "kendall", "top3_overlap", "best", "best_ref"]].to_string(index=False))
