"""Read-only audit of intermediate quantities (waiting model A, P100, threshold 5/50, Run_1 seed).

Builds the generated job script, executes its definitions (main() is not run), reproduces the
baseline-flow preparation of main(), then steps one seeded rainfall run and records edge-level
quantities. Writes JSON/CSV summaries to the output folder.
"""
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/claude/metro-rainfall-efficiency-mc")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/audit/out")
OUT.mkdir(parents=True, exist_ok=True)
GEN = OUT / "gen"
subprocess.run([sys.executable, str(REPO / "mc/build_cell.py"), "--period", "100", "--thresholds", "4",
                "--runs", "1", "--wait-model", "ning", "--out", str(GEN), "--jobs", "1"], check=True,
               stdout=subprocess.DEVNULL)
src = (GEN / "run_P100_ning.py").read_text(encoding="utf-8")
sys.path.insert(0, str(GEN))
ns = {"__name__": "audit_module", "__file__": str(GEN / "run_P100_ning.py")}
exec(compile(src, "run_P100_ning.py", "exec"), ns)
BASE = str(REPO / "inputs")

# --- baseline flows exactly as main() prepares them (MC branch) ---
full = pd.read_csv(os.path.join(BASE, "shanghai_metro_flow_with_code_final_20250802-20250831_v4.csv"))
full["time"] = pd.to_datetime(full["time"])
flow_df = full[full["time"].dt.date == pd.Timestamp("2025-08-08").date()].copy()
flow_df.rename(columns={"station_name_mapped": "station"}, inplace=True)
fixes = {"淞淞发路": "淞发路", "淞淞虹路": "淞虹路", "淞淞滨路": "淞滨路", "临洮洮路": "临洮路",
         "颛颛桥": "颛桥", "佘佘山": "佘山", "严御路": "康恒路"}
flow_df["station"] = flow_df["station"].astype(str).str.strip().str.replace("●", "·").str.replace(" ", "").map(lambda s: fixes.get(s, s))
m = flow_df["station"] == "浦电路"
if m.any():
    a = flow_df[m].copy(); b = flow_df[m].copy()
    a["station"] = "浦电路(4号线)"; b["station"] = "浦电路(6号线)"
    a[["inbound", "outbound"]] /= 2; b[["inbound", "outbound"]] /= 2
    flow_df = pd.concat([flow_df[~m], a, b], ignore_index=True)

ctx = ns["build_shared_runtime_context"](BASE, 250, 100, 180, 240, "09:00", flow_df, (5.0, 50.0),
                                         validate_fast_path=False, route_choice_mode="time_then_tau")
base_sys = ns["_DIAG_SYSTEM"]
fed = base_sys.fast_edge_data
G = base_sys.G
report = {}

# --- 1. network ---
types = Counter(sd["type"] for sd in fed)
riding = [sd for sd in fed if sd["type"] == "riding"]
boarding = [sd for sd in fed if sd["type"] == "boarding"]
report["network"] = {
    "station_nodes": len(base_sys.station_nodes), "line_nodes": sum(1 for n in G if G.nodes[n].get("layer") == "line"),
    "edges_by_type": dict(types), "lines": sorted({str(sd["line_id"]) for sd in riding}),
    "above_ground_riding_edges": int(sum(sd["is_above_ground"] for sd in riding)),
    "lines_with_above_ground": sorted({str(sd["line_id"]) for sd in riding if sd["is_above_ground"]}),
    "boarding_edges_without_line_id": int(sum(1 for sd in boarding if not sd["line_id"])),
}

# --- 2. flows: initial static file vs 2025-08-08 hourly ---
static_flow, static_sum, static_count = base_sys._process_cache()[("initial_passenger_flow", os.path.abspath(BASE))]
stations = base_sys.station_nodes
in_static = [s for s in stations if s in static_flow]
flow_names = set(f"{s}_车站层" for s in flow_df["station"].unique())
in_0808 = [s for s in stations if s in flow_names]
report["flows"] = {
    "stations": len(stations), "in_static_file_进站客流": len(in_static),
    "static_file_rows": int(static_count), "static_file_sheets_lines": "1-13,16,17 (15 sheets)",
    "stations_using_default_formula_for_weights": len(stations) - len(in_static),
    "in_20250808_csv": len(in_0808), "not_in_20250808_csv": len(stations) - len(in_0808),
    "examples_not_in_0808": [s for s in stations if s not in flow_names][:15],
    "examples_not_in_static": [s for s in stations if s not in static_flow][:15],
}
# flow used for weights (set at init from static file) vs flow used in waiting (08-08 hourly)
w_board = np.array([sd["flow_weight"] for sd in boarding])
w_ride = np.array([sd["flow_weight"] for sd in riding])
q = lambda a: {p: float(np.quantile(a, p / 100)) for p in (0, 5, 25, 50, 75, 95, 100)}
report["weights"] = {"boarding_flow_weight": q(w_board), "riding_flow_weight": q(w_ride),
                     "riding_weight_equal_100_default": int(np.isclose(w_ride, 100).sum())}

# --- 3. schedule / defaults (09:00-13:00 window, 240 steps) ---
intervals = np.asarray(ctx["intervals"])
no_rule = np.array([len(sd["riding_applicable_rules"]) == 0 for sd in fed])
ride_idx = np.array([sd["type"] == "riding" for sd in fed])
board_idx = np.array([sd["type"] == "boarding" and bool(sd["line_id"]) for sd in fed])
# whether a rule actually matched each minute (default 180 riding / 300 boarding otherwise)
matched = np.zeros_like(intervals, dtype=bool)
for t in range(intervals.shape[0]):
    clock = (base_sys.sim_start_min + t) % 1440
    for e, sd in enumerate(fed):
        matched[t, e] = any(r["start_min"] <= clock < r["end_min"] for r in sd["riding_applicable_rules"])
report["schedule"] = {
    "riding_edges_no_rule": int((no_rule & ride_idx).sum()), "boarding_edges_no_rule": int((no_rule & board_idx).sum()),
    "riding_edge_minutes_default_180": float((~matched[:, ride_idx]).mean()),
    "boarding_edge_minutes_default_300": float((~matched[:, board_idx]).mean()),
    "riding_time_used_s(=timetable interval)": q(intervals[:, ride_idx]),
    "boarding_headway_used_s": q(intervals[:, board_idx]),
    "lines_with_no_rule_riding": sorted({str(sd["line_id"]) for sd, nr in zip(fed, no_rule) if nr and sd["type"] == "riding"}),
}
# geometric inter-station distance for comparison with riding time
hav = base_sys.strategy._haversine if hasattr(base_sys, "strategy") else None
dists = []
for sd in riding:
    p1, p2 = base_sys.node_pos_list[sd["u_idx"]], base_sys.node_pos_list[sd["v_idx"]]
    dists.append(base_sys.strategy._haversine(p1[0], p1[1], p2[0], p2[1]))
dists = np.array(dists)
ride_t0 = intervals[0, ride_idx]
report["schedule"]["interstation_straightline_km"] = q(dists)
report["schedule"]["implied_speed_kmh_at_0900"] = q(dists / (ride_t0 / 3600))

# --- 4. walking, waiting (baseline) ---
bd = np.array([sd["boarding_dist"] for sd in boarding])
report["walking"] = {"boarding_dist_m(=100*degree)": q(bd), "walk_time_s": q(bd / 1.4),
                     "degree_counts": dict(Counter((bd / 100).astype(int).tolist()))}
flows_t = np.asarray(ctx["boarding_flows"])[:, board_idx]
ev = intervals[:, board_idx] + 30.0
waits = np.vectorize(ns["SubwayDIIMSystem"]._calc_wait_time)(ev, np.maximum(flows_t / 3600.0, 1e-4))
report["waiting_baseline"] = {"wait_s": q(waits), "wait_over_headway": q(waits / intervals[:, board_idx]),
                              "share_capped_7200": float((waits >= 7199.999).mean())}

# --- 5. path composition at 09:00 baseline (sampled OD pairs, NetworkX on total time) ---
import networkx as nx
bt = np.asarray(ctx["baseline_times"][0])
walk_c = np.zeros(len(fed)); wait_c = np.zeros(len(fed)); ride_c = np.zeros(len(fed))
for e, sd in enumerate(fed):
    if sd["type"] == "boarding":
        walk_c[e] = sd["boarding_dist"] / 1.4
        wait_c[e] = bt[e] - walk_c[e]
    else:
        ride_c[e] = bt[e]
H = nx.Graph()
for e, sd in enumerate(fed):
    u, v = sd["u_idx"], sd["v_idx"]
    if np.isfinite(bt[e]) and (not H.has_edge(u, v) or H[u][v]["t"] > bt[e]):
        H.add_edge(u, v, t=bt[e], walk=walk_c[e], wait=wait_c[e], ride=ride_c[e], type=sd["type"])
rng = random.Random(0)
st_idx = list(base_sys.station_node_indices)
rows = []
for _ in range(400):
    i, j = rng.sample(st_idx, 2)
    try:
        path = nx.shortest_path(H, i, j, weight="t")
    except nx.NetworkXNoPath:
        continue
    comp = {"walk": 0.0, "wait": 0.0, "ride": 0.0, "boarding_edges": 0, "riding_edges": 0}
    for a, b in zip(path[:-1], path[1:]):
        d = H[a][b]
        comp["walk"] += d["walk"]; comp["wait"] += d["wait"]; comp["ride"] += d["ride"]
        comp["boarding_edges" if d["type"] == "boarding" else "riding_edges"] += 1
    comp["total"] = comp["walk"] + comp["wait"] + comp["ride"]
    rows.append(comp)
pc = pd.DataFrame(rows)
pc["transfers"] = pc.boarding_edges / 2 - 1
report["path_composition_0900"] = {
    "sampled_pairs": len(pc), "total_min": q(pc.total / 60),
    "share_walk_mean": float((pc.walk / pc.total).mean()), "share_wait_mean": float((pc.wait / pc.total).mean()),
    "share_ride_mean": float((pc.ride / pc.total).mean()),
    "boarding_edges_per_path": dict(Counter(pc.boarding_edges.astype(int).tolist())),
    "riding_edges_per_path": q(pc.riding_edges),
    "note": "each station-to-station path uses >= 2 boarding edges; the alighting edge also carries the waiting time",
}
pc.to_csv(OUT / "path_composition_0900.csv", index=False)

# --- 6. one seeded rainfall run, with and without spatial noise ---
def run(seed, noise=True):
    random.seed(seed); np.random.seed(seed)
    strat = ns["ExtremeRainfall"](BASE, max_intensity=250, rainfall_type="chicago", P=100, rain_duration=180,
                                  speed_reduction_thresholds=(5.0, 50.0))
    strat.start_dt = pd.Timestamp("2025-08-08 09:00").to_pydatetime()
    if not noise:
        import math
        def arf(distance, self=strat):
            area = math.pi * distance ** 2
            factor = -0.058 * math.log(area) + 0.9949 if area >= 1 else 1.0
            return max(0, self.Q0 * factor)
        strat._calculate_arf_rainfall = arf
    system = ns["SubwayDIIMSystem"](BASE, strat, sim_start_time="09:00", shared_context=ctx,
                                    use_fast_shortest_path=True, route_choice_mode="time_then_tau")
    rec = {"ride_state": [], "board_closed": [], "water": [], "line_rain": [], "eff": []}
    for t in range(240):
        system.step(t)
        clock = system.sim_start_min + t
        eff = system.calculate_weighted_efficiency(clock, "Weekday", line_stats=system.current_line_stats)
        ls = system.current_line_stats
        states = []
        for sd in system.fast_edge_data:
            if sd["type"] != "riding":
                continue
            c = ls.get(str(sd["line_id"]))
            ctxd = {"is_above_ground": sd["is_above_ground"], "max_rain_above": c["max_rain_above"],
                    "max_water_total": c["max_water_total"]} if c else None
            states.append(system.strategy.get_speed_impact_factor(0.0, "riding", line_context=ctxd))
        rec["ride_state"].append(states)
        waters = np.array([system.G.nodes[n]["water"] for n in system.station_nodes])
        rec["water"].append(waters)
        rec["line_rain"].append({k: v["max_rain_above"] for k, v in ls.items()})
        rec["eff"].append(eff)
    return {k: (np.array(v) if k != "line_rain" else v) for k, v in rec.items()}

seed = 20260208 + 6 * 1_000_000 + 4 * 10_000 + 1
noisy = run(seed, True)
clean = run(seed, False)
def summarize(r, label):
    s = r["ride_state"]
    flips = (np.diff(s, axis=0) != 0).sum(axis=0)
    w = r["water"]
    return {
        f"{label}_restricted_edge_minutes": int((s == 0.5).sum()), f"{label}_stopped_edge_minutes": int((s <= 0.0001).sum()),
        f"{label}_riding_state_changes_per_edge": q(flips),
        f"{label}_water_mm_station_max_over_time": q(w.max(axis=0)),
        f"{label}_station_minutes_water_gt250": int((w > 250).sum()), f"{label}_station_minutes_water_gt450": int((w > 450).sum()),
        f"{label}_Ef_mean": float(np.mean(r["eff"])), f"{label}_Ef_min": float(np.min(r["eff"])),
    }
report["rain_run_P100_5_50_run1"] = {**summarize(noisy, "noisy"), **summarize(clean, "no_noise"),
                                     "seed": seed}
lr_noisy = pd.DataFrame(noisy["line_rain"]); lr_clean = pd.DataFrame(clean["line_rain"])
cols = [c for c in lr_noisy.columns if lr_noisy[c].max() > 0]
report["rain_run_P100_5_50_run1"]["line_max_rain_peak_mmh_noisy_vs_clean"] = {
    c: [round(float(lr_noisy[c].max()), 1), round(float(lr_clean[c].max()), 1)] for c in cols}
np.save(OUT / "ride_state_noisy.npy", noisy["ride_state"]); np.save(OUT / "ride_state_clean.npy", clean["ride_state"])
pd.DataFrame({"eff_noisy": noisy["eff"], "eff_clean": clean["eff"]}).to_csv(OUT / "eff_run1.csv", index=False)

# ARF factor over the study area
import math
d = np.linspace(0.6, 80, 200)
report["arf_factor_by_distance_km"] = {f"{x:.0f}": round(-0.058 * math.log(math.pi * x * x) + 0.9949, 3)
                                       for x in (1, 5, 10, 20, 40, 60, 80)}
(OUT / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=1))
