"""Build arc weight tables for the passenger-flow weight experiment (R2-2, R1-M4, audit item 4).

The v2 model evaluates tau = t / w on 2E directed arcs (boarding, alighting, riding both ways).
In the main results every arc takes the weight of its undirected edge, computed by _init_flow_weights
from the old file 进站客流.xlsx (99 of 408 stations fall back to a default).

Variants written here (one CSV per variant, key = arc_key, value = weight):
  static0808 : the same allocation rule as the main results (boarding edge = station flow / number of
               boarding edges; riding edge = mean of the two adjacent boarding weights), but station flow
               = mean of (inbound + outbound) / 2 over the hourly records 09:00-12:00 of 2025-08-08,
               i.e. the same day and window that drive the waiting times.
  od         : arc weight = passengers assigned to the arc (parallel duplicate arcs share the pooled load). Observed OD of one weekday (15-min cells,
               09:00-13:00) is assigned hour by hour to the no-rain minimum-time paths of the model at
               the middle of that hour (all-or-nothing, the same route rule as step 1 of the two-step
               choice). Boarding arcs get boardings (incl. transfers), alighting arcs alightings, riding
               arcs directional section loads. Negative OD cells are set to 0; weight = max(load, 1).
'unweighted' needs no file (all weights = 1, handled in sens_patch).

usage:
  python mc/make_weights.py static0808 OUT.csv
  python mc/make_weights.py od OUT.csv OD_NPZ          (OD_NPZ from validation_comay_20260903/prepared)
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_model(tmp):
    subprocess.run([sys.executable, str(HERE / "build_cell.py"), "--period", "100", "--thresholds", "1", "--runs", "1",
                    "--wait-model", "ning", "--flags", "v2", "--out", str(tmp)], check=True, stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(tmp))
    spec = importlib.util.spec_from_file_location("v2model", tmp / "run_P100_ning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flow_0808(inputs):
    """2025-08-08 hourly station flows, cleaned exactly as in main() of the MC script."""
    df = pd.read_csv(inputs / "shanghai_metro_flow_with_code_final_20250802-20250831_v4.csv")
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"].dt.date == pd.Timestamp("2025-08-08").date()].copy()
    df.rename(columns={"station_name_mapped": "station"}, inplace=True)
    fixes = {'淞淞发路': '淞发路', '淞淞虹路': '淞虹路', '淞淞滨路': '淞滨路', '临洮洮路': '临洮路',
             '颛颛桥': '颛桥', '佘佘山': '佘山', '严御路': '康恒路'}
    df["station"] = df["station"].map(lambda n: fixes.get(str(n).strip().replace('●', '·').replace(' ', ''),
                                                          str(n).strip().replace('●', '·').replace(' ', '')))
    mask = df["station"] == "浦电路"
    if mask.any():
        rows = df[mask]
        parts = []
        for name in ("浦电路(4号线)", "浦电路(6号线)"):
            part = rows.copy()
            part["station"] = name
            part[["inbound", "outbound"]] /= 2
            parts.append(part)
        df = pd.concat([df[~mask]] + parts, ignore_index=True)
    return df


def build_system(module, inputs, flow_df):
    context = module.build_shared_runtime_context(str(inputs), 250, 100, 180, 240, "09:00", flow_df, (16.0, 35.0))
    system = module._DIAG_SYSTEM
    system._ensure_arc_topology()
    return system, context


def arc_keys(system):
    keys, seen = [], {}
    E = len(system.fast_edge_data)
    for k in range(2 * E):
        sd = system.fast_edge_data[k % E]
        kind = sd["type"] if k < E or sd["type"] == "riding" else "alighting"
        key = f"{system.node_list[system.arc_u[k]]}|{system.node_list[system.arc_v[k]]}|{kind}|{sd['line_id']}"
        seen[key] = seen.get(key, 0) + 1          # parallel duplicate edges exist in the topology
        keys.append(f"{key}#{seen[key]}")
    if len(set(keys)) != len(keys):
        dup = pd.Series(keys).value_counts()
        raise SystemExit(f"arc keys are not unique: {dup[dup > 1].head(5).to_dict()}")
    return keys


def static_weights(system, flow_df):
    window = flow_df[flow_df["time"].dt.hour.isin([9, 10, 11, 12])]
    station_flow = window.assign(f=(window.inbound + window.outbound) / 2).groupby("station").f.mean()
    G = system.G
    flows, missing = {}, []
    for node, data in G.nodes(data=True):
        if data.get("layer") == "station":
            name = node[: -len("_车站层")] if node.endswith("_车站层") else node
            if name in station_flow.index:
                flows[node] = float(station_flow[name])
            else:
                missing.append(node)
    if missing:
        avg_flow = np.mean(list(flows.values()))
        avg_deg = np.mean([G.degree(n) for n, d in G.nodes(data=True) if d.get("layer") == "station"])
        for node in missing:
            flows[node] = avg_flow / avg_deg * G.degree(node)
    E = len(system.fast_edge_data)
    edge_w = np.empty(E)
    line_w = {}
    for k, sd in enumerate(system.fast_edge_data):
        if sd["type"] == "boarding":
            st = system.node_list[sd["boarding_station_idx"]]
            ln = system.node_list[sd["v_idx"] if sd["u_idx"] == sd["boarding_station_idx"] else sd["u_idx"]]
            edge_w[k] = flows[st] / max(1, G.degree[st])
            line_w[ln] = edge_w[k]
    for k, sd in enumerate(system.fast_edge_data):
        if sd["type"] == "riding":
            edge_w[k] = (line_w.get(system.node_list[sd["u_idx"]], 100) + line_w.get(system.node_list[sd["v_idx"]], 100)) / 2
    return np.concatenate((edge_w, edge_w)), {"stations_from_0808": len(flows) - len(missing), "stations_default": len(missing)}


def od_weights(system, context, od_path):
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    z = np.load(od_path)
    od = np.clip(z["flow"].astype(np.float64), 0, None)      # (96 slots of 15 min from 02:00, n, n)
    names = [str(s) for s in z["station_names"]]
    station_ids = {system.node_list[i]: i for i in system.station_node_indices}
    order = []
    for name in names:
        node = f"{name}_车站层"
        if node not in station_ids:
            raise SystemExit(f"OD station not in model: {name}")
        order.append(station_ids[node])
    order = np.asarray(order)
    n_nodes = len(system.node_list)
    load = np.zeros(2 * len(system.fast_edge_data))
    hours = []
    for hour in (9, 10, 11, 12):
        s0 = (hour * 60 - 120) // 15
        demand = od[s0:s0 + 4].sum(axis=0)
        np.fill_diagonal(demand, 0.0)
        step = hour * 60 + 30 - 9 * 60
        arc_times = np.asarray(context["baseline_times"][step], dtype=np.float64)
        pair_cost, pair_time, _ = system._arc_pair_costs(arc_times)
        # arc actually used for each node pair (same choice as the model)
        combined_choice = system.arc_pair_first.copy()
        tau = system._times_to_tau(arc_times)
        usable = np.isfinite(arc_times) & (arc_times > 0) & np.isfinite(tau) & (tau > 0)
        lam = system.route_tie_rtol * float(np.min(system.arc_flow_weights[usable]))
        combined = np.where(usable, arc_times + lam * tau, np.inf)
        for pair_index, idx in system.arc_pair_multi:
            combined_choice[pair_index] = idx[int(np.argmin(combined[idx]))]
        valid = np.isfinite(pair_cost)
        pairs = system.arc_pair_nodes[valid]
        graph = sparse.csr_matrix((pair_cost[valid], (pairs[:, 0], pairs[:, 1])), shape=(n_nodes, n_nodes))
        dist, pred = dijkstra(graph, directed=True, indices=order, return_predecessors=True)
        unassigned = 0.0
        for a in range(len(order)):
            node_demand = np.zeros(n_nodes)
            node_demand[order] = demand[a]
            reach = np.isfinite(dist[a])
            unassigned += node_demand[~reach].sum()
            node_demand[~reach] = 0.0
            for v in np.argsort(-np.where(reach, dist[a], -np.inf)):
                p = pred[a, v]
                if p < 0 or node_demand[v] == 0:
                    continue
                arc = combined_choice[system.arc_pair_id[p, v]]
                load[arc] += node_demand[v]
                node_demand[p] += node_demand[v]
        hours.append({"hour": hour, "od_trips": float(demand.sum()), "unassigned": float(unassigned)})
    # parallel duplicate arcs (same nodes, kind and line) carry the same service: pool their load
    base_keys = [key.rsplit("#", 1)[0] for key in arc_keys(system)]
    pooled = pd.Series(load).groupby(pd.Series(base_keys)).transform("sum").to_numpy()
    return np.maximum(pooled, 1.0), {"hours": hours, "arcs_zero_load": int((pooled == 0).sum()), "arcs": int(load.size)}


def main():
    mode, out = sys.argv[1], Path(sys.argv[2])
    inputs = ROOT / "inputs"
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"make_weights_{mode}"
    module = load_model(tmp)
    flow_df = flow_0808(inputs)
    system, context = build_system(module, inputs, flow_df)
    keys = arc_keys(system)
    if mode == "static0808":
        w, info = static_weights(system, flow_df)
    elif mode == "od":
        w, info = od_weights(system, context, sys.argv[3])
    else:
        raise SystemExit(mode)
    base = np.asarray(system.arc_flow_weights, dtype=np.float64)
    from scipy.stats import spearmanr
    info.update(mode=mode, spearman_vs_main=float(spearmanr(w, base).correlation))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"arc_key": keys, "weight": w, "weight_main": base}).to_csv(out, index=False)
    print(info)


if __name__ == "__main__":
    main()
