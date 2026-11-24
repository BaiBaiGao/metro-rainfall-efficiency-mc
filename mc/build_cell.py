"""Generate a runnable Monte Carlo script for one (period, thresholds, wait model) job.

The base script is the ninth-version two-step-route execution script
(mc/base_ninth_two_step.py, copied verbatim from 投稿结果重算_两步选路_20261118\\蒙特卡洛\\执行脚本\\P2.py).
Only the lines below are changed; every patch anchor must occur exactly once.

Wait models
-----------
legacy : original code. lambda = station flow / 3600 (passengers/s). Saturates (7200 s) almost everywhere.
ning   : Ning et al. (2019) conversion. lambda_tilde = (line flow / 3600 / n_door) * E(Y) / D,
         E(Y) = 0.7 s per passenger, D = 30 s minimum dwell. Station flow is split equally over the
         boarding edges (lines) of the station. Formula, 0.9 threshold and 7200 s cap unchanged.
half   : wait = headway / 2 (random arrival), capped at 7200 s.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "base_ninth_two_step.py"


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"patch anchor found {count} times: {old[:80]!r}")
    return text.replace(old, new)


def parse_indices(spec):
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not out or min(out) < 1 or max(out) > 23:
        raise SystemExit(f"bad threshold indices: {spec}")
    return sorted(set(out))


FLAG_NAMES = ("alight", "distance", "noise", "loop")


def parse_flags(spec):
    """'' -> all off; 'v2' -> all on; otherwise '+'-joined subset of alight, distance, noise, loop."""
    spec = (spec or "").strip()
    names = set(FLAG_NAMES) if spec == "v2" else {x for x in spec.split("+") if x}
    unknown = names - set(FLAG_NAMES)
    if unknown:
        raise SystemExit(f"unknown model flags: {sorted(unknown)}")
    flags = {name: name in names for name in FLAG_NAMES}
    flags.update(detour=1.2, speed=35.0, maglev_s=440.0)
    return flags


def build(args):
    src = BASE.read_text(encoding="utf-8")
    out_dir = Path(args.out).resolve()
    inputs = Path(args.inputs).resolve()
    thresholds = parse_indices(args.thresholds)

    header = (
        "from checkpoint import checked_batches, Path\n"
        f"OUTPUT_SUBDIR={str(out_dir)!r}\n"
        f"N_JOBS_GLOBAL={int(args.jobs)}\n"
        "BATCH_SIZE=4\n"
        f"CHECKPOINT_ROOT=Path({str(out_dir / 'checkpoints')!r})\n"
        f"WAIT_MODEL={args.wait_model!r}\n"
        f"NING_CONVERSION={0.7 / (float(args.n_door) * 30.0)!r}\n"
        f"FLOW_SPLIT={args.wait_model == 'ning'!r}\n"
        "WAIT_CAP=7200.0\n"
        f"SELECTED_THRESHOLDS={thresholds!r}\n"
        f"WAIT_DIAG={bool(args.diag)!r}\n"
    )
    # drop the original Windows-specific block (lines 46-50) and insert ours
    start = src.index("from checkpoint import checked_batches, Path\n")
    end = src.index("import networkx as nx\n", start)
    src = src[:start] + header + src[end:]

    src = replace_once(src, "MC_ITERATIONS_GLOBAL = 300\n", f"MC_ITERATIONS_GLOBAL = {int(args.runs)}\n")
    src = replace_once(
        src,
        "    BASE_PATH = 'F:\\\\生活、学习与工作\\\\【研究推进】\\\\【复杂系统随机动力学】\\\\coding filepath'\n",
        f"    BASE_PATH = {str(inputs)!r}\n",
    )
    src = replace_once(src, "        RAIN_P_PERIOD = 2       \n", f"        RAIN_P_PERIOD = {int(args.period)}\n")
    src = replace_once(
        src,
        "            for threshold_index, speed_thresholds in enumerate(threshold_pairs, start=1):\n",
        "            for threshold_index, speed_thresholds in enumerate(threshold_pairs, start=1):\n"
        "                if threshold_index not in SELECTED_THRESHOLDS:\n"
        "                    continue\n",
    )
    src = replace_once(
        src,
        "    @staticmethod\n    def _calc_wait_time(EV, L, dwell_time=30.0):\n        rho = L * 2 * EV\n",
        "    @staticmethod\n    def _calc_wait_time(EV, L, dwell_time=30.0):\n"
        "        if WAIT_MODEL == 'half':\n"
        "            return min(0.5 * (EV - dwell_time), WAIT_CAP)\n"
        "        if WAIT_MODEL == 'ning':\n"
        "            L = L * NING_CONVERSION\n"
        "        rho = L * 2 * EV\n",
    )
    src = replace_once(
        src,
        "                est['boarding_node_flow'] = self.G.nodes[station_node].get('flow', 1000)\n",
        "                est['boarding_node_flow'] = self.G.nodes[station_node].get('flow', 1000)\n"
        "                if FLOW_SPLIT:\n"
        "                    _n_board = sum(1 for _a, _b, _attr in self.G.edges(station_node, data=True)\n"
        "                                   if _attr.get('type') == 'boarding')\n"
        "                    est['boarding_split'] = 1.0 / max(1, _n_board)\n"
        "                    est['boarding_node_flow'] = est['boarding_node_flow'] * est['boarding_split']\n",
    )
    src = replace_once(
        src,
        "                    static_data['boarding_node_flow'] = self.G.nodes[node_name].get('flow', 1000)\n",
        "                    static_data['boarding_node_flow'] = self.G.nodes[node_name].get('flow', 1000) * static_data.get('boarding_split', 1.0)\n",
    )
    src = replace_once(
        src,
        "        context = system.build_shared_context(simulation_duration)\n",
        "        context = system.build_shared_context(simulation_duration)\n"
        "        globals()['_DIAG_SYSTEM'] = system\n",
    )
    src = replace_once(
        src,
        "    print(f\"⏱️ {scenario_label} 初始化与基准预计算总耗时: {time.perf_counter() - context_started:.1f}s\")\n",
        "    print(f\"⏱️ {scenario_label} 初始化与基准预计算总耗时: {time.perf_counter() - context_started:.1f}s\")\n"
        "    if WAIT_DIAG:\n"
        "        _write_wait_diag(shared_context, OUTPUT_DIR, scenario_label)\n",
    )
    from model_v2_patch import apply as apply_v2
    flags = parse_flags(args.flags)
    src = apply_v2(src, flags)
    from sens_patch import apply as apply_sens
    src = apply_sens(src, alpha=args.alpha, edge_stats=args.edge_stats, weights=args.weights, inputs=inputs)
    src = src.replace('if __name__ == "__main__":\n    main()', DIAG_FN + '\nif __name__ == "__main__":\n    main()')
    if "_write_wait_diag(shared_context" not in src or "def _write_wait_diag" not in src:
        raise SystemExit("diag patch failed")
    compile(src, "generated", "exec")

    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / f"run_P{args.period}_{args.wait_model}.py"
    script.write_text(src, encoding="utf-8")
    (out_dir / "checkpoint.py").write_text((HERE / "checkpoint.py").read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "period": int(args.period), "thresholds": thresholds, "runs": int(args.runs),
        "wait_model": args.wait_model, "n_door": float(args.n_door),
        "ning_conversion": 0.7 / (float(args.n_door) * 30.0), "flow_split": args.wait_model == "ning",
        "base_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "generated_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "model_flags": parse_flags(args.flags),
        "speed_retention_alpha": args.alpha if args.alpha is not None else 0.5,
        "alpha_patched": args.alpha is not None,
        "edge_stats": bool(args.edge_stats),
        "weights": args.weights,
        "weights_sha256": (hashlib.sha256((inputs / args.weights[5:]).read_bytes()).hexdigest()
                           if args.weights.startswith("file:") else None),
    }
    (out_dir / "job_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(script)


DIAG_FN = r'''
def _write_wait_diag(shared_context, output_dir, scenario_label):
    """Baseline (no-rain) waiting-time diagnostics over all boarding edges and time steps."""
    system = globals().get('_DIAG_SYSTEM')
    if system is None:
        return
    idx = [i for i, sd in enumerate(system.fast_edge_data) if sd['type'] == 'boarding' and sd['line_id']]
    intervals = np.asarray(shared_context['intervals'])[:, idx]
    flows = np.asarray(shared_context['boarding_flows'])[:, idx]
    ev = intervals + 30.0
    lam = np.maximum(flows / 3600.0, 0.0001)
    if WAIT_MODEL == 'ning':
        lam = lam * NING_CONVERSION
    rho = 2.0 * lam * ev
    waits = np.vectorize(SubwayDIIMSystem._calc_wait_time)(ev, np.maximum(flows / 3600.0, 0.0001))
    q = lambda a: {p: float(np.quantile(a, p / 100)) for p in (0, 5, 25, 50, 75, 95, 100)}
    diag = {
        'scenario': scenario_label, 'wait_model': WAIT_MODEL, 'boarding_edges': len(idx),
        'time_steps': int(intervals.shape[0]),
        'rho_quantiles': q(rho) if WAIT_MODEL != 'half' else None,
        'share_rho_ge_0p9': float((rho >= 0.9).mean()) if WAIT_MODEL != 'half' else None,
        'share_wait_capped_7200': float((waits >= 7199.999).mean()),
        'wait_quantiles_s': q(waits),
        'headway_quantiles_s': q(intervals),
        'flow_quantiles_per_h': q(flows),
    }
    with open(os.path.join(output_dir, f'wait_diag_{scenario_label}_{WAIT_MODEL}.json'), 'w', encoding='utf-8') as handle:
        json.dump(diag, handle, ensure_ascii=False, indent=2)
    print('WAIT_DIAG', json.dumps(diag, ensure_ascii=False))
'''

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--period", type=int, required=True, choices=[2, 5, 10, 25, 50, 100])
    p.add_argument("--thresholds", default="1-21", help="1-based indices into the 23 threshold pairs (22 = 16/35, 23 = 80/100), e.g. 1-7 or 1,4,9")
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--wait-model", choices=["legacy", "ning", "half"], required=True)
    p.add_argument("--n-door", type=float, default=12.0)
    p.add_argument("--inputs", default=str(HERE.parent / "inputs"))
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--diag", action="store_true")
    p.add_argument("--flags", default="", help="model revisions: v2 (all) or alight+distance+noise+loop subset")
    p.add_argument("--alpha", type=float, default=None, help="speed retention factor on limited above-ground sections (ninth version: 0.5)")
    p.add_argument("--weights", default="main", help="main | unweighted | file:<csv relative to inputs>")
    p.add_argument("--edge-stats", action="store_true", help="write per-run limited/stopped edge-minute counts")
    build(p.parse_args())
