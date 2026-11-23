"""Generate the standalone submission-pair runner from its archived notebook cell."""
import argparse
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERIODS = (2, 5, 10, 25, 50, 100)
THRESHOLDS = ("16_35", "80_100")
MODES = ("legacy_tau", "time_then_tau")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one patch anchor: {old!r}")
    return source.replace(old, new, 1)


def notebook_cell(threshold, period):
    book = HERE.parent / "sources" / f"paired_{threshold}.ipynb"
    cells = json.loads(book.read_text(encoding="utf-8-sig"))["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    matches = [
        source for source in sources
        if re.search(rf"(?m)^        RAIN_P_PERIOD = {period}\s*$", source)
    ]
    if len(matches) != 1:
        raise ValueError(f"{book}: expected one cell for P{period}")
    return matches[0]


def patch_wait(source):
    initialization = "                est['boarding_node_flow'] = self.G.nodes[u if self.G.nodes[u]['layer']=='station' else v].get('flow', 1000)"
    source = replace_once(source, initialization, """\
                station_node = u if self.G.nodes[u]['layer'] == 'station' else v
                est['boarding_node_flow'] = self.G.nodes[station_node].get('flow', 1000)
                if WAIT_MODEL == 'ning':
                    est['boarding_edge_count'] = sum(
                        attrs['type'] == 'boarding'
                        for _, _, attrs in self.G.edges(station_node, data=True)
                    )
                    est['boarding_node_flow'] /= est['boarding_edge_count']""")
    refresh = "                                    est['boarding_node_flow'] = self.G.nodes[st_node_name].get('flow', 1000)"
    source = replace_once(source, refresh, refresh + """
                                    if WAIT_MODEL == 'ning':
                                        est['boarding_node_flow'] /= est['boarding_edge_count']""")
    wait = "                def calc_w(EV, L):"
    return replace_once(source, wait, wait + """
                    if WAIT_MODEL == 'ning':
                        L *= NING_CONVERSION""")


def build(args):
    source = patch_wait(notebook_cell(args.threshold, args.period))
    routing = (HERE / "paired_routing.pyfrag").read_text(encoding="utf-8")
    routing = replace_once(routing, "        edge_times = []", """\
        edge_times = []
        if WAIT_MODEL == 'ning' and not hasattr(self, 'baseline_wait_edges'):
            self.baseline_wait_edges = self.baseline_wait_capped = 0""")
    routing = replace_once(
        routing, "            dry, _, _ = self._calc_times_fast(edge, 0.0, clock, line_stats=None)",
        """\
            dry, _, dry_debug = self._calc_times_fast(edge, 0.0, clock, line_stats=None)
            if WAIT_MODEL == 'ning' and 't_wait_n' in dry_debug:
                self.baseline_wait_edges += 1
                self.baseline_wait_capped += dry_debug['t_wait_n'] >= 7200.0""",
    )
    marker = "    def calculate_weighted_efficiency("
    source = replace_once(source, marker, routing + marker)
    start = source.index("def simulation_worker(")
    end = source.index("# [新增] 数据一致性校验函数", start)
    worker = source[start:end]
    worker = replace_once(worker, "\n    try:\n", """
    started = time.perf_counter()
    seed = 29000000 + period * 10000 + SEED_OFFSET + run_id
    random.seed(seed)
    np.random.seed(seed)
    try:
        new_efficiency = []
""")
    worker, hits = re.subn(
        r"(?m)^            ef = sys_obj.calculate_weighted_efficiency[^\n]+",
        "            ef, new_ef = sys_obj.paired_efficiencies(curr_clk, getattr(sys_obj, 'current_line_stats', None))\n"
        "            new_efficiency.append(new_ef)",
        worker,
    )
    if hits != 1:
        raise ValueError("Expected one worker efficiency call")
    worker = replace_once(worker, "        del sys_obj", """\
        diagnostics = {
            'run_id': run_id,
            'seed': seed,
            'baseline_wait_edges': getattr(sys_obj, 'baseline_wait_edges', 0),
            'baseline_wait_capped': getattr(sys_obj, 'baseline_wait_capped', 0),
        }
        del sys_obj""")
    worker = replace_once(worker, "        return (run_id, r_eff", """\
        diagnostics['elapsed_seconds'] = time.perf_counter() - started
        return (run_id, r_eff""")
    worker = replace_once(worker, "r_aff, r_mnr)", "r_aff, r_mnr, new_efficiency, diagnostics)")
    worker = replace_once(
        worker, "        return (run_id, None, str(e), None, None, None, None, None, None)",
        "        raise RuntimeError(f'Run_{run_id} failed') from e",
    )

    # Keep the notebook's random scenario and exact flow preparation.
    scenario = source.index("        # 原始蒙特卡洛模式配置", end)
    scenario_end = source.index("    # N_JOBS 控制并行核心数", scenario)
    scenario_code = "".join(
        line[4:] if line.startswith("    ") else line
        for line in source[scenario:scenario_end].splitlines(keepends=True)
    )
    scenario_code, hits = re.subn(
        r"(?m)^    MC_ITERATIONS = 100[^\n]*", "    MC_ITERATIONS = RUN_END - RUN_START + 1", scenario_code
    )
    if hits != 1:
        raise ValueError("Expected one Monte Carlo run count")
    source = source[:start] + worker + "\ndef main():\n"
    source += '    BASE_PATH = Path(__file__).resolve().parent / INPUTS_RELATIVE\n' + scenario_code + RUNNER
    source = replace_once(source, '            print(f"❌ 导出失败: {e}")', "            raise")
    if re.search(r"""['"][A-Za-z]:\\|17:20""", source):
        raise ValueError("Generated code still contains a Windows path or pause")

    out = Path(args.out).resolve()
    relative_inputs = Path(os.path.relpath(Path(args.inputs).resolve(), out)).as_posix()
    offset = 0 if args.threshold == "16_35" else 2000
    header = (
        "from pathlib import Path\n"
        "from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra\n"
        "from joblib import parallel_backend\n"
        f"INPUTS_RELATIVE = {relative_inputs!r}\n"
        f"WAIT_MODEL = {args.wait_model!r}\n"
        f"NING_CONVERSION = 0.7 / ({args.n_door} * 30)\n"
        f"SEED_OFFSET = {offset}\n"
        f"RUN_START, RUN_END, N_JOBS = {args.run_start}, {args.run_end}, {args.jobs}\n"
        f"MODES = {MODES!r}\n\n"
    )
    script = out / "run.py"
    compile(header + source, str(script), "exec")
    out.mkdir(parents=True, exist_ok=True)
    script.write_text(header + source, encoding="utf-8")
    return script


RUNNER = """
    if flow_df is None or flow_df.empty:
        raise RuntimeError('Missing baseline flows for 2025-08-08')
    output = Path(__file__).resolve().parent
    for mode in MODES:
        if list((output / mode).glob('MonteCarlo*.xlsx')):
            raise FileExistsError(f'Use a fresh output directory: {output / mode}')
    recorders = {mode: MonteCarloRecorder(list(range(SIM_DURATION))) for mode in MODES}
    diagnostics = []
    with parallel_backend('loky', n_jobs=N_JOBS, inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS, max_nbytes=None, verbose=10, return_as='generator')(
            delayed(simulation_worker)(
                run_id, BASE_PATH, 250, RAIN_P_PERIOD, RAIN_DURATION,
                SIM_DURATION, START_TIME_STR, real_flow_data=flow_df,
            )
            for run_id in range(RUN_START, RUN_END + 1)
        )
        for run_id, old, *tail in results:
            *metrics, new, diagnostic = tail
            if any(len(series) != SIM_DURATION for series in (old, new, *metrics)):
                raise ValueError(f'Run_{run_id} has incomplete series')
            recorders['legacy_tau'].add_run_data(run_id, old, *metrics)
            recorders['time_then_tau'].add_run_data(run_id, new, *metrics)
            total = diagnostic['baseline_wait_edges']
            diagnostic['baseline_wait_capped_share'] = (
                diagnostic['baseline_wait_capped'] / total if total else None
            )
            diagnostics.append(diagnostic)
            print('RUN_DIAGNOSTICS', json.dumps(diagnostic), flush=True)
    for mode, recorder in recorders.items():
        folder = output / mode
        folder.mkdir(parents=True, exist_ok=True)
        recorder.save_to_excel(folder)
    (output / 'paired_diagnostics.json').write_text(
        json.dumps(diagnostics, indent=2) + '\\n', encoding='utf-8'
    )


if __name__ == "__main__":
    main()
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", choices=THRESHOLDS, required=True)
    parser.add_argument("--period", type=int, choices=PERIODS, required=True)
    parser.add_argument("--run-start", type=int, required=True)
    parser.add_argument("--run-end", type=int, required=True)
    parser.add_argument("--wait-model", choices=("legacy", "ning"), required=True)
    parser.add_argument("--n-door", type=int, default=12)
    parser.add_argument("--inputs", type=Path, default=HERE.parent / "inputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.run_start <= args.run_end <= 100:
        parser.error("Require 1 <= run-start <= run-end <= 100")
    if args.jobs < 1 or args.n_door < 1:
        parser.error("jobs and n-door must be positive")
    print(build(args))
