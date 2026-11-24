"""Patches for the sensitivity experiments (R2-4 speed reduction; later weights and walking distance).

alpha      : speed retention factor on above-ground sections when slow <= max line rain < stop.
             The ninth-version value 0.5 is replaced by the constant SPEED_RETENTION_FACTOR; it acts on
             the riding time (t / alpha) and on the headway used for waiting (hw / alpha), exactly as 0.5 did.
             The stop factor 0.0001, the underground 450 mm rule and the boarding (walking) factors are unchanged.
edge_stats : per run, count riding edge-minutes that are limited (slower than baseline) or stopped (inf),
             and boarding edge-minutes that are closed (inf) or at/above 7200 s. Written as one JSON per run
             under <out>/edge_stats/ and collected by mc/collect_edge_stats.py. Does not change any result.
"""


def replace_once(text, old, new):
    found = text.count(old)
    if found != 1:
        raise SystemExit(f"sens patch anchor found {found} times: {old[:90]!r}")
    return text.replace(old, new)


ALPHA_ANCHOR = (
    "                if is_above == 1: # 地上线主要受瞬时强降雨（能见度）影响\n"
    "                    if max_rain >= self.speed_stop_threshold: return 0.0001 # 达到停运阈值\n"
    "                    elif max_rain >= self.speed_slow_threshold: return 0.5 # 达到限速阈值\n"
)

EDGE_STATS_INIT = "        anomaly_count = 0\n        anomaly_messages = []\n"
EDGE_STATS_LOOP = "            rain_edge_times[edge_index] = current_time\n"
WORKER_END = "        del system\n        del strategy\n        gc.collect()\n        return (run_id, *result_arrays)\n"


def apply(src, alpha=None, edge_stats=False):
    header = ""
    if alpha is not None:
        alpha = float(alpha)
        if not 0.0001 < alpha <= 1.0:
            raise SystemExit(f"alpha must be in (0.0001, 1]: {alpha}")
        header += f"SPEED_RETENTION_FACTOR = {alpha!r}\n"
        src = replace_once(src, ALPHA_ANCHOR, ALPHA_ANCHOR.replace(
            "return 0.5 # 达到限速阈值", "return SPEED_RETENTION_FACTOR # 达到限速阈值 [敏感性 α]"))
    header += f"EDGE_STATS = {bool(edge_stats)!r}\n"
    src = replace_once(src, "import networkx as nx\n", header + "import networkx as nx\n")
    if not edge_stats:
        return src

    src = replace_once(
        src, EDGE_STATS_INIT,
        EDGE_STATS_INIT
        + "        _es = self.__dict__.setdefault('_edge_stats', {'steps': 0, 'riding_edge_min': 0, 'riding_above_edge_min': 0,\n"
          "            'riding_limited_edge_min': 0, 'riding_stopped_edge_min': 0, 'boarding_edge_min': 0,\n"
          "            'boarding_closed_edge_min': 0, 'boarding_ge7200_edge_min': 0,\n"
          "            'minutes_any_limited': 0, 'minutes_any_stopped': 0})\n"
          "        _es['steps'] += 1\n"
          "        _lim_now = 0\n"
          "        _stop_now = 0\n",
    )
    src = replace_once(
        src, EDGE_STATS_LOOP,
        EDGE_STATS_LOOP
        + "            if static_data['type'] == 'riding' and static_data['line_id']:\n"
          "                _es['riding_edge_min'] += 1\n"
          "                if static_data.get('is_above_ground'):\n"
          "                    _es['riding_above_edge_min'] += 1\n"
          "                if current_time == float('inf'):\n"
          "                    _es['riding_stopped_edge_min'] += 1\n"
          "                    _stop_now += 1\n"
          "                elif current_time > baseline_edge_times[edge_index] * (1.0 + 1e-9):\n"
          "                    _es['riding_limited_edge_min'] += 1\n"
          "                    _lim_now += 1\n"
          "            elif static_data['type'] == 'boarding' and static_data['line_id']:\n"
          "                _es['boarding_edge_min'] += 1\n"
          "                if current_time == float('inf'):\n"
          "                    _es['boarding_closed_edge_min'] += 1\n"
          "                elif current_time >= 7200.0:\n"
          "                    _es['boarding_ge7200_edge_min'] += 1\n",
    )
    src = replace_once(
        src, "        if anomaly_count > 0:\n            print(f\"\\n🚨 [CRITICAL SCAN]",
        "        _es['minutes_any_limited'] += int(_lim_now > 0)\n"
        "        _es['minutes_any_stopped'] += int(_stop_now > 0)\n"
        "        if anomaly_count > 0:\n            print(f\"\\n🚨 [CRITICAL SCAN]",
    )
    src = replace_once(
        src, WORKER_END,
        "        if EDGE_STATS:\n"
        "            _es_dir = os.path.join(OUTPUT_SUBDIR, 'edge_stats')\n"
        "            os.makedirs(_es_dir, exist_ok=True)\n"
        "            _slow, _stop = speed_reduction_thresholds\n"
        "            _es_row = dict(getattr(system, '_edge_stats', {}), period=int(period), slow=float(_slow),\n"
        "                           stop=float(_stop), run=int(run_id), seed=None if random_seed is None else int(random_seed))\n"
        "            with open(os.path.join(_es_dir, f'P{int(period)}_slow{_slow:g}_stop{_stop:g}_Run{int(run_id)}.json'),\n"
        "                      'w', encoding='utf-8') as _handle:\n"
        "                json.dump(_es_row, _handle)\n"
        + WORKER_END,
    )
    return src
