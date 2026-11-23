"""Model revisions of 2026-11-23 (audit items 1-3 and the line-4 loop schedule), applied as
guarded text patches to the ninth-version two-step script. With every flag off the generated
script behaves exactly like the original (checked by the regression gate).

Flags (module-level constants written into the generated script):
  ALIGHT_WALK_ONLY  item 1: directed station access. Boarding arc (station -> line) = walk + wait;
                    alighting arc (line -> station) = walk only. Riding edges are used in both
                    directions with the same time. Route choice and evaluation run on the
                    directed graph (SciPy directed Dijkstra; NetworkX MultiDiGraph reference).
  RIDING_TIME_MODE  item 2: 'interval' (original: riding time = timetable headway) or 'distance':
                    riding time = RIDING_DETOUR x straight-line inter-station distance / RIDING_SPEED_KMH;
                    Maglev uses MAGLEV_RIDE_S. Headways are still used for waiting.
  NOISE_MODE        item 3: 'minute' (original: N(0, 0.3) redrawn at every call) or 'fixed':
                    one N(0, 0.3) draw per location per simulation run.
  FIX_LOOP_SCHEDULE timetable rules whose from/to station are the same station (Line 4 loop)
                    apply to every station of that line.
"""


def replace_once(text, old, new, count=1):
    found = text.count(old)
    if found != count:
        raise SystemExit(f"model_v2 patch anchor found {found} times (expected {count}): {old[:90]!r}")
    return text.replace(old, new)


THRESHOLD_EXTRA = """SPEED_REDUCTION_THRESHOLD_PAIRS = [
    (slow, stop)
    for slow in THRESHOLD_VALUES
    for stop in THRESHOLD_VALUES
    if slow < stop
]
# [20261123] indices 22 and 23: the two submission scenarios (Fig 10-14), same seed scheme
SPEED_REDUCTION_THRESHOLD_PAIRS += [(16.0, 35.0), (80.0, 100.0)]
"""

ARC_METHODS = r'''
    # ===== [20261123 修改 1] 有向进出站：上车弧 = 步行 + 候车，下车弧 = 只有步行 =====
    # 弧向量长度 2E：前 E 个与原边时间完全相同（上车边取上车方向）；
    # 后 E 个为反向弧：上车边的反向 = 下车（只计步行），运行边的反向 = 同一时间。
    def _is_arc_vector(self, times):
        return ALIGHT_WALK_ONLY and np.asarray(times).shape[-1] == 2 * len(self.fast_edge_data)

    def _ensure_arc_topology(self):
        if getattr(self, '_arc_ready', False):
            return
        E = len(self.fast_edge_data)
        u = np.empty(E, dtype=np.int64); v = np.empty(E, dtype=np.int64)
        board = np.zeros(E, dtype=bool)
        for k, sd in enumerate(self.fast_edge_data):
            if sd['type'] == 'boarding':
                station = sd['boarding_station_idx']
                line_node = sd['v_idx'] if sd['u_idx'] == station else sd['u_idx']
                u[k], v[k] = station, line_node
                board[k] = True
            else:
                u[k], v[k] = sd['u_idx'], sd['v_idx']
        self.arc_u = np.concatenate((u, v)); self.arc_v = np.concatenate((v, u))
        self.arc_is_boarding_edge = board
        self.arc_flow_weights = np.concatenate((self.edge_flow_weights, self.edge_flow_weights))
        pair_map = {}
        for k in range(2 * E):
            pair_map.setdefault((int(self.arc_u[k]), int(self.arc_v[k])), []).append(k)
        ordered = sorted(pair_map)
        self.arc_pair_nodes = np.asarray(ordered, dtype=np.int64)
        self.arc_pair_indices = [np.asarray(pair_map[p], dtype=np.int64) for p in ordered]
        n = len(self.node_list)
        self.arc_pair_id = np.full((n, n), -1, dtype=np.int32)
        self.arc_pair_id[self.arc_pair_nodes[:, 0], self.arc_pair_nodes[:, 1]] = np.arange(len(ordered), dtype=np.int32)
        self.arc_pair_first = np.asarray([ix[0] for ix in self.arc_pair_indices], dtype=np.int64)
        self.arc_pair_multi = [(i, ix) for i, ix in enumerate(self.arc_pair_indices) if len(ix) > 1]
        self.arc_base_walk = np.array([
            sd['boarding_dist'] / 1.4 if sd['type'] == 'boarding' else np.nan for sd in self.fast_edge_data
        ], dtype=np.float64)
        self._arc_ready = True

    def _arc_vector(self, edge_times, walk_times):
        self._ensure_arc_topology()
        edge_times = np.asarray(edge_times, dtype=np.float64)
        backward = np.where(self.arc_is_boarding_edge, np.asarray(walk_times, dtype=np.float64), edge_times)
        return np.concatenate((edge_times, backward))

    def _baseline_walk_vector(self):
        self._ensure_arc_topology()
        return self.arc_base_walk

    def _walk_time_only(self, static_data, water_depth, line_stats=None):
        line_id = static_data['line_id']
        context = None
        if line_id and line_stats and str(line_id) in line_stats:
            line_data = line_stats[str(line_id)]
            context = {
                'is_above_ground': static_data.get('is_above_ground', 0),
                'max_rain_above': line_data['max_rain_above'],
                'max_water_total': line_data['max_water_total'],
            }
        speed_factor = self.strategy.get_speed_impact_factor(water_depth, 'boarding', line_context=context)
        current_speed = 1.4 * speed_factor
        if current_speed < 0.01:
            return float('inf')
        return static_data['boarding_dist'] / current_speed

    def _arc_pair_costs(self, arc_times):
        self._ensure_arc_topology()
        arc_times = np.asarray(arc_times, dtype=np.float64)
        tau = self._times_to_tau(arc_times)
        usable = np.isfinite(arc_times) & (arc_times > 0) & np.isfinite(tau) & (tau > 0)
        if not usable.any():
            return None
        tie_lambda = self.route_tie_rtol * float(np.min(self.arc_flow_weights[usable]))
        combined = np.where(usable, arc_times + tie_lambda * tau, np.inf)
        chosen = self.arc_pair_first.copy()
        for pair_index, arc_indices in self.arc_pair_multi:
            chosen[pair_index] = arc_indices[int(np.argmin(combined[arc_indices]))]
        pair_cost = combined[chosen]
        finite = np.isfinite(pair_cost)
        return pair_cost, np.where(finite, arc_times[chosen], np.inf), np.where(finite, tau[chosen], np.inf)

    def _arc_station_matrices(self, arc_times):
        count = len(self.station_node_indices)
        costs = self._arc_pair_costs(arc_times)
        if costs is None:
            empty = np.full((count, count), np.inf, dtype=np.float64)
            np.fill_diagonal(empty, 0.0)
            return empty, empty.copy()
        pair_cost, pair_time, pair_tau = costs
        n = len(self.node_list)
        valid = np.isfinite(pair_cost)
        pairs = self.arc_pair_nodes[valid]
        graph = sparse.csr_matrix((pair_cost[valid], (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        distances, predecessors = csgraph_dijkstra(
            graph, directed=True, indices=self.station_node_indices, return_predecessors=True
        )
        columns = np.broadcast_to(np.arange(n), predecessors.shape)
        is_root = predecessors < 0
        ancestor = np.where(is_root, columns, predecessors)
        pair_ids = self.arc_pair_id[ancestor, columns]
        safe_ids = np.where(pair_ids >= 0, pair_ids, 0)
        path_tau = np.where(is_root, 0.0, pair_tau[safe_ids])
        path_time = np.where(is_root, 0.0, pair_time[safe_ids])
        for _ in range(64):
            next_ancestor = np.take_along_axis(ancestor, ancestor, axis=1)
            if np.array_equal(next_ancestor, ancestor):
                break
            path_tau = path_tau + np.take_along_axis(path_tau, ancestor, axis=1)
            path_time = path_time + np.take_along_axis(path_time, ancestor, axis=1)
            ancestor = next_ancestor
        else:
            raise RuntimeError("[FAST_PATH_VALIDATION_FAILED] 有向前驱树累加未收敛")
        unreachable = ~np.isfinite(distances)
        path_tau[unreachable] = np.inf
        path_time[unreachable] = np.inf
        return path_tau[:, self.station_node_indices], path_time[:, self.station_node_indices]

    def _arc_networkx_reference(self, arc_times):
        """校验专用：NetworkX MultiDiGraph 独立实现。"""
        self._ensure_arc_topology()
        arc_times = np.asarray(arc_times, dtype=np.float64)
        tau = self._times_to_tau(arc_times)
        usable = np.isfinite(arc_times) & (arc_times > 0) & np.isfinite(tau) & (tau > 0)
        tie_lambda = self.route_tie_rtol * float(np.min(self.arc_flow_weights[usable])) if usable.any() else 0.0
        D = nx.MultiDiGraph()
        D.add_nodes_from(range(len(self.node_list)))
        for k in np.flatnonzero(usable):
            D.add_edge(int(self.arc_u[k]), int(self.arc_v[k]), time=float(arc_times[k]), tau=float(tau[k]),
                       cost=float(arc_times[k] + tie_lambda * tau[k]))
        def weight_of(name):
            return lambda a, b, data: min(attr[name] for attr in data.values())
        cost_w, time_w = weight_of('cost'), weight_of('time')
        stations = [int(i) for i in self.station_node_indices]
        count = len(stations)
        tau_m = np.full((count, count), np.inf); time_m = np.full((count, count), np.inf); pure_m = np.full((count, count), np.inf)
        for i, s in enumerate(stations):
            _, paths = nx.single_source_dijkstra(D, s, weight=cost_w)
            pure = nx.single_source_dijkstra_path_length(D, s, weight=time_w)
            for j, t in enumerate(stations):
                if t == s:
                    tau_m[i, j] = time_m[i, j] = pure_m[i, j] = 0.0
                    continue
                if t in pure:
                    pure_m[i, j] = pure[t]
                path = paths.get(t)
                if path is None:
                    continue
                tt = ta = 0.0
                for a, b in zip(path[:-1], path[1:]):
                    best = min(D[a][b].values(), key=lambda attr: attr['cost'])
                    ta += best['tau']; tt += best['time']
                tau_m[i, j], time_m[i, j] = ta, tt
        return tau_m, time_m, pure_m

'''

FIXED_NOISE_METHOD = r'''
    def _calculate_arf_rainfall_at(self, distance, position):
        """[20261123 修改 3] NOISE_MODE='fixed'：每个位置在一次模拟内只抽一次 N(0, 0.3) 扰动。"""
        if NOISE_MODE != 'fixed':
            return self._calculate_arf_rainfall(distance)
        area = math.pi * (distance**2)
        factor = -0.058 * math.log(area) + 0.9949 if area >= 1 else 1.0
        Q_temp = self.Q0 * factor
        cache = self.__dict__.setdefault('_fixed_noise', {})
        key = (round(float(position[0]), 6), round(float(position[1]), 6))
        noise = cache.get(key)
        if noise is None:
            noise = float(np.random.normal(0, 0.3))
            cache[key] = noise
        return max(0, Q_temp + noise * abs(Q_temp))

'''


def apply(src, flags):
    """flags: dict with keys alight, distance, noise, loop (bool) plus riding parameters."""
    header = (
        f"ALIGHT_WALK_ONLY = {bool(flags['alight'])!r}\n"
        f"RIDING_TIME_MODE = {'distance' if flags['distance'] else 'interval'!r}\n"
        f"RIDING_DETOUR = {float(flags.get('detour', 1.2))!r}\n"
        f"RIDING_SPEED_KMH = {float(flags.get('speed', 35.0))!r}\n"
        f"MAGLEV_RIDE_S = {float(flags.get('maglev_s', 440.0))!r}\n"
        f"NOISE_MODE = {'fixed' if flags['noise'] else 'minute'!r}\n"
        f"FIX_LOOP_SCHEDULE = {bool(flags['loop'])!r}\n"
    )
    src = replace_once(src, "import networkx as nx\n", header + "import networkx as nx\n")
    src = replace_once(src, THRESHOLD_EXTRA.split("# [20261123]")[0], THRESHOLD_EXTRA)

    # ---- item 3: fixed spatial noise ----
    src = replace_once(src, "        spatial_val = self._calculate_arf_rainfall(dist)\n",
                       "        spatial_val = self._calculate_arf_rainfall_at(dist, position)\n", count=2)
    src = replace_once(src, "    def _calculate_arf_rainfall(self, distance):\n",
                       FIXED_NOISE_METHOD + "    def _calculate_arf_rainfall(self, distance):\n")

    # ---- loop schedule (Line 4) ----
    src = replace_once(
        src,
        "    def _is_station_in_section(self, line_id, station_name, from_st, to_st):\n",
        "    def _is_station_in_section(self, line_id, station_name, from_st, to_st):\n"
        "        if FIX_LOOP_SCHEDULE and from_st == to_st:\n"
        "            line_cache = self.station_pos_cache.get(line_id)\n"
        "            if line_cache and from_st in line_cache:\n"
        "                return station_name in line_cache\n",
    )

    # ---- item 2: distance-based riding time ----
    src = replace_once(
        src,
        "                est['riding_pos_mid'] = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)\n",
        "                est['riding_pos_mid'] = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)\n"
        "                if RIDING_TIME_MODE == 'distance':\n"
        "                    _km = self.strategy._haversine(p1[0], p1[1], p2[0], p2[1])\n"
        "                    est['riding_base_time'] = (MAGLEV_RIDE_S if str(lid) == 'Maglev'\n"
        "                                               else RIDING_DETOUR * _km / RIDING_SPEED_KMH * 3600.0)\n",
    )
    src = replace_once(
        src,
        "            factor = self.strategy.get_speed_impact_factor(\n                water_depth, 'riding', line_context=context\n            )\n",
        "            if RIDING_TIME_MODE == 'distance':\n"
        "                base_interval = float(static_data['riding_base_time'])\n"
        "            factor = self.strategy.get_speed_impact_factor(\n                water_depth, 'riding', line_context=context\n            )\n",
    )
    src = replace_once(
        src,
        "            factor = self.strategy.get_speed_impact_factor(water_depth, 'riding', line_context=context)\n            if factor <= 0.0001:\n",
        "            if RIDING_TIME_MODE == 'distance':\n"
        "                base_interval = float(static_data['riding_base_time'])\n"
        "            factor = self.strategy.get_speed_impact_factor(water_depth, 'riding', line_context=context)\n            if factor <= 0.0001:\n",
    )

    # ---- item 1: directed boarding / alighting ----
    src = replace_once(src, """    def _times_to_tau(self, edge_times):
        edge_times = np.asarray(edge_times, dtype=np.float64)
        tau = np.full(edge_times.shape, np.inf, dtype=np.float64)
        mask = (self.edge_flow_weights > 0) & np.isfinite(edge_times)
        tau[mask] = edge_times[mask] / self.edge_flow_weights[mask]
        return tau
""", """    def _times_to_tau(self, edge_times):
        edge_times = np.asarray(edge_times, dtype=np.float64)
        weights = self.edge_flow_weights
        if self._is_arc_vector(edge_times):
            self._ensure_arc_topology()
            weights = self.arc_flow_weights
        tau = np.full(edge_times.shape, np.inf, dtype=np.float64)
        mask = (weights > 0) & np.isfinite(edge_times)
        tau[mask] = edge_times[mask] / weights[mask]
        return tau
""")
    src = replace_once(
        src,
        "    def _two_step_station_matrices(self, edge_times):\n        \"\"\"返回 (τ 矩阵, 时间矩阵)，行列为车站层节点；不可达为 inf，对角线为 0。\"\"\"\n",
        "    def _two_step_station_matrices(self, edge_times):\n        \"\"\"返回 (τ 矩阵, 时间矩阵)，行列为车站层节点；不可达为 inf，对角线为 0。\"\"\"\n"
        "        if self._is_arc_vector(edge_times):\n"
        "            return self._arc_station_matrices(edge_times)\n",
    )
    src = replace_once(
        src,
        "    def _two_step_networkx_reference(self, edge_times):\n        \"\"\"校验专用：NetworkX 独立实现。返回 (τ 矩阵, 选中路径时间矩阵, 纯时间最短路矩阵)。\"\"\"\n",
        "    def _two_step_networkx_reference(self, edge_times):\n        \"\"\"校验专用：NetworkX 独立实现。返回 (τ 矩阵, 选中路径时间矩阵, 纯时间最短路矩阵)。\"\"\"\n"
        "        if self._is_arc_vector(edge_times):\n"
        "            return self._arc_networkx_reference(edge_times)\n",
    )
    src = replace_once(src, "    def calculate_weighted_efficiency(", ARC_METHODS + "    def calculate_weighted_efficiency(")
    # step(): walking times for the alighting arcs, then convert to the 2E arc vector
    src = replace_once(
        src,
        "        rain_edge_times = np.empty(edge_count, dtype=np.float64)\n",
        "        rain_edge_times = np.empty(edge_count, dtype=np.float64)\n"
        "        rain_walk = np.full(edge_count, np.nan, dtype=np.float64)\n",
    )
    src = replace_once(
        src,
        "            rain_edge_times[edge_index] = current_time\n",
        "            rain_edge_times[edge_index] = current_time\n"
        "            if ALIGHT_WALK_ONLY and static_data['type'] == 'boarding':\n"
        "                rain_walk[edge_index] = self._walk_time_only(static_data, average_water, line_stats)\n",
    )
    src = replace_once(
        src,
        "        self.current_line_stats = line_stats\n",
        "        if ALIGHT_WALK_ONLY:\n"
        "            rain_edge_times = self._arc_vector(rain_edge_times, rain_walk)\n"
        "            if np.asarray(baseline_edge_times).shape[-1] != 2 * edge_count:\n"
        "                baseline_edge_times = self._arc_vector(baseline_edge_times, self._baseline_walk_vector())\n"
        "        self.current_line_stats = line_stats\n",
    )
    # build_shared_context(): baseline arrays carry the 2E arc vector
    src = replace_once(
        src,
        "        baseline_times = np.empty((simulation_duration, edge_count), dtype=np.float64)\n"
        "        baseline_tau = np.empty((simulation_duration, edge_count), dtype=np.float64)\n",
        "        arc_count = 2 * edge_count if ALIGHT_WALK_ONLY else edge_count\n"
        "        baseline_times = np.empty((simulation_duration, arc_count), dtype=np.float64)\n"
        "        baseline_tau = np.empty((simulation_duration, arc_count), dtype=np.float64)\n",
    )
    src = replace_once(
        src,
        "            baseline_tau[time_step] = self._times_to_tau(baseline_times[time_step])\n",
        "            if ALIGHT_WALK_ONLY:\n"
        "                baseline_times[time_step] = self._arc_vector(\n"
        "                    baseline_times[time_step, :edge_count].copy(), self._baseline_walk_vector())\n"
        "            baseline_tau[time_step] = self._times_to_tau(baseline_times[time_step])\n",
    )
    return src
