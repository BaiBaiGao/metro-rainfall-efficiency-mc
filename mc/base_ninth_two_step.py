# ==============================================================================
# 多组降速/停运阈值配置（单位：mm/h；作用于代码中的 60 分钟滑动平均雨强）
# 每个二元组 = (开始限速阈值, 停运阈值)，必须满足 0 <= 前者 < 后者。
# 只需修改下面这个列表，后续每个降雨重现期代码单元都会自动逐组计算。
# ==============================================================================
THRESHOLD_VALUES = [5.0, 10.0, 20.0, 35.0, 50.0, 70.0, 100.0]

SPEED_REDUCTION_THRESHOLD_PAIRS = [
    (slow, stop)
    for slow in THRESHOLD_VALUES
    for stop in THRESHOLD_VALUES
    if slow < stop
]

# 不同阈值组使用相同的随机种子序列，便于做公平的敏感性比较。
RANDOM_SEED_BASE = 20260208


# ==============================================================================
# [高速版配置] 独立场景种子、固定并行度与算法校验
# ==============================================================================
RETURN_PERIOD_SEED_IDS = {2: 1, 5: 2, 10: 3, 25: 4, 50: 5, 100: 6}
SEED_PERIOD_STRIDE = 1_000_000
SEED_THRESHOLD_STRIDE = 10_000
MC_ITERATIONS_GLOBAL = 300
N_JOBS_GLOBAL = 40
USE_FAST_SHORTEST_PATH = True
VALIDATE_FAST_PATH = True
FAST_PATH_RTOL = 1e-12
FAST_PATH_ATOL = 1e-12
FAST_PATH_EFFICIENCY_ATOL = 1e-10

# ==============================================================================
# [两步选路 20261118] 路径选择与评价口径
#   "time_then_tau": 第一步按实际通行时间 t（秒）选最短时间路径；
#                    第二步在选中的路径上累加 τ = t / w 作为评价成本，
#                    E = Σ_{i≠j} 1/τ_ij / [N(N-1)]，相对效率 = E_rain / E_base。
#                    多条路径总时间相对差 ≤ ROUTE_TIE_RTOL 时视为等时，取 τ 较小者。
#   "legacy_tau"   : 第九版原口径（直接在 τ 上做 Dijkstra），仅用于对照复现。
# 结果另存到 BASE_PATH 下的 OUTPUT_SUBDIR，不覆盖第九版原结果。
# ==============================================================================
ROUTE_CHOICE_MODE = "time_then_tau"
ROUTE_TIE_RTOL = 1e-9
OUTPUT_SUBDIR = "两步选路结果_v20261118"

from checkpoint import checked_batches, Path
OUTPUT_SUBDIR='E:\\codex\\时间网络效率\\投稿结果重算_两步选路_20261118\\蒙特卡洛'
N_JOBS_GLOBAL=4
BATCH_SIZE=4
CHECKPOINT_ROOT=Path('E:\\codex\\时间网络效率\\投稿结果重算_两步选路_20261118\\蒙特卡洛\\checkpoints\\85561922d4d3e6165bb1afbd897447c65f1d7b2365aecff3984b91c5309c2d06')
import networkx as nx
import json
import math
import pandas as pd
import numpy as np
import random
import os
import re
import gc
import time
import sys
import sqlite3
import warnings
from abc import ABC, abstractmethod
from scipy import sparse
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from PIL import Image
from datetime import datetime, timedelta

# [修改 1] 引入 joblib 用于 Jupyter 环境下的并行计算
from joblib import Parallel, delayed, parallel_backend, wrap_non_picklable_objects
import multiprocessing as mp

# 禁用插值产生的可能的警告
warnings.filterwarnings("ignore")

# ==============================================================================
# [全局配置] 并行计算环境设置
# ==============================================================================
# 限制 Numpy/OpenBLAS 的底层线程数。
# 原因：在 Joblib 多进程模式下，如果每个子进程再自动开启多线程进行矩阵运算，
# 会导致 CPU 资源争抢（Oversubscription）甚至死锁，显著降低仿真效率。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ==============================================================================
# 1. 策略模式：定义扰动（降雨）行为
# ==============================================================================

# ==============================================================================
# [Loky V2] 缓存逻辑已收进类方法，不保留 Notebook 全局 _cached_* 函数。
# ==============================================================================
class DisturbanceStrategy(ABC):
    @staticmethod
    def _process_cache():
        process = mp.current_process()
        cache = getattr(process, '_rain_mc_static_cache', None)
        if cache is None:
            cache = {}
            setattr(process, '_rain_mc_static_cache', cache)
        return cache

    """
    抽象基类：定义系统受到的外部干扰策略。
    
    Methods:
        get_intensity: 获取特定时空的干扰强度。
        get_speed_impact_factor: 获取干扰对系统运行速度的影响系数。
    """
    @abstractmethod
    def get_intensity(self, time_step, position): pass
    @abstractmethod
    def get_speed_impact_factor(self, intensity, edge_type, **kwargs): pass

class ExtremeRainfall(DisturbanceStrategy):
    """
    极端降雨策略模型：模拟一个移动的暴雨中心及其对地铁站点的积水影响。
    
    包含物理过程：
    1. 芝加哥雨型（Chicago Design Storm）生成时间序列降雨。
    2. 移动风暴中心（Moving Storm Center）。
    3. 空间衰减（ARF）计算。
    4. SCS-CN 径流模型计算地表积水。
    """
    def __init__(self, base_path, max_intensity=201.9, rainfall_type='chicago', P=100, r=0.405, rain_duration=180,
                 speed_reduction_thresholds=(16.0, 35.0)):
        """
        初始化降雨模型参数。
        
        Args:
            max_intensity (float): 降雨峰值强度参数 (mm/h 或相关系数)。
            P (int): 降雨重现期 (Return Period, e.g., 100年一遇)。
            r (float): 雨峰位置系数 (Peak Position parameter)，0-1之间，0.4代表峰值出现在前40%。
            rain_duration (int): 降雨持续总时长 (分钟)。
        """
        # 设定模拟的空间范围（上海经纬度大致范围）
        self.lat_range = (30.6984, 31.5139)
        self.lon_range = (120.849, 121.966)
        
        # 初始化暴雨中心起始位置和移动速度
        self.center_loc, self.velocity = self._init_random_boundary_start()
        
        self.Q0 = max_intensity  # 降雨强度基准
        self.base_path = base_path
        self.rainfall_type = rainfall_type
        self.P_period = P        # 重现期
        self.r_peak = r          # 雨峰系数
        self.rain_duration = rain_duration 
        self.speed_slow_threshold, self.speed_stop_threshold = map(float, speed_reduction_thresholds)
        if not (0 <= self.speed_slow_threshold < self.speed_stop_threshold):
            raise ValueError(f"无效降速阈值 {speed_reduction_thresholds}：必须满足 0 <= 限速阈值 < 停运阈值")
        
        self.chicago_profile = None 
        self.img_array = None
        self._load_drainage_image() # 加载区域排水能力图
        self._distance_cache = {}   # 缓存站点到雨中心的距离，减少重复三角函数计算
        
        # 如果是芝加哥雨型，预计算随时间变化的强度曲线
        if self.rainfall_type == 'chicago': self._precompute_chicago_profile()

    def _init_random_boundary_start(self):
        """
        随机初始化降雨中心。
        逻辑：选择地图四个边界之一作为起点，目标点设为地图中心附近，以此计算移动矢量。
        """
        lat_min, lat_max = self.lat_range
        lon_min, lon_max = self.lon_range
        side = random.choice([0, 1, 2, 3]) # 0:上(北), 1:下(南), 2:左(西), 3:右(东)
        
        # 设定目标点为区域中心附近略带随机偏移
        target_lat = (lat_min + lat_max) / 2 + random.uniform(-0.1, 0.1)
        target_lon = (lon_min + lon_max) / 2 + random.uniform(-0.1, 0.1)
        
        if side == 0:   start_lat, start_lon = lat_max, random.uniform(lon_min, lon_max)
        elif side == 1: start_lat, start_lon = lat_min, random.uniform(lon_min, lon_max)
        elif side == 2: start_lat, start_lon = random.uniform(lat_min, lat_max), lon_min
        else:           start_lat, start_lon = random.uniform(lat_min, lat_max), lon_max
        
        speed_kmh = random.uniform(25, 40) # 降雨系统移动速度 (km/h)
        speed_km_min = speed_kmh / 60.0
        
        # 计算移动单位向量（转换为经纬度增量）
        delta_lat_km = (target_lat - start_lat) * 111.0
        delta_lon_km = (target_lon - start_lon) * 96.0
        dist_km = math.sqrt(delta_lat_km**2 + delta_lon_km**2) or 1
        
        v_lat_deg = ((delta_lat_km / dist_km) * speed_km_min) / 111.0
        v_lon_deg = ((delta_lon_km / dist_km) * speed_km_min) / 96.0
        return (start_lat, start_lon), (v_lat_deg, v_lon_deg)

    def move_step(self):
        """每一分钟移动一次降雨中心，并清空距离缓存（因为中心变了，所有距离需重算）"""
        curr_lat, curr_lon = self.center_loc
        v_lat, v_lon = self.velocity
        self.center_loc = (curr_lat + v_lat, curr_lon + v_lon)
        self._distance_cache = {} 

    def _precompute_chicago_profile(self):
        """
        预计算芝加哥雨型（Chicago Design Storm）的时间分布。
        公式来源：Keifer & Chu (1957)，参数 A, n, b 针对上海地区暴雨强度公式适配。
        """
        times = np.arange(0, self.rain_duration + 1, 1) 
        intensities = []
        t_peak_time = self.rain_duration * self.r_peak
        
        # A, n, b 为上海地区的降雨公式参数（需根据研究区域调整）
        A = 9.581 * (1 + 0.846 * np.log10(self.P_period))
        n = 0.656
        b = 7
        
        for t in times:
            # 峰前时刻公式
            if t <= t_peak_time:
                t_b = t_peak_time - t
                num = A * ((1 - n) * (t_b / self.r_peak) + b)
                den = math.pow((t_b / self.r_peak) + b, 1 + n)
            # 峰后时刻公式
            else:
                t_a = t - t_peak_time
                num = A * ((1 - n) * (t_a / (1 - self.r_peak)) + b)
                den = math.pow((t_a / (1 - self.r_peak)) + b, 1 + n)
            intensities.append(num / den)
        
        # 模拟雨后由于排水不畅导致的“长尾效应”（指数衰减）
        tail_duration = 120  
        if intensities:
            last_val = intensities[-1]
            for t_ex in range(1, tail_duration + 1):
                val = last_val * math.exp(-0.1 * t_ex)
                intensities.append(val if val > 0.001 else 0.0)
        
        # 存储为累计降雨量 (Cumulative Rainfall)，方便后续计算径流
        self.chicago_profile = np.cumsum(intensities)

    def _load_drainage_image(self):
        """加载排水图；缓存直接保存在当前工作进程中。"""
        path = os.path.join(self.base_path, "排水能力_简化.png")
        cache = self._process_cache()
        key = ('drainage_image', os.path.abspath(path))
        try:
            if key not in cache:
                with Image.open(path) as image:
                    image_array = np.array(image)
                image_array.setflags(write=False)
                cache[key] = image_array
            self.img_array = cache[key]
        except FileNotFoundError:
            # 与旧实现一致：仅在文件缺失时生成随机调试图，并保持随机数调用顺序。
            self.img_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    def _haversine(self, lat1, lon1, lat2, lon2):
        """计算地球表面两点间的大圆距离 (km)"""
        R = 6371.0 
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    def _lat_lon_to_pixel(self, lat, lon):
        """经纬度坐标与图像像素坐标的转换"""
        if self.img_array is None: return 0, 0
        h, w = self.img_array.shape[:2]
        lat_min, lat_max = self.lat_range
        lon_min, lon_max = self.lon_range
        px = int((lon - lon_min) / (lon_max - lon_min) * w)
        py = int((lat_max - lat) / (lat_max - lat_min) * h)
        return min(max(px, 0), w - 1), min(max(py, 0), h - 1)

    def _get_drainage_from_image(self, lat, lon):
        """从像素颜色值映射到实际排水能力 (mm/h)"""
        px, py = self._lat_lon_to_pixel(lat, lon)
        # 获取灰度值或R通道值
        r_val = self.img_array[py, px][0] if self.img_array.ndim == 3 else self.img_array[py, px]
        r_norm = r_val / 255.0
        # 简单的阈值映射规则
        if r_val == 0: return 27
        elif r_norm <= 0.9: return 36
        elif r_norm <= 0.99: return 27
        else: return 50

    def _calculate_arf_rainfall(self, distance):
        """空间衰减计算：根据距雨中心距离计算面雨量折减系数 (Area Reduction Factor, ARF)"""
        area = math.pi * (distance**2)
        # ARF 经验公式
        factor = -0.058 * math.log(area) + 0.9949 if area >= 1 else 1.0
        Q_temp = self.Q0 * factor
        # 增加空间随机扰动
        return max(0, Q_temp + np.random.normal(0, 0.3 * abs(Q_temp)))

    def get_instantaneous_intensity(self, time_step, position, precomputed_dist=None):
        """
        获取某时刻、某位置的瞬时降雨强度 (mm/h)。
        主要用于地上线路的能见度/运行风险判断。
        [修改说明] 引入60分钟滑动平均，解决芝加哥雨型瞬时峰值过高的问题。
        """
        if precomputed_dist is not None: dist = precomputed_dist
        else:
            if position in self._distance_cache: dist = self._distance_cache[position]
            else:
                dist = self._haversine(self.center_loc[0], self.center_loc[1], position[0], position[1])
                self._distance_cache[position] = dist
                
        spatial_val = self._calculate_arf_rainfall(dist)
        
        if self.rainfall_type == 'chicago':
            if time_step >= len(self.chicago_profile) - 1: return 0.0
            scale = spatial_val / self.Q0 if self.Q0 > 0 else 0
            
            # [核心修改] 使用 60分钟滑动窗口计算小时雨强
            idx_now = min(int(time_step), len(self.chicago_profile) - 1)
            window_size = 60 
            idx_prev = max(0, idx_now - window_size)
            
            # 这一时段内的总雨量 (mm)
            rain_depth_in_window = self.chicago_profile[idx_now] - self.chicago_profile[idx_prev]
            
            # 时段长度 (min)
            duration = idx_now - idx_prev
            if duration <= 0: return 0.0
            
            # 换算成小时雨强 (mm/h)
            intensity_hourly = (rain_depth_in_window / duration) * 60.0
            
            return max(0, intensity_hourly * scale)
        else:
            # 简单均匀降雨模式
            return max(0, spatial_val / (self.rain_duration / 60.0)) if self.rain_duration > 0 else 0

    def get_intensity(self, time_step, position, precomputed_dist=None):
        """
        核心计算：计算地表净积水深度 (mm)。
        使用了 SCS-CN (Soil Conservation Service Curve Number) 径流模型的简化形式。
        Logic: 积水 = 降雨径流(Runoff) - 城市管网排水(Drainage)
        """
        lat, lon = position
        if precomputed_dist is not None: dist = precomputed_dist
        else:
            if position in self._distance_cache: dist = self._distance_cache[position]
            else:
                dist = self._haversine(self.center_loc[0], self.center_loc[1], lat, lon)
                self._distance_cache[position] = dist
        spatial_val = self._calculate_arf_rainfall(dist)
        
        # 1. 计算累计降雨量 (Cumulative Rainfall)
        if self.rainfall_type == 'chicago':
            scale = spatial_val / self.Q0 if self.Q0 > 0 else 0
            idx = min(int(time_step), len(self.chicago_profile) - 1)
            cum_rain = self.chicago_profile[idx] * scale
        else:
            progress = min(time_step / float(self.rain_duration), 1.0)
            cum_rain = spatial_val * progress
            
        # 2. SCS 径流转换
        S = 15.7 # 潜在最大入渗量 (Potential Maximum Retention)，与土壤/地表类型有关
        # SCS方程：Q = (P - 0.2S)^2 / (P + 0.8S)
        runoff = (cum_rain - 0.05 * S)**2 / (cum_rain + 0.95 * S) if cum_rain > 0.05 * S else 0
        
        # 3. 扣除区域排水
        drainage = (self._get_drainage_from_image(lat, lon) / 60.0) * time_step
        return max(0, runoff - drainage) # 返回值为净积水深度

    def get_speed_impact_factor(self, water_depth, edge_type, line_context=None):
        """
        计算速度折减系数：将物理积水深度转换为对系统运行/步行速度的影响 (0.0 - 1.0)。
        1.0 表示正常速度，0.0 表示完全中断。
        """
        # [修改] 提取线路整体最大积水深度，用于判断是否触发全线停运
        # 如果 max_water_total > 450，意味着线路上某处已严重积水，该线路应整体停运
        max_water_line = 0.0
        if line_context:
            max_water_line = line_context.get('max_water_total', 0.0)

        # [修改] 核心逻辑变更：
        # 如果当前站点水深 > 450mm OR 线路任意位置水深 > 450mm (由 line_context 传入)，
        # 则视为线路瘫痪。
        is_line_failed = (water_depth > 450) or (max_water_line > 450)

        if edge_type == 'boarding': 
            # 进站/步行速度折减 (基于经验研究的步速-水深函数，如 Ishigaki et al.)
            # [修改] 若线路瘫痪（不论是本站淹水还是线路其他站淹水），进站系数设为 0.0001
            if is_line_failed:
                return 0.0001
            
            if water_depth <= 0: return 1.0
            if water_depth <= 250: return max(0.0001, 0.9 - (water_depth / 250.0) * 0.1)
            elif water_depth <= 450: return 0.0001
            else: return 0.0001 # 兜底逻辑，实际上 is_line_failed 已覆盖
            
        elif edge_type == 'riding': 
            # 乘车运行速度折减
            if line_context:
                is_above = line_context.get('is_above_ground', 0)
                max_rain = line_context.get('max_rain_above', 0.0)
                # max_water_line 已经在上面提取了
                
                if is_above == 1: # 地上线主要受瞬时强降雨（能见度）影响
                    if max_rain >= self.speed_stop_threshold: return 0.0001 # 达到停运阈值
                    elif max_rain >= self.speed_slow_threshold: return 0.5 # 达到限速阈值
                else: # 地下线主要受区间进水影响
                    # [修改] 若线路上最大积水超过450mm，列车停运，影响因子变为0.0001
                    if max_water_line > 450: return 0.0001
            return 1.0

# ==============================================================================
# [新增] 真实数据降雨策略类
# ==============================================================================
class RealDataRainfall(DisturbanceStrategy):
    """
    真实数据降雨策略：基于 SQLite 数据库中的历史/实时降雨数据进行空间插值。
    模拟 2025-07-29 等特定日期的真实降雨分布。
    """
    def __init__(self, base_path, db_name="shanghai_rain_history.db", start_datetime_str="2025-07-29 00:00",
                 speed_reduction_thresholds=(16.0, 35.0)):
        self.base_path = base_path
        self.db_path = os.path.join(base_path, db_name)
        self.start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
        self.speed_slow_threshold, self.speed_stop_threshold = map(float, speed_reduction_thresholds)
        if not (0 <= self.speed_slow_threshold < self.speed_stop_threshold):
            raise ValueError(f"无效降速阈值 {speed_reduction_thresholds}：必须满足 0 <= 限速阈值 < 停运阈值")
        self.center_loc = (31.2304, 121.4737) # 默认中心设为上海市中心，适配接口（真实数据无中心概念）
        
        # 缓存机制，避免每分钟每节点都去查库插值
        self._last_query_time_str = None
        self._cached_grid_interpolator = None # 存储 griddata 的结果或KDTree
        self._cached_df = None
        
        # 上海区域网格参数
        self.SH_LON_MIN, self.SH_LON_MAX = 120.8, 122.3
        self.SH_LAT_MIN, self.SH_LAT_MAX = 30.6, 31.9
        
        self.img_array = None
        self._load_drainage_image()

    def _load_drainage_image(self):
        """加载排水图；缓存直接保存在当前工作进程中。"""
        path = os.path.join(self.base_path, "排水能力_简化.png")
        cache = self._process_cache()
        key = ('drainage_image', os.path.abspath(path))
        try:
            if key not in cache:
                with Image.open(path) as image:
                    image_array = np.array(image)
                image_array.setflags(write=False)
                cache[key] = image_array
            self.img_array = cache[key]
        except Exception:
            self.img_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

    def _lat_lon_to_pixel(self, lat, lon):
        """复用父类的坐标转换逻辑"""
        if self.img_array is None: return 0, 0
        h, w = self.img_array.shape[:2]
        # 注意：这里需要与 ExtremeRainfall 的范围保持一致，或者使用本类定义的范围
        # 假设排水图是覆盖 self.SH_LAT_MIN/MAX 范围的
        lat_min, lat_max = 30.6984, 31.5139 # 这里沿用 ExtremeRainfall 的范围设定以匹配图片
        lon_min, lon_max = 120.849, 121.966
        px = int((lon - lon_min) / (lon_max - lon_min) * w)
        py = int((lat_max - lat) / (lat_max - lat_min) * h)
        return min(max(px, 0), w - 1), min(max(py, 0), h - 1)

    def _get_drainage_from_image(self, lat, lon):
        """复用父类的排水能力获取逻辑"""
        px, py = self._lat_lon_to_pixel(lat, lon)
        r_val = self.img_array[py, px][0] if self.img_array.ndim == 3 else self.img_array[py, px]
        r_norm = r_val / 255.0
        if r_val == 0: return 27
        elif r_norm <= 0.9: return 36
        elif r_norm <= 0.99: return 27
        else: return 50
    
    # 距离计算方法
    def _haversine(self, lat1, lon1, lat2, lon2):
        """计算地球表面两点间的大圆距离 (km)"""
        R = 6371.0 
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def move_step(self):
        """真实数据不需要移动风暴中心，此方法留空适配接口"""
        pass

    def _fetch_and_interpolate(self, current_time_str):
        """从数据库读取特定时间的数据并构建插值函数"""
        # 如果时间没变，直接返回缓存
        if current_time_str == self._last_query_time_str and self._cached_df is not None:
            return
        
        self._last_query_time_str = current_time_str
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            # 使用 LIKE 模糊匹配时间 (e.g., '2025-07-29 05:00%')
            query = """
                SELECT s.lon, s.lat, r.rain_value 
                FROM rain_data r
                JOIN stations s ON r.station_id = s.station_id
                WHERE r.datetime LIKE ?
            """
            df = pd.read_sql_query(query, conn, params=(f"{current_time_str}%",))
            
            if len(df) < 3:
                # 数据不足时，假设无雨
                self._cached_df = pd.DataFrame({'lon': [], 'lat': [], 'rain_value': []})
            else:
                self._cached_df = df
        except Exception as e:
            # print(f"DB Error: {e}") 
            self._cached_df = pd.DataFrame({'lon': [], 'lat': [], 'rain_value': []})
        finally:
            if conn: conn.close()

    def get_instantaneous_intensity(self, time_step, position):
        """获取瞬时雨强 (mm/h)"""
        # 计算当前绝对时间
        curr_dt = self.start_dt + timedelta(minutes=time_step)
        # 数据库是每小时数据 (e.g. 05:00, 06:00)，这里简单取最近的小时，或者按分钟插值
        # 为了性能，这里取整点小时 (format: YYYY-MM-DD HH)
        # 如果需要更精细，可以修改为 HH:MM 并确保数据库有该精度，或者在两个小时之间做线性插值
        # 这里采用：取当前小时的数据（阶梯状变化）
        target_time_str = curr_dt.strftime("%Y-%m-%d %H") # 匹配数据库的 '2025-07-29 05' 开头
        
        self._fetch_and_interpolate(target_time_str)
        
        if self._cached_df is None or len(self._cached_df) == 0:
            return 0.0
        
        # 使用 Nearest 或 Linear 插值获取特定位置的雨量
        # 由于每次调用 griddata 开销大，这里针对单点计算：
        # 实际上 griddata 适合生成网格。对于单点查询，计算距离最近的几个站做 IDW 更快。
        # 这里简化使用 scipy.interpolate.griddata 对单点插值
        points = self._cached_df[['lat', 'lon']].values
        values = self._cached_df['rain_value'].values
        
        # 注意：griddata 输入是 (points, values, xi)
        # points: (N, 2), xi: (M, 2)
        try:
            val = griddata(points, values, np.array([position]), method='nearest')[0]
            return max(0, val)
        except:
            return 0.0

    def get_intensity(self, time_step, position, precomputed_dist=None):
        """
        计算地表积水深度 (mm)。
        对于真实数据，我们假设数据库里存的是“小时雨量 (mm)”。
        为了计算积水，我们需要累计雨量。
        注意：这是一个简化处理。真实的积水模拟需要连续的时间积分。
        这里我们假设：积水 = 当前时刻瞬时雨强累积 - 排水。
        """
        lat, lon = position
        
        # 获取瞬时雨强 (假设数据库存储的是 mm/h 或者该小时的累计雨量)
        intensity_mm_h = self.get_instantaneous_intensity(time_step, position)
        
        # 计算当前时间步（1分钟）带来的新增雨量
        rain_inc = intensity_mm_h / 60.0 
        
        # 在这里，DisturbanceStrategy 是无状态的(stateless)关于“积水历史”的吗？
        # ExtremeRainfall 类里是用 cumulative profile 算的。
        # 如果我们没有累计曲线，我们只能近似。
        # 方案：假设过去 N 小时的雨都下下来了，扣除排水。
        # 但这非常慢。
        # 妥协方案：SubwayDIIMSystem 的 G.nodes[n]['water'] 是每一帧刷新的。
        # 如果 ExtremeRainfall 是每次重算 total accumulation，那这里也得这么做。
        # 但查询历史所有小时的数据太慢了。
        
        # *修正*：SubwayDIIMSystem.step() 里是直接赋值：
        # self.G.nodes[n]['water'] = self.strategy.get_intensity(...)
        # 所以 get_intensity 必须返回“当前积水深度”。
        
        # 对于 RealData，我们难以每次从 0时刻开始积分。
        # 建议：仅返回瞬时雨强导致的“潜在积水”，或者基于简单的“当前雨强 * 因子”来近似积水风险。
        # 或者，为了更准确，我们在 SimulationSystem 里维护积水状态，而不是在这里计算 total。
        # 但为了不改动 System 的逻辑（System 假设 Strategy 返回积水深度），
        # 我们这里做一个近似： 积水深度 ~= 最近 1 小时的雨量 * 径流系数 - 排水 * 1小时
        
        S = 15.7
        # 假设 effective rain 是过去 60 分钟的累积（即当前 hourly value）
        cum_rain = intensity_mm_h # 粗略假设
        
        runoff = (cum_rain - 0.05 * S)**2 / (cum_rain + 0.95 * S) if cum_rain > 0.05 * S else 0
        
        # 排水能力 (mm/h)
        drainage_rate = self._get_drainage_from_image(lat, lon)
        
        # 估算积水：径流 - 排水（假设排水一直在工作）
        # 这里仅仅是一个状态快照，无法完美模拟时序累积，除非重构 System
        water_depth = max(0, runoff - drainage_rate) 
        
        return water_depth

    def get_speed_impact_factor(self, water_depth, edge_type, line_context=None):
        """复用父类的速度折减逻辑"""
        # [修改] 提取线路整体最大积水深度
        max_water_line = 0.0
        if line_context:
            max_water_line = line_context.get('max_water_total', 0.0)

        # [修改] 判断是否全线停运
        is_line_failed = (water_depth > 450) or (max_water_line > 450)

        if edge_type == 'boarding': 
            # [修改] 若线路瘫痪，进站系数设为 0.0001
            if is_line_failed:
                return 0.0001
                
            if water_depth <= 0: return 1.0
            if water_depth <= 250: return max(0.01, 0.9 - (water_depth / 250.0) * 0.1)
            elif water_depth <= 450: return max(0.0001, 0.8 - ((water_depth - 250) / 200.0) * 0.7)
            else: return 0.0001 
            
        elif edge_type == 'riding': 
            if line_context:
                is_above = line_context.get('is_above_ground', 0)
                max_rain = line_context.get('max_rain_above', 0.0)
                
                if is_above == 1: 
                    if max_rain >= self.speed_stop_threshold: return 0.0001  # 达到停运阈值
                    elif max_rain >= self.speed_slow_threshold: return 0.5 
                else: 
                    if max_water_line > 450: return 0.0001 # [修改] 地下线受淹停运
            return 1.0


# ==============================================================================
# 2. 核心系统：地铁网络拓扑与 DIIM (动态进程间影响模型) 模拟
# ==============================================================================
class SubwayDIIMSystem:
    @staticmethod
    def _process_cache():
        process = mp.current_process()
        cache = getattr(process, '_rain_mc_static_cache', None)
        if cache is None:
            cache = {}
            setattr(process, '_rain_mc_static_cache', cache)
        return cache

    """
    地铁动态影响模拟系统 (Dynamic Inoperability Input-Output Model System)
    
    主要功能：
    1. 构建双层网络图（Station Layer + Line Layer）。
    2. 加载静态数据（拓扑、时刻表、客流）。
    3. 执行时间步进仿真，计算 DIIM 状态更新。
    4. 计算网络性能指标（加权全局效率）。
    """
    def __init__(self, base_path, disturbance_strategy, sim_start_time='05:30', real_flow_df=None, shared_context=None, use_fast_shortest_path=True, validate_fast_path=False,
                 route_choice_mode='time_then_tau', route_tie_rtol=1e-9):
        self.base_path = base_path
        self.strategy = disturbance_strategy
        self.real_flow_df = real_flow_df # [新增] 接收真实客流数据 DataFrame
        self.shared_context = shared_context
        self.use_fast_shortest_path = bool(use_fast_shortest_path)
        self.validate_fast_path = bool(validate_fast_path)
        # [两步选路 20261118] 选路口径：time_then_tau（按时间选路、按 τ 评价）或 legacy_tau（第九版原口径）
        if route_choice_mode not in ('time_then_tau', 'legacy_tau'):
            raise ValueError(f"未知的 route_choice_mode: {route_choice_mode}")
        self.route_choice_mode = route_choice_mode
        self.route_tie_rtol = float(route_tie_rtol)
        if shared_context is not None and shared_context.get('route_choice_mode', 'legacy_tau') != route_choice_mode:
            raise RuntimeError("共享基准上下文的选路口径与当前系统不一致")
        self._validated_time_steps = set()
        self._last_step_index = None
        self._last_rain_edge_times = None
        self._last_base_edge_times = None
        self._last_has_edge_anomaly = False
        self.G = nx.MultiGraph() # 使用多重图构建分层网络，允许两个节点间存在多种关系的边
        
        try:
            t_obj = datetime.strptime(sim_start_time, "%H:%M")
            self.sim_start_min = t_obj.hour * 60 + t_obj.minute
            self.sim_start_dt_obj = t_obj # 保存 datetime 对象用于计算
        except: 
            self.sim_start_min = 330
            self.sim_start_dt_obj = datetime(2000, 1, 1, 5, 30)
        
        # 内部缓存与索引映射
        self.id_map_long_to_short = {}
        self.line_stations_index = {}
        self.schedule_rules = [] 
        self.schedule_lookup = {}
        self.station_pos_cache = {}
        self.fast_edge_data = [] # 扁平化的边数据列表，用于极速遍历（避开NX图查询开销）
        self.node_idx_map = {}
        self.node_pos_list = []
        self.node_dists_list = []
        self.node_st_names = []
        
        # 顺序执行初始化流程
        self._load_id_mapping()           # 1. 映射线路ID (Excel -> Dict)
        self._load_json_topology()        # 2. 加载基础拓扑 (JSON)
        self._load_detailed_schedule()    # 3. 加载列车时刻/发车间隔
        self._build_station_index()       # 4. 构建索引
        self._build_graph_topology()      # 5. 构建分层网络图 (Graph Building)
        self._apply_ground_elevation_status() # 6. 标记地上/地下线
        self._load_passenger_flow_data()  # 7. 加载客流权重 (静态初始值)
        self._build_single_layer_network()# 8. 构建单层简化网 (用于辅助计算)
        self._init_diim_matrices()        # 9. 初始化动态影响矩阵 (DIIM Init)
        
        # 韧性恢复参数（设定恢复速度）
        # c(t) 为受损程度，恢复过程遵循指数衰减
        T_rec = 15.0 
        self.k_recovery = (1.0 / T_rec) * math.log(1.0 / 0.01)
        
        self._init_flow_weights()         # 初始化流权重
        self._precompute_static_simulation_data() # 预处理静态数据加速计算
        self.congestion_warning_triggered = False

    def _load_id_mapping(self):
        """从当前工作进程的缓存恢复线路 ID 映射。"""
        cache = self._process_cache()
        key = ('id_mapping', os.path.abspath(self.base_path))
        if key not in cache:
            path = os.path.join(self.base_path, "线路id与名称对应-20251210.xlsx")
            result = {}
            try:
                if os.path.exists(path):
                    dataframe = pd.read_excel(path)
                else:
                    dataframe = pd.read_csv(path + " - Sheet1.csv")
                for _, row in dataframe.iterrows():
                    if pd.isna(row['Line ID']):
                        continue
                    long_id = str(int(row['Line ID'])) if isinstance(row['Line ID'], float) else str(row['Line ID'])
                    line_name = str(row['Line Name'])
                    short_id = None
                    if "浦江" in line_name:
                        short_id = "PJ"
                    elif "磁浮" in line_name:
                        short_id = "Maglev"
                    elif "号线" in line_name:
                        match = re.search(r'(\d+)', line_name)
                        if match:
                            short_id = match.group(1)
                    if short_id:
                        result[long_id] = short_id
            except Exception:
                pass
            cache[key] = result
        self.id_map_long_to_short = dict(cache[key])

    def _load_json_topology(self):
        """从当前工作进程的缓存读取地铁拓扑 JSON。"""
        cache = self._process_cache()
        path = os.path.join(self.base_path, "subway.json")
        key = ('json', os.path.abspath(path))
        if key not in cache:
            with open(path, "r", encoding="utf-8") as file_handle:
                cache[key] = json.load(file_handle)
        self.raw_json = cache[key]

    def _load_detailed_schedule(self):
        """从当前工作进程的缓存恢复详细发车间隔规则。"""
        cache = self._process_cache()
        key = ('schedule_rules', os.path.abspath(self.base_path))
        if key not in cache:
            path = os.path.join(self.base_path, "line_schedule_detailed.xlsx")
            result = []
            try:
                dataframe = pd.read_excel(path)
                dataframe['LineID'] = dataframe['LineID'].apply(
                    lambda value: str(int(value)) if isinstance(value, float) else str(value).strip()
                )
                for _, row in dataframe.iterrows():
                    start_datetime = pd.to_datetime(str(row['Start_Time']), format='%H:%M:%S', errors='coerce')
                    end_datetime = pd.to_datetime(str(row['End_Time']), format='%H:%M:%S', errors='coerce')
                    result.append({
                        'line_id': row['LineID'], 'day_type': str(row['DayType']).strip(),
                        'start_min': start_datetime.hour * 60 + start_datetime.minute if pd.notnull(start_datetime) else 0,
                        'end_min': end_datetime.hour * 60 + end_datetime.minute if pd.notnull(end_datetime) else 1440,
                        'from_st': str(row['From_Station']).strip(),
                        'to_st': str(row['To_Station']).strip(),
                        'interval': float(row['Interval_Sec']),
                    })
            except Exception:
                pass
            cache[key] = tuple(result)
        self.schedule_rules = list(cache[key])
        self.schedule_lookup = {}
        for rule in self.schedule_rules:
            lookup_key = (rule['line_id'], rule['day_type'])
            self.schedule_lookup.setdefault(lookup_key, []).append(rule)

    def _build_station_index(self):
        """构建车站在线路中的顺序索引，用于后续判断某车站是否在区间车的覆盖范围内"""
        for line_info in self.raw_json['l']:
            short_id = self.id_map_long_to_short.get(str(line_info['ls']))
            if not short_id: continue
            st_names = [st['n'] for st in line_info['st']]
            if short_id not in self.line_stations_index: 
                self.line_stations_index[short_id] = []
                self.station_pos_cache[short_id] = {}
            for st in st_names:
                if st not in self.line_stations_index[short_id]: 
                    self.line_stations_index[short_id].append(st)
                    self.station_pos_cache[short_id][st] = len(self.line_stations_index[short_id]) - 1

    def _is_station_in_section(self, line_id, station_name, from_st, to_st):
        """逻辑判定：特定车站是否位于指定的调度区间（from_st 到 to_st）内"""
        line_cache = self.station_pos_cache.get(line_id)
        if not line_cache: return False
        idx_s, idx_f, idx_t = line_cache.get(station_name), line_cache.get(from_st), line_cache.get(to_st)
        if idx_s is None or idx_f is None or idx_t is None: return False
        return min(idx_f, idx_t) <= idx_s <= max(idx_f, idx_t)

    def _build_graph_topology(self):
        """
        构建三维分层模型 (Space-L space)：
        1. 车站层 (Station Layer): 物理车站位置。
        2. 线路层 (Line Layer): 特定线路的站台。
        3. 边 (Edges):
           - boarding: 车站层 <-> 线路层 (进出站)。
           - riding: 线路层 <-> 线路层 (乘车)。
        """
        for line_info in self.raw_json['l']:
            short_id = self.id_map_long_to_short.get(str(line_info['ls']))
            if not short_id: continue
            sts = line_info['st']
            for st in sts:
                pos = tuple(map(float, st['sl'].split(',')))[::-1] # 转为 (lat, lon)
                node_st = f"{st['n']}_车站层"
                node_ln = f"{st['n']}_线路层_{short_id}"
                
                # 添加节点
                if not self.G.has_node(node_st): self.G.add_node(node_st, layer='station', pos=pos, flow=1000)
                self.G.add_node(node_ln, layer='line', pos=pos, line_id=short_id, station_name=st['n'])
                
                # 添加进站边 (Boarding/Alighting Link)
                self.G.add_edge(node_st, node_ln, type='boarding', weight=0)
            
            # 添加运行区间边 (Riding Link)
            for i in range(len(sts) - 1):
                u, v = f"{sts[i]['n']}_线路层_{short_id}", f"{sts[i+1]['n']}_线路层_{short_id}"
                if self.G.has_node(u) and self.G.has_node(v):
                    self.G.add_edge(u, v, type='riding', weight=0, line_id=short_id)

    def _apply_ground_elevation_status(self):
        """使用当前工作进程缓存的地上区间集合标记运行边。"""
        cache = self._process_cache()
        key = ('above_ground', os.path.abspath(self.base_path))
        if key not in cache:
            file_path = os.path.join(self.base_path, "上海地铁地上线归纳.txt")
            above_set = set()
            try:
                with open(file_path, 'r', encoding='utf-8') as file_handle:
                    for line in file_handle:
                        if '：' not in line:
                            continue
                        parts = line.strip().split('：')
                        match = re.search(r'(\d+)', parts[0])
                        line_id = match.group(1) if match else ("PJ" if "浦江" in parts[0] else None)
                        if not line_id:
                            continue
                        for group in parts[1].split('，'):
                            stations = group.strip().split('-')
                            for index in range(len(stations) - 1):
                                pair = tuple(sorted((stations[index].strip(), stations[index + 1].strip())))
                                above_set.add((line_id, pair[0], pair[1]))
            except Exception:
                pass
            cache[key] = frozenset(above_set)
        above_set = cache[key]
        for u, v, edge_key, attributes in self.G.edges(keys=True, data=True):
            if attributes.get('type') == 'riding':
                line_id = attributes.get('line_id')
                station_u = self.G.nodes[u].get('station_name')
                station_v = self.G.nodes[v].get('station_name')
                pair = tuple(sorted((station_u, station_v))) if station_u and station_v else None
                attributes['is_above_ground'] = (
                    1 if pair and (line_id, pair[0], pair[1]) in above_set else 0
                )

    def _load_passenger_flow_data(self):
        """从当前工作进程缓存的初始客流设置静态客流权重。"""
        cache = self._process_cache()
        key = ('initial_passenger_flow', os.path.abspath(self.base_path))
        if key not in cache:
            file_path = os.path.join(self.base_path, "进站客流.xlsx")
            passenger_flow = {}
            sum_flow, count = 0.0, 0
            try:
                workbook = pd.ExcelFile(file_path)
                for sheet in workbook.sheet_names:
                    for _, row in workbook.parse(sheet, header=None).iterrows():
                        value = ((row[1] or 0) + (row[2] or 0)) / 2
                        key_name = f"{row[0]}_车站层"
                        passenger_flow[key_name] = passenger_flow.get(key_name, 0) + value
                        sum_flow += value
                        count += 1
            except Exception:
                pass
            cache[key] = (passenger_flow, sum_flow, count)
        passenger_flow, sum_flow, count = cache[key]
        average_flow = sum_flow / max(count, 1) or 1000
        average_degree = np.mean([
            degree for node, degree in self.G.degree()
            if self.G.nodes[node].get('layer') == 'station'
        ]) or 1
        for node, data in self.G.nodes(data=True):
            if data.get('layer') == 'station':
                data['flow'] = passenger_flow.get(
                    node, (average_flow / average_degree) * self.G.degree(node)
                )

    def _build_single_layer_network(self):
        """构建一个简化的单层站点间网络，用于辅助拓扑特征计算"""
        self.single_layer_net = nx.Graph()
        for u, v, attr in self.G.edges(data=True):
            if attr.get('type') == 'riding':
                n1 = self.G.nodes[u].get('station_name', u.split('_')[0])
                n2 = self.G.nodes[v].get('station_name', v.split('_')[0])
                if n1 != n2: self.single_layer_net.add_edge(n1, n2)

    def _init_flow_weights(self):
        """初始化边权：将节点流分配到相邻的进站边和运行边上"""
        ln_weights = {}
        # 1. 进站边权重
        for u, v, k, attr in self.G.edges(keys=True, data=True):
            if attr['type'] == 'boarding':
                st = u if self.G.nodes[u].get('layer')=='station' else v
                ln = v if st == u else u
                w = self.G.nodes[st].get('flow', 1000) / max(1, self.G.degree[st])
                attr['flow_weight'] = w
                ln_weights[ln] = w
        # 2. 运行边权重（取两端节点平均）
        for u, v, k, attr in self.G.edges(keys=True, data=True):
            if attr['type'] == 'riding':
                attr['flow_weight'] = (ln_weights.get(u, 100)+ln_weights.get(v, 100))/2

    def _precompute_static_simulation_data(self):
        """
        [极速仿真优化] 预计算所有不会随时间改变的数据结构。
        将图数据扁平化为 List/Array，避免在 step() 循环中频繁调用 NetworkX 的 lookup 方法。
        """
        nodes = list(self.G.nodes())
        self.node_list = nodes
        self.node_idx_map = {n: i for i, n in enumerate(nodes)}
        self.node_dists_list = [0.0] * len(nodes)
        for i, n in enumerate(nodes):
            self.node_pos_list.append(self.G.nodes[n]['pos'])
            self.node_st_names.append(self.G.nodes[n].get('station_name', n.split('_')[0]))
        
        target_day_type = 'Weekday'
        for u, v, k in self.edge_list:
            attr = self.G.edges[u, v, k]
            u_idx, v_idx = self.node_idx_map[u], self.node_idx_map[v]
            lid = attr.get('line_id') or self.G.nodes[u].get('line_id') or self.G.nodes[v].get('line_id')
            est = {
                'u_idx': u_idx, 'v_idx': v_idx, 'type': attr['type'], 'line_id': lid,
                'is_above_ground': attr.get('is_above_ground', 0),
                'flow_weight': attr.get('flow_weight', 1.0),
                'boarding_dist': 0.0, 'boarding_node_flow': 1000.0, 'boarding_station_idx': None,
                'riding_pos_mid': None, 'riding_applicable_rules': []
            }
            # 预绑定该边适用的调度规则（避免运行时搜索）
            if lid:
                rules = self.schedule_lookup.get((str(lid), target_day_type), [])
                st_u, st_v = self.node_st_names[u_idx], self.node_st_names[v_idx]
                check_st = st_u if attr['type'] == 'boarding' else None
                for r in rules:
                    if check_st:
                        if self._is_station_in_section(str(lid), check_st, r['from_st'], r['to_st']):
                             est['riding_applicable_rules'].append(r)
                    else:
                        if self._is_station_in_section(str(lid), st_u, r['from_st'], r['to_st']) and \
                           self._is_station_in_section(str(lid), st_v, r['from_st'], r['to_st']):
                             est['riding_applicable_rules'].append(r)
            if attr['type'] == 'boarding':
                st = self.node_st_names[u_idx]
                deg = self.single_layer_net.degree[st] if st in self.single_layer_net else 1
                est['boarding_dist'] = deg * 100.0 # 模拟车站内步行距离
                station_node = u if self.G.nodes[u]['layer'] == 'station' else v
                est['boarding_station_idx'] = self.node_idx_map[station_node]
                est['boarding_node_flow'] = self.G.nodes[station_node].get('flow', 1000)
            else:
                p1, p2 = self.G.nodes[u]['pos'], self.G.nodes[v]['pos']
                est['riding_pos_mid'] = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            self.fast_edge_data.append(est)

        self._finalize_fast_topology()

    def _finalize_fast_topology(self):
        """预计算 SciPy 稀疏最短路所需的节点和并行边分组。"""
        self.station_nodes = [n for n in self.node_list if self.G.nodes[n].get('layer') == 'station']
        self.station_node_indices = np.asarray([self.node_idx_map[n] for n in self.station_nodes], dtype=np.int32)
        self.edge_flow_weights = np.asarray([sd['flow_weight'] for sd in self.fast_edge_data], dtype=np.float64)
        pair_map = {}
        for edge_index, sd in enumerate(self.fast_edge_data):
            a, b = sorted((sd['u_idx'], sd['v_idx']))
            pair_map.setdefault((a, b), []).append(edge_index)
        ordered_pairs = sorted(pair_map)
        self.sparse_pair_nodes = np.asarray(ordered_pairs, dtype=np.int32)
        self.sparse_pair_edge_indices = [np.asarray(pair_map[pair], dtype=np.int32) for pair in ordered_pairs]
        # [两步选路 20261118] 节点对编号矩阵与并行边分组（静态拓扑，只算一次）
        node_count = len(self.node_list)
        self.pair_id_matrix = np.full((node_count, node_count), -1, dtype=np.int32)
        if len(ordered_pairs):
            pair_ids = np.arange(len(ordered_pairs), dtype=np.int32)
            self.pair_id_matrix[self.sparse_pair_nodes[:, 0], self.sparse_pair_nodes[:, 1]] = pair_ids
            self.pair_id_matrix[self.sparse_pair_nodes[:, 1], self.sparse_pair_nodes[:, 0]] = pair_ids
        self.pair_first_edge = np.asarray([indices[0] for indices in self.sparse_pair_edge_indices], dtype=np.int64)
        self.pair_multi_edges = [
            (pair_index, indices) for pair_index, indices in enumerate(self.sparse_pair_edge_indices) if len(indices) > 1
        ]

        signature = tuple(
            (sd['u_idx'], sd['v_idx'], sd['type'], str(sd['line_id']), float(sd['flow_weight']))
            for sd in self.fast_edge_data
        )
        self.edge_signature = signature
        if self.shared_context is not None and tuple(self.shared_context['edge_signature']) != signature:
            raise RuntimeError("共享基准上下文的边顺序与当前系统不一致")

    def _times_to_tau(self, edge_times):
        edge_times = np.asarray(edge_times, dtype=np.float64)
        tau = np.full(edge_times.shape, np.inf, dtype=np.float64)
        mask = (self.edge_flow_weights > 0) & np.isfinite(edge_times)
        tau[mask] = edge_times[mask] / self.edge_flow_weights[mask]
        return tau

    def _build_sparse_graph(self, tau):
        pair_weights = np.full(len(self.sparse_pair_edge_indices), np.inf, dtype=np.float64)
        for pair_index, edge_indices in enumerate(self.sparse_pair_edge_indices):
            pair_weights[pair_index] = np.min(tau[edge_indices])
        valid = np.isfinite(pair_weights) & (pair_weights > 0)
        pairs = self.sparse_pair_nodes[valid]
        values = pair_weights[valid]
        if len(values) == 0:
            return sparse.csr_matrix((len(self.node_list), len(self.node_list)), dtype=np.float64)
        rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
        cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
        data = np.concatenate((values, values))
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(self.node_list), len(self.node_list)))

    def _efficiency_scipy_from_tau(self, tau, return_station_distances=False):
        graph = self._build_sparse_graph(tau)
        distances = csgraph_dijkstra(
            graph, directed=False, indices=self.station_node_indices, return_predecessors=False
        )
        station_distances = distances[:, self.station_node_indices]
        valid = np.isfinite(station_distances) & (station_distances > 0)
        sum_inverse = np.sum(1.0 / station_distances[valid], dtype=np.float64)
        count = len(self.station_nodes)
        efficiency = sum_inverse / (count * (count - 1)) if count >= 2 else 0.0
        if return_station_distances:
            return efficiency, station_distances
        return efficiency

    # ==========================================================================
    # [两步选路 20261118] 第一步：按实际通行时间 t 选最短时间路径；
    #                     第二步：在选中的路径上累加 τ = t / w，用 τ 计算效率。
    # 实现：一次 SciPy Dijkstra，边权 c = t + λτ，λ = ROUTE_TIE_RTOL × min(w)。
    # 由于 τ_e = t_e / w_e ≤ t_e / min(w)，任意路径满足 t ≤ c ≤ t(1+ROUTE_TIE_RTOL)，
    # 因此总时间更短（相对差 > ROUTE_TIE_RTOL）的路径一定被选中；等时路径取 τ 较小者。
    # 选中路径的时间与 τ 沿前驱树用指针倍增精确累加，不使用 c 本身。
    # 可用边的判定与原 τ 图一致：t 有限且 > 0，w > 0。
    # ==========================================================================
    def _two_step_pair_costs(self, edge_times):
        edge_times = np.asarray(edge_times, dtype=np.float64)
        tau = self._times_to_tau(edge_times)
        usable = np.isfinite(edge_times) & (edge_times > 0) & np.isfinite(tau) & (tau > 0)
        if not usable.any():
            return None
        tie_lambda = self.route_tie_rtol * float(np.min(self.edge_flow_weights[usable]))
        combined = np.where(usable, edge_times + tie_lambda * tau, np.inf)
        chosen = self.pair_first_edge.copy()
        for pair_index, edge_indices in self.pair_multi_edges:
            chosen[pair_index] = edge_indices[int(np.argmin(combined[edge_indices]))]
        pair_cost = combined[chosen]
        finite = np.isfinite(pair_cost)
        pair_time = np.where(finite, edge_times[chosen], np.inf)
        pair_tau = np.where(finite, tau[chosen], np.inf)
        return pair_cost, pair_time, pair_tau

    def _two_step_station_matrices(self, edge_times):
        """返回 (τ 矩阵, 时间矩阵)，行列为车站层节点；不可达为 inf，对角线为 0。"""
        count = len(self.station_node_indices)
        costs = self._two_step_pair_costs(edge_times)
        if costs is None:
            empty = np.full((count, count), np.inf, dtype=np.float64)
            np.fill_diagonal(empty, 0.0)
            return empty, empty.copy()
        pair_cost, pair_time, pair_tau = costs
        node_count = len(self.node_list)
        valid = np.isfinite(pair_cost)
        pairs = self.sparse_pair_nodes[valid]
        values = pair_cost[valid]
        graph = sparse.csr_matrix(
            (np.concatenate((values, values)),
             (np.concatenate((pairs[:, 0], pairs[:, 1])), np.concatenate((pairs[:, 1], pairs[:, 0])))),
            shape=(node_count, node_count),
        )
        distances, predecessors = csgraph_dijkstra(
            graph, directed=False, indices=self.station_node_indices, return_predecessors=True
        )
        columns = np.broadcast_to(np.arange(node_count), predecessors.shape)
        is_root = predecessors < 0
        ancestor = np.where(is_root, columns, predecessors)
        pair_ids = self.pair_id_matrix[ancestor, columns]
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
            raise RuntimeError("[FAST_PATH_VALIDATION_FAILED] 前驱树累加未收敛")
        unreachable = ~np.isfinite(distances)
        path_tau[unreachable] = np.inf
        path_time[unreachable] = np.inf
        return path_tau[:, self.station_node_indices], path_time[:, self.station_node_indices]

    def _efficiency_two_step_from_times(self, edge_times, return_station_matrices=False):
        tau_matrix, time_matrix = self._two_step_station_matrices(edge_times)
        valid = np.isfinite(tau_matrix) & (tau_matrix > 0)
        count = len(self.station_nodes)
        sum_inverse = np.sum(1.0 / tau_matrix[valid], dtype=np.float64)
        efficiency = sum_inverse / (count * (count - 1)) if count >= 2 else 0.0
        if return_station_matrices:
            return efficiency, tau_matrix, time_matrix
        return efficiency

    def _efficiency_from_edge_times(self, edge_times):
        """按当前口径由边时间计算绝对效率。"""
        if self.route_choice_mode == 'time_then_tau':
            return self._efficiency_two_step_from_times(edge_times)
        return self._efficiency_scipy_from_tau(self._times_to_tau(edge_times))

    def _two_step_networkx_reference(self, edge_times):
        """校验专用：NetworkX 独立实现。返回 (τ 矩阵, 选中路径时间矩阵, 纯时间最短路矩阵)。"""
        edge_times = np.asarray(edge_times, dtype=np.float64)
        tau = self._times_to_tau(edge_times)
        usable = np.isfinite(edge_times) & (edge_times > 0) & np.isfinite(tau) & (tau > 0)
        tie_lambda = self.route_tie_rtol * float(np.min(self.edge_flow_weights[usable])) if usable.any() else 0.0
        for edge_index, (u, v, k) in enumerate(self.edge_list):
            attributes = self.G.edges[u, v, k]
            ok = bool(usable[edge_index])
            attributes['_ts_time'] = float(edge_times[edge_index]) if ok else None
            attributes['_ts_tau'] = float(tau[edge_index]) if ok else None
            attributes['_ts_cost'] = float(edge_times[edge_index] + tie_lambda * tau[edge_index]) if ok else None

        def weight_of(name):
            def weight(u, v, data):
                values = [attr[name] for attr in data.values() if attr.get(name) is not None]
                return min(values) if values else None
            return weight

        cost_weight, time_weight = weight_of('_ts_cost'), weight_of('_ts_time')
        count = len(self.station_nodes)
        tau_matrix = np.full((count, count), np.inf, dtype=np.float64)
        time_matrix = np.full((count, count), np.inf, dtype=np.float64)
        pure_time_matrix = np.full((count, count), np.inf, dtype=np.float64)
        for i, source in enumerate(self.station_nodes):
            _, paths = nx.single_source_dijkstra(self.G, source, weight=cost_weight)
            pure = nx.single_source_dijkstra_path_length(self.G, source, weight=time_weight)
            for j, target in enumerate(self.station_nodes):
                if target == source:
                    tau_matrix[i, j] = time_matrix[i, j] = pure_time_matrix[i, j] = 0.0
                    continue
                if target in pure:
                    pure_time_matrix[i, j] = pure[target]
                path = paths.get(target)
                if path is None:
                    continue
                total_tau = total_time = 0.0
                for a, b in zip(path[:-1], path[1:]):
                    best = min(
                        (attr for attr in self.G[a][b].values() if attr.get('_ts_cost') is not None),
                        key=lambda attr: attr['_ts_cost'],
                    )
                    total_tau += best['_ts_tau']
                    total_time += best['_ts_time']
                tau_matrix[i, j] = total_tau
                time_matrix[i, j] = total_time
        return tau_matrix, time_matrix, pure_time_matrix

    def _validate_two_step_paths(self, rain_times, base_times, relative_time):
        """校验：(1) 选中路径的时间等于独立纯时间 Dijkstra 的最短时间；(2) τ 与 NetworkX 参考一致。"""
        eff_atol = float(globals().get('FAST_PATH_EFFICIENCY_ATOL', 1e-10))
        for label, times in (('rain', rain_times), ('base', base_times)):
            fast_eff, fast_tau, fast_time = self._efficiency_two_step_from_times(times, True)
            ref_tau, ref_time, pure_time = self._two_step_networkx_reference(times)
            if not np.array_equal(np.isfinite(fast_tau), np.isfinite(ref_tau)) or \
               not np.array_equal(np.isfinite(fast_time), np.isfinite(pure_time)):
                raise RuntimeError(f"[FAST_PATH_VALIDATION_FAILED] t={relative_time}, {label} 两步选路可达性不一致")
            finite = np.isfinite(fast_tau)
            if not np.allclose(fast_time[finite], pure_time[finite], rtol=1e-8, atol=1e-9):
                difference = np.max(np.abs(fast_time[finite] - pure_time[finite]))
                raise RuntimeError(f"[FAST_PATH_VALIDATION_FAILED] t={relative_time}, {label} 选中路径不是最短时间路径，最大时间差={difference}")
            if not np.allclose(fast_tau[finite], ref_tau[finite], rtol=1e-9, atol=1e-12):
                difference = np.max(np.abs(fast_tau[finite] - ref_tau[finite]))
                raise RuntimeError(f"[FAST_PATH_VALIDATION_FAILED] t={relative_time}, {label} 选中路径 τ 与参考不一致，最大差={difference}")
            valid = finite & (ref_tau > 0)
            count = len(self.station_nodes)
            ref_eff = np.sum(1.0 / ref_tau[valid]) / (count * (count - 1)) if count >= 2 else 0.0
            if abs(fast_eff - ref_eff) > max(eff_atol, 1e-9 * abs(ref_eff)):
                raise RuntimeError(f"[FAST_PATH_VALIDATION_FAILED] t={relative_time}, {label} 两步选路效率差={abs(fast_eff - ref_eff)}")
        self._validated_time_steps.add(relative_time)
        print(f"✅ 两步选路（时间选路→τ评价）SciPy/NetworkX 校验通过: t={relative_time}")

    def _report_path_anomaly_two_step(self, current_time_min):
        """两步口径下 τ 可以低于基准（受灾后改走 τ 更小的路线），因此只检查路径时间是否反常变短。"""
        _, _, rain_time = self._efficiency_two_step_from_times(self._last_rain_edge_times, True)
        _, _, base_time = self._efficiency_two_step_from_times(self._last_base_edge_times, True)
        bad = np.isfinite(rain_time) & (rain_time < base_time - 1e-6)
        if bad.any():
            i, j = map(int, np.argwhere(bad)[0])
            print(f"\n🔥🔥🔥 [PATH TIME ANOMALY] at Time {current_time_min} 🔥🔥🔥")
            print(f"Source: {self.station_nodes[i]} -> Target: {self.station_nodes[j]}")
            print(f"Time Rain: {rain_time[i, j]:.6f} < Time Base: {base_time[i, j]:.6f}")

    def _efficiency_networkx_from_tau(self, tau, return_station_distances=False):
        for edge_index, (u, v, k) in enumerate(self.edge_list):
            self.G.edges[u, v, k]['_legacy_tau'] = float(tau[edge_index])
        count = len(self.station_nodes)
        matrix = np.full((count, count), np.inf, dtype=np.float64)
        sum_inverse = 0.0
        for source_index, source in enumerate(self.station_nodes):
            lengths = nx.single_source_dijkstra_path_length(self.G, source, weight='_legacy_tau')
            for target_index, target in enumerate(self.station_nodes):
                if target == source:
                    matrix[source_index, target_index] = 0.0
                    continue
                distance = lengths.get(target, float('inf'))
                matrix[source_index, target_index] = distance
                if distance > 0:
                    sum_inverse += 1.0 / distance
        efficiency = sum_inverse / (count * (count - 1)) if count >= 2 else 0.0
        if return_station_distances:
            return efficiency, matrix
        return efficiency

    def _validate_fast_shortest_paths(self, rain_tau, base_tau, relative_time):
        fast_rain, fast_rain_dist = self._efficiency_scipy_from_tau(rain_tau, True)
        fast_base, fast_base_dist = self._efficiency_scipy_from_tau(base_tau, True)
        legacy_rain, legacy_rain_dist = self._efficiency_networkx_from_tau(rain_tau, True)
        legacy_base, legacy_base_dist = self._efficiency_networkx_from_tau(base_tau, True)

        rtol = float(globals().get('FAST_PATH_RTOL', 1e-12))
        atol = float(globals().get('FAST_PATH_ATOL', 1e-12))
        eff_atol = float(globals().get('FAST_PATH_EFFICIENCY_ATOL', 1e-10))
        for label, fast_dist, legacy_dist in (
            ('rain', fast_rain_dist, legacy_rain_dist),
            ('base', fast_base_dist, legacy_base_dist),
        ):
            if not np.array_equal(np.isfinite(fast_dist), np.isfinite(legacy_dist)):
                raise RuntimeError(f"[FAST_PATH_VALIDATION_FAILED] t={relative_time}, {label} 可达性不一致")
            finite = np.isfinite(fast_dist) & np.isfinite(legacy_dist)
            if not np.allclose(fast_dist[finite], legacy_dist[finite], rtol=rtol, atol=atol):
                difference = np.max(np.abs(fast_dist[finite] - legacy_dist[finite]))
                raise RuntimeError(f"[FAST_PATH_VALIDATION_FAILED] t={relative_time}, {label} 最大距离差={difference}")
        if abs(fast_rain - legacy_rain) > eff_atol or abs(fast_base - legacy_base) > eff_atol:
            raise RuntimeError(
                f"[FAST_PATH_VALIDATION_FAILED] t={relative_time}, "
                f"效率差 rain={abs(fast_rain-legacy_rain)}, base={abs(fast_base-legacy_base)}"
            )
        self._validated_time_steps.add(relative_time)
        print(f"✅ SciPy/NetworkX 等价性校验通过: t={relative_time}")

    def _report_path_anomaly(self, rain_tau, base_tau, current_time_min):
        """仅在边级单调性异常时启用原 NetworkX 路径诊断。"""
        for edge_index, (u, v, k) in enumerate(self.edge_list):
            self.G.edges[u, v, k]['tau'] = float(rain_tau[edge_index])
            self.G.edges[u, v, k]['tau_baseline'] = float(base_tau[edge_index])
        for source in self.station_nodes:
            rain_lengths = nx.single_source_dijkstra_path_length(self.G, source, weight='tau')
            base_lengths = nx.single_source_dijkstra_path_length(self.G, source, weight='tau_baseline')
            for target in self.station_nodes:
                if target == source:
                    continue
                rain_distance = rain_lengths.get(target, float('inf'))
                base_distance = base_lengths.get(target, float('inf'))
                if rain_distance < base_distance - 1e-6:
                    print(f"\n🔥🔥🔥 [PATH ANOMALY DETECTED] at Time {current_time_min} 🔥🔥🔥")
                    print(f"Source: {source} -> Target: {target}")
                    print(f"Dist Rain: {rain_distance:.6f} < Dist Base: {base_distance:.6f}")
                    return

    def calculate_weighted_efficiency(self, current_time_min, day_type, line_stats=None):
        """使用 step() 已计算的边时间计算加权网络效率。"""
        relative_time = int(current_time_min - self.sim_start_min)
        if self._last_step_index != relative_time or self._last_rain_edge_times is None:
            raise RuntimeError("calculate_weighted_efficiency 必须紧接对应时间步的 step() 调用")

        if self.route_choice_mode == 'time_then_tau':
            # [两步选路 20261118] 按时间选路、按 τ 评价
            rain_efficiency = self._efficiency_two_step_from_times(self._last_rain_edge_times)
            if self.shared_context is not None:
                base_efficiency = float(self.shared_context['baseline_efficiency'][relative_time])
            else:
                base_efficiency = self._efficiency_two_step_from_times(self._last_base_edge_times)
            validation_times = {0, 60, 120, 180, 239}
            if self.validate_fast_path and relative_time in validation_times and relative_time not in self._validated_time_steps:
                self._validate_two_step_paths(self._last_rain_edge_times, self._last_base_edge_times, relative_time)
            if self._last_has_edge_anomaly:
                self._report_path_anomaly_two_step(current_time_min)
            return rain_efficiency / base_efficiency if base_efficiency > 0 else 1.0

        rain_tau = self._times_to_tau(self._last_rain_edge_times)
        base_tau = self._times_to_tau(self._last_base_edge_times)
        if self.use_fast_shortest_path:
            rain_efficiency = self._efficiency_scipy_from_tau(rain_tau)
            if self.shared_context is not None:
                base_efficiency = float(self.shared_context['baseline_efficiency'][relative_time])
            else:
                base_efficiency = self._efficiency_scipy_from_tau(base_tau)
        else:
            rain_efficiency = self._efficiency_networkx_from_tau(rain_tau)
            base_efficiency = self._efficiency_networkx_from_tau(base_tau)

        validation_times = {0, 60, 120, 180, 239}
        if self.validate_fast_path and relative_time in validation_times and relative_time not in self._validated_time_steps:
            self._validate_fast_shortest_paths(rain_tau, base_tau, relative_time)

        if self._last_has_edge_anomaly:
            self._report_path_anomaly(rain_tau, base_tau, current_time_min)

        return rain_efficiency / base_efficiency if base_efficiency > 0 else 1.0

    def _init_diim_matrices(self):
        """初始化动态进程间影响模型 (DIIM) 矩阵结构"""
        self.edge_list = list(self.G.edges(keys=True))
        self.edge_idx = {e: i for i, e in enumerate(self.edge_list)}
        n = len(self.edge_list)
        self.q_vector = np.zeros(n) # 存储各边的受损失效率 q
        self.node_to_boarding_indices = {}
        for n_st in [n for n, d in self.G.nodes(data=True) if d['layer']=='station']:
            idxs = []
            for u, v, k in self.G.edges(n_st, keys=True):
                if self.G.edges[u,v,k]['type'] == 'boarding':
                    k_t = (u,v,k) if (u,v,k) in self.edge_idx else (v,u,k)
                    if k_t in self.edge_idx: idxs.append(self.edge_idx[k_t])
            self.node_to_boarding_indices[n_st] = idxs

    @staticmethod
    def _calc_wait_time(EV, L, dwell_time=30.0):
        rho = L * 2 * EV
        rho_threshold = 0.9
        if rho < rho_threshold:
            denominator = 2 * ((1.0 / L) - 2 * EV)
            if denominator <= 1e-6:
                value = 1800.0
            else:
                value = 2.5 * EV + (dwell_time ** 2 + 3 * EV ** 2) / denominator
            return min(value, 3600.0)
        denominator_limit = 2 * ((1.0 / L) * (1 - rho_threshold))
        time_limit = 2.5 * EV + (dwell_time ** 2 + 3 * EV ** 2) / denominator_limit
        penalty_factor = (rho / rho_threshold) ** 2
        return min(time_limit * penalty_factor, 7200.0)

    @staticmethod
    def _resolve_base_interval(static_data, lookup_time):
        default = 300.0 if static_data['type'] == 'boarding' else 180.0
        for rule in static_data['riding_applicable_rules']:
            if rule['start_min'] <= lookup_time < rule['end_min']:
                return float(rule['interval'])
        return default

    def _calc_times_reference(self, static_data, water_depth, current_time_min, line_stats=None):
        """校验专用：按旧实现的逐边时刻表扫描顺序计算通行时间。"""
        lookup_time = current_time_min % 1440
        edge_type = static_data['type']
        line_id = static_data['line_id']
        context = None
        if line_id and line_stats and str(line_id) in line_stats:
            line_data = line_stats[str(line_id)]
            context = {
                'is_above_ground': static_data.get('is_above_ground', 0),
                'max_rain_above': line_data['max_rain_above'],
                'max_water_total': line_data['max_water_total'],
            }
        if edge_type == 'boarding':
            distance = static_data['boarding_dist']
            speed_factor = self.strategy.get_speed_impact_factor(
                water_depth, 'boarding', line_context=context
            )
            current_speed = 1.4 * speed_factor
            if current_speed < 0.01:
                return float('inf'), 100.0
            walk_current = distance / current_speed
            walk_normal = distance / 1.4
            wait_current, wait_normal = 0.0, 0.0
            if line_id:
                base_interval = 300.0
                for rule in static_data['riding_applicable_rules']:
                    if rule['start_min'] <= lookup_time < rule['end_min']:
                        base_interval = float(rule['interval'])
                        break
                riding_factor = self.strategy.get_speed_impact_factor(
                    water_depth, 'riding', line_context=context
                )
                current_interval = base_interval / max(riding_factor, 0.0001)
                arrival_rate = max(float(static_data['boarding_node_flow']) / 3600.0, 0.0001)
                wait_normal = self._calc_wait_time(base_interval + 30.0, arrival_rate)
                wait_current = self._calc_wait_time(current_interval + 30.0, arrival_rate)
            return walk_current + wait_current, walk_normal + wait_normal
        if edge_type == 'riding':
            base_interval = 180.0
            for rule in static_data['riding_applicable_rules']:
                if rule['start_min'] <= lookup_time < rule['end_min']:
                    base_interval = float(rule['interval'])
                    break
            factor = self.strategy.get_speed_impact_factor(
                water_depth, 'riding', line_context=context
            )
            if factor <= 0.0001:
                return float('inf'), base_interval
            return base_interval / factor, base_interval
        return 0.0, 0.0

    def _calc_times_fast(self, static_data, water_depth, current_time_min, line_stats=None,
                         base_interval_override=None, boarding_flow_override=None):
        """与原公式相同，但不构造未使用的调试字典，也不重复扫描已预计算的时刻表。"""
        lookup_time = current_time_min % 1440
        edge_type = static_data['type']
        line_id = static_data['line_id']
        context = None
        if line_id and line_stats and str(line_id) in line_stats:
            line_data = line_stats[str(line_id)]
            context = {
                'is_above_ground': static_data.get('is_above_ground', 0),
                'max_rain_above': line_data['max_rain_above'],
                'max_water_total': line_data['max_water_total'],
            }

        if edge_type == 'boarding':
            distance = static_data['boarding_dist']
            speed_factor = self.strategy.get_speed_impact_factor(water_depth, 'boarding', line_context=context)
            current_speed = 1.4 * speed_factor
            if current_speed < 0.01:
                return float('inf'), 100.0
            walk_current = distance / current_speed
            walk_normal = distance / 1.4
            wait_current, wait_normal = 0.0, 0.0
            if line_id:
                base_interval = (
                    float(base_interval_override)
                    if base_interval_override is not None
                    else self._resolve_base_interval(static_data, lookup_time)
                )
                riding_factor = self.strategy.get_speed_impact_factor(water_depth, 'riding', line_context=context)
                current_interval = base_interval / max(riding_factor, 0.0001)
                passenger_flow = (
                    float(boarding_flow_override)
                    if boarding_flow_override is not None
                    else float(static_data['boarding_node_flow'])
                )
                arrival_rate = max(passenger_flow / 3600.0, 0.0001)
                wait_normal = self._calc_wait_time(base_interval + 30.0, arrival_rate)
                wait_current = self._calc_wait_time(current_interval + 30.0, arrival_rate)
            return walk_current + wait_current, walk_normal + wait_normal

        if edge_type == 'riding':
            base_interval = (
                float(base_interval_override)
                if base_interval_override is not None
                else self._resolve_base_interval(static_data, lookup_time)
            )
            factor = self.strategy.get_speed_impact_factor(water_depth, 'riding', line_context=context)
            if factor <= 0.0001:
                return float('inf'), base_interval
            return base_interval / factor, base_interval
        return 0.0, 0.0

    def _update_dynamic_flow(self, time_step):
        if self.real_flow_df is None or not hasattr(self.strategy, 'start_dt'):
            return
        current_time = self.strategy.start_dt + timedelta(minutes=time_step)
        if current_time.minute != 0 and time_step != 0:
            return
        lookup_hour = current_time.replace(minute=0, second=0, microsecond=0)
        try:
            batch = self.real_flow_df[self.real_flow_df['time'] == lookup_hour]
            if batch.empty:
                return
            for _, row in batch.iterrows():
                node_name = f"{row['station']}_车站层"
                if self.G.has_node(node_name):
                    self.G.nodes[node_name]['flow'] = (row['inbound'] + row['outbound']) / 2
            for static_data in self.fast_edge_data:
                station_index = static_data.get('boarding_station_idx')
                if static_data['type'] == 'boarding' and station_index is not None:
                    node_name = self.node_list[station_index]
                    static_data['boarding_node_flow'] = self.G.nodes[node_name].get('flow', 1000)
        except Exception:
            pass

    def build_shared_context(self, simulation_duration, day_type='Weekday'):
        """一次性生成所有 Run 共用的客流、时刻表和正常情景数据。"""
        edge_count = len(self.fast_edge_data)
        intervals = np.empty((simulation_duration, edge_count), dtype=np.float64)
        boarding_flows = np.empty((simulation_duration, edge_count), dtype=np.float64)
        baseline_times = np.empty((simulation_duration, edge_count), dtype=np.float64)
        baseline_tau = np.empty((simulation_duration, edge_count), dtype=np.float64)
        baseline_efficiency = np.empty(simulation_duration, dtype=np.float64)

        for time_step in range(simulation_duration):
            current_time = self.sim_start_min + time_step
            self._update_dynamic_flow(time_step)
            lookup_time = current_time % 1440
            for edge_index, static_data in enumerate(self.fast_edge_data):
                interval = self._resolve_base_interval(static_data, lookup_time)
                flow = float(static_data['boarding_node_flow'])
                normal_time, _ = self._calc_times_fast(
                    static_data, 0.0, current_time, line_stats=None,
                    base_interval_override=interval, boarding_flow_override=flow,
                )
                intervals[time_step, edge_index] = interval
                boarding_flows[time_step, edge_index] = flow
                baseline_times[time_step, edge_index] = normal_time
            baseline_tau[time_step] = self._times_to_tau(baseline_times[time_step])
            baseline_efficiency[time_step] = self._efficiency_from_edge_times(baseline_times[time_step])

        for array in (intervals, boarding_flows, baseline_times, baseline_tau, baseline_efficiency):
            array.setflags(write=False)
        context = {
            'edge_signature': self.edge_signature,
            'intervals': intervals,
            'boarding_flows': boarding_flows,
            'baseline_times': baseline_times,
            'baseline_tau': baseline_tau,
            'baseline_efficiency': baseline_efficiency,
            'route_choice_mode': self.route_choice_mode,
        }
        if self.validate_fast_path:
            self._preflight_validate_shared_context(context, simulation_duration)
        return context

    def _preflight_validate_shared_context(self, context, simulation_duration):
        """正式批量前核对预计算边时间，并验证实际拓扑的两套最短路实现。"""
        sample_times = sorted({
            index for index in (0, 60, 120, 180, 239, simulation_duration - 1)
            if 0 <= index < simulation_duration
        })
        original_flows = [float(sd['boarding_node_flow']) for sd in self.fast_edge_data]
        try:
            for time_step in sample_times:
                current_time = self.sim_start_min + time_step
                for edge_index, static_data in enumerate(self.fast_edge_data):
                    if static_data['type'] == 'boarding':
                        static_data['boarding_node_flow'] = float(
                            context['boarding_flows'][time_step, edge_index]
                        )
                    reference_time, _ = self._calc_times_reference(
                        static_data, 0.0, current_time, line_stats=None
                    )
                    cached_time = float(context['baseline_times'][time_step, edge_index])
                    if not np.isclose(reference_time, cached_time, rtol=0.0, atol=1e-12, equal_nan=True):
                        raise RuntimeError(
                            f"[EDGE_TIME_VALIDATION_FAILED] t={time_step}, edge={edge_index}, "
                            f"reference={reference_time}, cached={cached_time}"
                        )

                if self.route_choice_mode == 'time_then_tau':
                    base_times = np.asarray(context['baseline_times'][time_step], dtype=np.float64)
                    test_times = base_times.copy()
                    finite_indices = np.flatnonzero(np.isfinite(test_times) & (test_times > 0))
                    if len(finite_indices):
                        factors = 1.0 + (finite_indices % 7) * 0.05
                        test_times[finite_indices] *= factors
                        test_times[finite_indices[::97]] = np.inf
                    self._validate_two_step_paths(test_times, base_times, time_step)
                    continue
                base_tau = np.asarray(context['baseline_tau'][time_step], dtype=np.float64)
                test_tau = base_tau.copy()
                finite_indices = np.flatnonzero(np.isfinite(test_tau) & (test_tau > 0))
                if len(finite_indices):
                    factors = 1.0 + (finite_indices % 7) * 0.05
                    test_tau[finite_indices] *= factors
                    test_tau[finite_indices[::97]] = np.inf
                self._validate_fast_shortest_paths(test_tau, base_tau, time_step)
        finally:
            for static_data, original_flow in zip(self.fast_edge_data, original_flows):
                static_data['boarding_node_flow'] = original_flow
        print(
            f"✅ 正式批量前校验完成：{len(sample_times)} 个代表时间点，"
            f"预计算边时间与旧公式一致，SciPy/NetworkX 一致（选路口径={self.route_choice_mode}）。"
        )

    def step(self, t_minute, day_type='Weekday'):
        """推进一个时间步；受灾边时间只计算一次并复用于 DIIM 和效率。"""
        current_clock = self.sim_start_min + t_minute
        if self.shared_context is None:
            self._update_dynamic_flow(t_minute)

        self.strategy.move_step()
        center_lat, center_lon = self.strategy.center_loc

        for index, (lat, lon) in enumerate(self.node_pos_list):
            self.node_dists_list[index] = self.strategy._haversine(center_lat, center_lon, lat, lon)
        node_waters = []
        for index, node in enumerate(self.node_list):
            water = self.strategy.get_intensity(
                t_minute, self.G.nodes[node]['pos'], precomputed_dist=self.node_dists_list[index]
            )
            self.G.nodes[node]['water'] = water
            node_waters.append(water)

        line_stats = {}
        for static_data in self.fast_edge_data:
            if static_data['type'] == 'riding' and static_data['line_id']:
                line_id = str(static_data['line_id'])
                if line_id not in line_stats:
                    line_stats[line_id] = {'max_rain_above': 0.0, 'max_water_total': 0.0}
                average_water = (node_waters[static_data['u_idx']] + node_waters[static_data['v_idx']]) / 2
                if average_water > line_stats[line_id]['max_water_total']:
                    line_stats[line_id]['max_water_total'] = average_water
                if static_data['is_above_ground']:
                    rain = self.strategy.get_instantaneous_intensity(t_minute, static_data['riding_pos_mid'])
                    if rain > line_stats[line_id]['max_rain_above']:
                        line_stats[line_id]['max_rain_above'] = rain

        edge_count = len(self.fast_edge_data)
        current_damage = np.zeros(edge_count, dtype=np.float64)
        rain_edge_times = np.empty(edge_count, dtype=np.float64)
        if self.shared_context is not None:
            intervals = self.shared_context['intervals'][t_minute]
            boarding_flows = self.shared_context['boarding_flows'][t_minute]
            baseline_edge_times = self.shared_context['baseline_times'][t_minute]
        else:
            intervals = np.empty(edge_count, dtype=np.float64)
            boarding_flows = np.empty(edge_count, dtype=np.float64)
            baseline_edge_times = np.empty(edge_count, dtype=np.float64)
            lookup_time = current_clock % 1440
            for edge_index, static_data in enumerate(self.fast_edge_data):
                intervals[edge_index] = self._resolve_base_interval(static_data, lookup_time)
                boarding_flows[edge_index] = static_data['boarding_node_flow']
                baseline_edge_times[edge_index], _ = self._calc_times_fast(
                    static_data, 0.0, current_clock, line_stats=None,
                    base_interval_override=intervals[edge_index],
                    boarding_flow_override=boarding_flows[edge_index],
                )

        anomaly_count = 0
        anomaly_messages = []
        for edge_index, static_data in enumerate(self.fast_edge_data):
            average_water = (node_waters[static_data['u_idx']] + node_waters[static_data['v_idx']]) / 2
            current_time, normal_time = self._calc_times_fast(
                static_data, average_water, current_clock, line_stats,
                base_interval_override=intervals[edge_index],
                boarding_flow_override=boarding_flows[edge_index],
            )
            rain_edge_times[edge_index] = current_time
            baseline_time = baseline_edge_times[edge_index]
            if current_time < baseline_time - 0.0001:
                anomaly_count += 1
                if len(anomaly_messages) < 3:
                    anomaly_messages.append(
                        f"   [BAD EDGE] {self.node_st_names[static_data['u_idx']]}->"
                        f"{self.node_st_names[static_data['v_idx']]} ({static_data['type']}) | "
                        f"Rain={current_time:.4f} < Base={baseline_time:.4f}"
                    )
            if current_time == float('inf'):
                value = 1.0
            elif current_time > 0:
                value = 1.0 - (normal_time / current_time)
            else:
                value = 0.0
            current_damage[edge_index] = max(0.0, min(1.0, value))

        if anomaly_count > 0:
            print(f"\n🚨 [CRITICAL SCAN] Time {t_minute}: Found {anomaly_count}/{edge_count} edges where Rain is FASTER!")
            print("\n".join(anomaly_messages))

        self.q_vector = np.clip(
            (1 - self.k_recovery) * self.q_vector + self.k_recovery * current_damage, 0, 1
        )
        affected_lines = set()
        for edge_index, q_value in enumerate(self.q_vector):
            if q_value > 0.1:
                line_id = self.fast_edge_data[edge_index]['line_id']
                if line_id:
                    affected_lines.add(line_id)
        node_risks = []
        for indices in self.node_to_boarding_indices.values():
            node_risks.append(max(self.q_vector[index] for index in indices) if indices else 0)

        self.current_line_stats = line_stats
        self._last_step_index = t_minute
        self._last_rain_edge_times = rain_edge_times
        self._last_base_edge_times = np.asarray(baseline_edge_times, dtype=np.float64)
        self._last_has_edge_anomaly = anomaly_count > 0

        average_water = np.mean(node_waters)
        max_node_risk = max(node_risks) if node_risks else 0
        average_q = np.nanmean(self.q_vector) if len(self.q_vector) > 0 else 0.0
        average_q = np.nan_to_num(average_q)
        return average_q, average_water, len(affected_lines), max_node_risk

# ==============================================================================
# 3. 蒙特卡洛数据记录器：收集并保存多次随机实验的数据
# ==============================================================================
class MonteCarloRecorder:
    def __init__(self, time_steps):
        self.times = np.asarray(time_steps)
        self.run_data = {}

    def add_run_data(self, run_id, eff, risk, water, inten, lats, lons, aff_ln, max_nr):
        self.run_data[int(run_id)] = (
            np.asarray(eff), np.asarray(risk), np.asarray(water), np.asarray(inten),
            np.asarray(lats), np.asarray(lons), np.asarray(aff_ln), np.asarray(max_nr),
        )

    def _frame(self, metric_index):
        columns = {'Time': self.times}
        for run_id in sorted(self.run_data):
            columns[f"Run_{run_id}"] = self.run_data[run_id][metric_index]
        return pd.DataFrame(columns)

    def save_to_excel(self, base_path, threshold_pair, scenario_label):
        slow_threshold, stop_threshold = threshold_pair
        threshold_label = f"slow{slow_threshold:g}_stop{stop_threshold:g}".replace('.', 'p')
        filename = os.path.join(
            base_path,
            f"MonteCarlo_Results_{scenario_label}_{threshold_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        matches = list(Path(base_path).glob(f"MonteCarlo_Results_{scenario_label}_{threshold_label}_*.xlsx"))
        if len(matches) > 1:
            raise RuntimeError(f"同一情景存在多份工作簿：{matches}")
        if matches:
            columns = pd.read_excel(matches[0], sheet_name='Efficiency', nrows=0).columns.tolist()
            expected = ['Time'] + [f'Run_{run}' for run in sorted(self.run_data)]
            if columns != expected:
                raise RuntimeError(f"已有工作簿不完整：{matches[0]}")
            print(f"复用已导出的工作簿 {matches[0]}", flush=True)
            return
        pending = str(Path(filename).with_suffix(".pending.xlsx"))
        print(f"💾 正在导出蒙特卡洛数据到: {filename}")
        export_start = time.perf_counter()
        sheets = (
            ('Efficiency', 0), ('Risk_Avg_Q', 1), ('Water_Level_Avg', 2),
            ('Intensity_Center', 3), ('Location_Lat', 4), ('Location_Lon', 5),
            ('Affected_Line_Count', 6), ('Max_Node_Risk', 7),
        )
        try:
            try:
                import xlsxwriter  # noqa: F401
                engine = 'xlsxwriter'
            except ImportError:
                engine = 'openpyxl'
            with pd.ExcelWriter(pending, engine=engine) as writer:
                for sheet_name, metric_index in sheets:
                    frame = self._frame(metric_index)
                    frame.to_excel(writer, sheet_name=sheet_name, index=False)
                    del frame
            os.replace(pending, filename)
            print(f"✅ 导出完成（engine={engine}，耗时 {time.perf_counter() - export_start:.1f}s）。")
        except Exception as exc:
            raise RuntimeError(f"导出失败: {filename}") from exc

# ==========================================
# 5. [Worker] 独立仿真任务：供多进程并行调用
# 注意：该函数必须定义在顶层，以便 Joblib 进行序列化 (Pickle)
# ==========================================
def _loky_serialization_smoke_test(worker_callable):
    """验证强制 Cloudpickle 包装后的 Worker 可在 Loky 子进程中恢复。"""
    return os.getpid(), callable(worker_callable), type(worker_callable).__name__


def simulation_worker(run_id, base_path, intensity, period, rain_dur, sim_dur, start_str,
                      use_real_data=False, real_db_name=None, real_flow_data=None,
                      speed_reduction_thresholds=(16.0, 35.0), random_seed=None,
                      shared_context=None, use_fast_shortest_path=True, validate_fast_path=False,
                      route_choice_mode='time_then_tau', route_tie_rtol=1e-9):
    try:
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        if use_real_data:
            strategy = RealDataRainfall(
                base_path, db_name=real_db_name, start_datetime_str=start_str,
                speed_reduction_thresholds=speed_reduction_thresholds
            )
        else:
            strategy = ExtremeRainfall(
                base_path, max_intensity=intensity, rainfall_type='chicago', P=period,
                rain_duration=rain_dur, speed_reduction_thresholds=speed_reduction_thresholds
            )
            base_date_str = "2025-08-08"
            full_start_str = f"{base_date_str} {start_str}"
            try:
                strategy.start_dt = datetime.strptime(full_start_str, "%Y-%m-%d %H:%M")
            except ValueError:
                pass

        system = SubwayDIIMSystem(
            base_path, strategy, sim_start_time=start_str, real_flow_df=real_flow_data,
            shared_context=shared_context, use_fast_shortest_path=use_fast_shortest_path,
            validate_fast_path=validate_fast_path,
            route_choice_mode=route_choice_mode, route_tie_rtol=route_tie_rtol,
        )

        time_object = datetime.strptime(start_str, "%Y-%m-%d %H:%M") if len(start_str) > 5 else datetime.strptime(start_str, "%H:%M")
        start_minute = time_object.hour * 60 + time_object.minute
        result_arrays = [np.empty(sim_dur, dtype=np.float64) for _ in range(8)]
        result_arrays[6] = np.empty(sim_dur, dtype=np.int32)
        reference_position = (31.2304, 121.4737)

        for time_step in range(sim_dur):
            average_q, average_water, affected_lines, max_node_risk = system.step(time_step)
            current_clock = start_minute + time_step
            efficiency = system.calculate_weighted_efficiency(
                current_clock, 'Weekday', line_stats=getattr(system, 'current_line_stats', None)
            )
            center_latitude, center_longitude = (0, 0)
            if hasattr(system.strategy, 'center_loc'):
                center_latitude, center_longitude = system.strategy.center_loc
            center_intensity = system.strategy.get_instantaneous_intensity(time_step, reference_position)

            result_arrays[0][time_step] = efficiency
            result_arrays[1][time_step] = average_q
            result_arrays[2][time_step] = average_water
            result_arrays[3][time_step] = center_intensity
            result_arrays[4][time_step] = center_latitude
            result_arrays[5][time_step] = center_longitude
            result_arrays[6][time_step] = affected_lines
            result_arrays[7][time_step] = max_node_risk

        del system
        del strategy
        gc.collect()
        return (run_id, *result_arrays)
    except Exception as exc:
        return (run_id, None, str(exc), None, None, None, None, None, None)


# Notebook 中显式强制按值序列化 Worker，避免 Windows spawn 按 __main__ 名称查找。
simulation_worker = wrap_non_picklable_objects(simulation_worker)

def build_shared_runtime_context(base_path, intensity, period, rain_duration, simulation_duration,
                                 start_string, flow_data, threshold_pair, validate_fast_path=False,
                                 use_real_data=False, real_db_name=None,
                                 route_choice_mode='time_then_tau', route_tie_rtol=1e-9):
    """在父进程中一次性构建所有 Run 共用的确定性基准上下文。"""
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    started = time.perf_counter()
    try:
        random.seed(0)
        np.random.seed(0)
        if use_real_data:
            strategy = RealDataRainfall(
                base_path, db_name=real_db_name, start_datetime_str=start_string,
                speed_reduction_thresholds=threshold_pair,
            )
        else:
            strategy = ExtremeRainfall(
                base_path, max_intensity=intensity, rainfall_type='chicago', P=period,
                rain_duration=rain_duration, speed_reduction_thresholds=threshold_pair,
            )
            strategy.start_dt = datetime.strptime(f"2025-08-08 {start_string}", "%Y-%m-%d %H:%M")
        system = SubwayDIIMSystem(
            base_path, strategy, sim_start_time=start_string, real_flow_df=flow_data,
            shared_context=None, use_fast_shortest_path=True,
            validate_fast_path=validate_fast_path,
            route_choice_mode=route_choice_mode, route_tie_rtol=route_tie_rtol,
        )
        context = system.build_shared_context(simulation_duration)
        print(f"✅ 共享基准上下文构建完成，耗时 {time.perf_counter() - started:.1f}s")
        return context
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)


def validate_all_scenario_seeds(threshold_pairs, monte_carlo_iterations):
    period_ids = globals().get('RETURN_PERIOD_SEED_IDS', {2: 1, 5: 2, 10: 3, 25: 4, 50: 5, 100: 6})
    base = int(globals().get('RANDOM_SEED_BASE', 20260208))
    period_stride = int(globals().get('SEED_PERIOD_STRIDE', 1_000_000))
    threshold_stride = int(globals().get('SEED_THRESHOLD_STRIDE', 10_000))
    if monte_carlo_iterations >= threshold_stride:
        raise ValueError("MC_ITERATIONS 必须小于 SEED_THRESHOLD_STRIDE")
    if len(threshold_pairs) * threshold_stride + monte_carlo_iterations >= period_stride:
        raise ValueError("种子步长不足以隔离全部阈值")
    seeds = []
    for period_id in period_ids.values():
        for threshold_index in range(1, len(threshold_pairs) + 1):
            scenario_base = base + period_id * period_stride + threshold_index * threshold_stride
            seeds.extend(scenario_base + run_index for run_index in range(1, monte_carlo_iterations + 1))
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("不同重现期、阈值或 Run 出现随机种子冲突")
    if min(seeds) < 0 or max(seeds) > np.iinfo(np.uint32).max:
        raise RuntimeError("随机种子超出 NumPy RandomState 支持范围")
    print(f"✅ 随机种子唯一性校验通过：{len(seeds):,} 个任务，范围 {min(seeds)}–{max(seeds)}")

# [新增] 数据一致性校验函数
def verify_data_consistency(base_path, flow_df):
    """
    检查 Excel 客流数据、JSON 拓扑数据、DB 降雨数据中的车站名称一致性。
    """
    print("\n🔍 正在进行数据一致性校验...")
    
    # 1. 获取 Excel 中的车站列表
    excel_stations = set(flow_df['station'].unique())
    
    # 2. 获取 JSON 中的车站列表
    json_path = os.path.join(base_path, "subway.json")
    json_stations = set()
    try:
        with open(json_path, "r", encoding="utf-8") as f: 
            data = json.load(f)
            for l in data['l']:
                for st in l['st']:
                    json_stations.add(st['n'])
    except:
        print("⚠️ 无法读取 subway.json，跳过拓扑校验。")

    # 3. 打印比对结果 (Excel vs JSON)
    if json_stations:
        missing_in_json = excel_stations - json_stations
        missing_in_excel = json_stations - excel_stations
        
        print(f"  - Excel 中有 {len(excel_stations)} 个车站")
        print(f"  - JSON 中有 {len(json_stations)} 个车站")
        
        if missing_in_json:
            print(f"⚠️ 警告: Excel 中有 {len(missing_in_json)} 个车站在 JSON 拓扑中找不到 (可能无法更新其客流):")
            print(f"    示例: {list(missing_in_json)[:5]} ...")
        else:
            print("✅ Excel 所有车站均已在 JSON 拓扑中找到。")
            
        if missing_in_excel:
            # 这通常不是问题，因为 Excel 可能只包含部分车站数据
            pass

    # 4. 获取 DB 中的车站列表 (简单检查)
    db_path = os.path.join(base_path, "shanghai_rain_history.db")
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT station_name FROM stations") # 假设表名 stations，列名 station_name
            db_stations = set([r[0] for r in cursor.fetchall()])
            conn.close()
            
            print(f"  - DB 中有 {len(db_stations)} 个气象监测站")
            # 气象站和地铁站通常不同名，这里仅做数量确认，不做集合比对
        except:
            print("⚠️ 无法读取数据库中的站点信息，跳过校验。")
    else:
        print("⚠️ 数据库文件不存在。")
    print("--------------------------------------------------\n")


# ==========================================
# 6. 主程序入口 (Joblib 并行执行版)
# ==========================================
def main():
    print("\n🧩 FAST_NOTEBOOK_BUILD = 20261118-TWO-STEP-ROUTE (基于 20260902-LOKY-V2)")
    # === 参数配置区域 ===
    BASE_PATH = 'F:\\生活、学习与工作\\【研究推进】\\【复杂系统随机动力学】\\coding filepath'
    
    # 从 notebook 最前面的配置单元读取；单独运行本单元时回退到原阈值。
    threshold_pairs = globals().get("SPEED_REDUCTION_THRESHOLD_PAIRS", [(16.0, 35.0)])
    random_seed_base = int(globals().get("RANDOM_SEED_BASE", 20260208))
    period_seed_ids = globals().get("RETURN_PERIOD_SEED_IDS", {2: 1, 5: 2, 10: 3, 25: 4, 50: 5, 100: 6})
    period_seed_stride = int(globals().get("SEED_PERIOD_STRIDE", 1_000_000))
    threshold_seed_stride = int(globals().get("SEED_THRESHOLD_STRIDE", 10_000))
    use_fast_shortest_path = bool(globals().get("USE_FAST_SHORTEST_PATH", True))
    validate_fast_path = bool(globals().get("VALIDATE_FAST_PATH", True))
    # [两步选路 20261118]
    route_choice_mode = str(globals().get("ROUTE_CHOICE_MODE", "time_then_tau"))
    route_tie_rtol = float(globals().get("ROUTE_TIE_RTOL", 1e-9))
    output_subdir = str(globals().get("OUTPUT_SUBDIR", "两步选路结果_v20261118"))
    print(f"🧭 选路口径 = {route_choice_mode}，等时判据相对容差 = {route_tie_rtol:g}")
    threshold_pairs = [tuple(map(float, pair)) for pair in threshold_pairs]
    if not threshold_pairs:
        raise ValueError("SPEED_REDUCTION_THRESHOLD_PAIRS 不能为空")
    if len(set(threshold_pairs)) != len(threshold_pairs):
        raise ValueError("SPEED_REDUCTION_THRESHOLD_PAIRS 中存在重复阈值组")
    for pair in threshold_pairs:
        if len(pair) != 2 or not (0 <= pair[0] < pair[1]):
            raise ValueError(f"无效降速阈值 {pair}：必须满足 0 <= 限速阈值 < 停运阈值")
    
    # --- 模式开关 ---
    USE_REAL_DATA = False   # [开关] 是否使用 2025.07.29-08.01 的真实数据
    # ----------------
    
    if USE_REAL_DATA:
        # 真实数据模式配置
        MC_ITERATIONS = int(globals().get("MC_ITERATIONS_GLOBAL", 300))
        START_TIME_STR = "2025-07-29 05:00" # 仿真开始时间
        END_TIME_STR   = "2025-08-01 23:00" # 仿真结束时间
        #START_TIME_STR = "2025-07-29 07:00" # 仿真开始时间
        #END_TIME_STR   = "2025-07-29 08:00" # 仿真结束时间
        
        # 计算总分钟数
        t_start = datetime.strptime(START_TIME_STR, "%Y-%m-%d %H:%M")
        t_end   = datetime.strptime(END_TIME_STR, "%Y-%m-%d %H:%M")
        SIM_DURATION = int((t_end - t_start).total_seconds() / 60)
        
        DB_NAME = "shanghai_rain_history.db"
        FLOW_FILE = "shanghai_metro_flow_final_complete.xlsx" # 支持 csv 或 xlsx
        
        print(f"\n🚀 启动真实数据仿真模式 ({START_TIME_STR} -> {END_TIME_STR}, Duration={SIM_DURATION}min)")
        
        # 加载客流数据 (Pre-load flow data)
        flow_path = os.path.join(BASE_PATH, FLOW_FILE)
        print(f"📂 加载客流文件: {flow_path}")
        try:
            if flow_path.endswith('.csv'):
                flow_df = pd.read_csv(flow_path)
            else:
                flow_df = pd.read_excel(flow_path)

            # 1. 基础清洗：转字符串、去空格、统一符号
            flow_df['station'] = flow_df['station'].astype(str).str.strip().str.replace('●', '·').str.replace(' ', '')

            # 2. 特殊处理：将“浦电路”拆分为4号线和6号线，客流平分
            # 筛选出原始名称为 "浦电路" 的所有行
            pudian_mask = (flow_df['station'] == '浦电路')
            
            if pudian_mask.any():
                # 提取这些行
                pudian_rows = flow_df[pudian_mask].copy()
                
                # 构造4号线数据 (取名为 JSON 中对应的 '浦电路(4号线)', 客流除以2)
                pudian_4 = pudian_rows.copy()
                pudian_4['station'] = '浦电路(4号线)'
                pudian_4['inbound'] = pudian_4['inbound'] / 2.0
                pudian_4['outbound'] = pudian_4['outbound'] / 2.0
                
                # 构造6号线数据 (取名为 JSON 中对应的 '浦电路(6号线)', 客流除以2)
                pudian_6 = pudian_rows.copy()
                pudian_6['station'] = '浦电路(6号线)'
                pudian_6['inbound'] = pudian_6['inbound'] / 2.0
                pudian_6['outbound'] = pudian_6['outbound'] / 2.0
                
                # 从原表中删除“浦电路”，并追加拆分后的两部分数据
                flow_df = flow_df[~pudian_mask] # 删除原行
                flow_df = pd.concat([flow_df, pudian_4, pudian_6], ignore_index=True)
                print(f"ℹ️ 已将 {len(pudian_rows)} 条'浦电路'数据拆分为4号线与6号线 (客流平分)。")

            # 【新增修复】 简单的站点名称清洗，解决匹配警告
            def clean_station_name(name):
                name = str(name).strip()
                # 1. 替换特殊符号 '●' 为标准符号 '·' 
                name = name.replace('●', '·').replace(' ', '')
                
                # 2. 修正已知的 Excel 录入笔误及旧站名
                fixes = {
                    # --- 笔误修正 ---
                    '淞淞发路': '淞发路',
                    '淞淞虹路': '淞虹路',
                    '淞淞滨路': '淞滨路',  # [新增] 修正笔误
                    '临洮洮路': '临洮路',  # [新增] 修正笔误
                    '颛颛桥': '颛桥',
                    '佘佘山': '佘山',
                    # --- 站名变更/映射 ---               
                    # --- 暂未开通/无拓扑车站 ---
                    '龙居路': '龙居路',    # 14号线暂缓开通，拓扑中无此站，保留原名即可
                    '严御路': '康恒路',    # [新增] 严御路已更名为康恒路(11号线) 。但是拓扑里面没它。
                }
                return fixes.get(name, name)

            flow_df['station'] = flow_df['station'].apply(clean_station_name)

            # 预处理：转换时间列，方便后续查询
            flow_df['time'] = pd.to_datetime(flow_df['time'])
            
            # 执行一致性校验
            verify_data_consistency(BASE_PATH, flow_df)
            
        except Exception as e:
            print(f"❌ 读取客流文件失败: {e}")
            return

    else:
        # 原始蒙特卡洛模式配置
        MC_ITERATIONS = int(globals().get("MC_ITERATIONS_GLOBAL", 300))
        RAIN_P_PERIOD = 2       
        RAIN_DURATION = 180       
        SIM_DURATION = 240        
        START_TIME_STR = "09:00"     
        print(f"\n🚀 启动随机蒙特卡洛仿真模式 (Runs={MC_ITERATIONS})")
        DB_NAME = None
        
        # [修改] 读取 08.08 的基准客流数据 (全量 CSV)
        flow_file_full = "shanghai_metro_flow_with_code_final_20250802-20250831_v4.csv"
        flow_path = os.path.join(BASE_PATH, flow_file_full)
        print(f"📂 加载基准客流文件 (2025.08.08): {flow_path}")
        
        try:
            # 读取 CSV
            full_df = pd.read_csv(flow_path)
            
            # 过滤 2025-08-08 的数据
            # [修改] 原代码的 format='%Y-%m-%d %H' 无法解析 ":00:00" 部分，改为自动推断
            full_df['time'] = pd.to_datetime(full_df['time']) 
            
            target_date = pd.Timestamp("2025-08-08").date()
            flow_df = full_df[full_df['time'].dt.date == target_date].copy()
            
            # 重命名列以匹配 SubwayDIIMSystem
            # csv: inbound, outbound, station_name_mapped
            # system expects: inbound, outbound, station
            flow_df.rename(columns={'station_name_mapped': 'station'}, inplace=True)
            
            # 站点名称清洗 (确保与 if 块中的逻辑一致)
            def clean_station_name(name):
                name = str(name).strip()
                name = name.replace('●', '·').replace(' ', '')
                fixes = {
                    '淞淞发路': '淞发路',
                    '淞淞虹路': '淞虹路',
                    '淞淞滨路': '淞滨路',
                    '临洮洮路': '临洮路', 
                    '颛颛桥': '颛桥',
                    '佘佘山': '佘山',
                    '严御路': '康恒路',
                }
                return fixes.get(name, name)
            flow_df['station'] = flow_df['station'].apply(clean_station_name)
            
            # 处理浦电路拆分 (简单起见，这里复用逻辑)
            pudian_mask = (flow_df['station'] == '浦电路')
            if pudian_mask.any():
                pudian_rows = flow_df[pudian_mask].copy()
                pudian_4 = pudian_rows.copy(); pudian_4['station'] = '浦电路(4号线)'; pudian_4[['inbound','outbound']] /= 2
                pudian_6 = pudian_rows.copy(); pudian_6['station'] = '浦电路(6号线)'; pudian_6[['inbound','outbound']] /= 2
                flow_df = flow_df[~pudian_mask]
                flow_df = pd.concat([flow_df, pudian_4, pudian_6], ignore_index=True)
            
            print(f"✅ 已加载 2025.08.08 客流数据，共 {len(flow_df)} 条记录")
            
        except Exception as e:
            print(f"❌ 读取客流文件失败: {e}")
            flow_df = None

    # N_JOBS 控制并行核心数
    # N_JOBS = 10 if USE_REAL_DATA else 30 # 真实数据读取DB开销大，建议减少并发数或设为1
    N_JOBS = int(globals().get("N_JOBS_GLOBAL", 40))

    # [两步选路 20261118] 结果写入独立子目录，不覆盖第九版原结果
    OUTPUT_DIR = os.path.join(BASE_PATH, output_subdir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 结果目录: {OUTPUT_DIR}")
    efficiency_diagnostics = []

    scenario_label = "RealData" if USE_REAL_DATA else f"P{RAIN_P_PERIOD:g}"
    period_for_worker = None if USE_REAL_DATA else RAIN_P_PERIOD
    rain_duration_for_worker = SIM_DURATION if USE_REAL_DATA else RAIN_DURATION

    validate_all_scenario_seeds(threshold_pairs, MC_ITERATIONS)
    if USE_REAL_DATA:
        period_seed_id = 0
    else:
        if int(RAIN_P_PERIOD) not in period_seed_ids:
            raise ValueError(f"缺少重现期 P{RAIN_P_PERIOD:g} 的种子编号")
        period_seed_id = int(period_seed_ids[int(RAIN_P_PERIOD)])

    context_started = time.perf_counter()
    shared_context = build_shared_runtime_context(
        BASE_PATH, 250, period_for_worker, rain_duration_for_worker,
        SIM_DURATION, START_TIME_STR, flow_df, threshold_pairs[0],
        validate_fast_path=validate_fast_path,
        use_real_data=USE_REAL_DATA, real_db_name=DB_NAME,
        route_choice_mode=route_choice_mode, route_tie_rtol=route_tie_rtol,
    )
    print(f"⏱️ {scenario_label} 初始化与基准预计算总耗时: {time.perf_counter() - context_started:.1f}s")

    with parallel_backend('loky', n_jobs=N_JOBS, inner_max_num_threads=1):
        with Parallel(
            n_jobs=N_JOBS, verbose=10, pre_dispatch=N_JOBS, batch_size=1,
            max_nbytes=None, mmap_mode=None,
        ) as parallel:
            smoke_results = parallel(
                delayed(_loky_serialization_smoke_test)(simulation_worker)
                for _ in range(N_JOBS)
            )
            if not all(callable_ok for _, callable_ok, _ in smoke_results):
                raise RuntimeError("Loky Worker 序列化冒烟测试失败")
            print(
                f"✅ Loky 序列化冒烟测试通过：40 个测试任务，"
                f"启动/复用 {len({pid for pid, _, _ in smoke_results})} 个工作进程；"
                f"Worker 类型={smoke_results[0][2]}。"
            )

            for threshold_index, speed_thresholds in enumerate(threshold_pairs, start=1):
                slow_threshold, stop_threshold = speed_thresholds
                print(
                    f"\n{'=' * 80}\n"
                    f"🚦 阈值情景 {threshold_index}/{len(threshold_pairs)}: "
                    f"限速 >= {slow_threshold:g} mm/h，停运 >= {stop_threshold:g} mm/h\n"
                    f"{'=' * 80}"
                )
                threshold_started = time.perf_counter()
                initialization_started = time.perf_counter()
                recorder = MonteCarloRecorder(list(range(SIM_DURATION)))
                scenario_seed_base = (
                    random_seed_base
                    + period_seed_id * period_seed_stride
                    + threshold_index * threshold_seed_stride
                )
                print(
                    f"⏱️ 阈值 {speed_thresholds} 初始化耗时: "
                    f"{time.perf_counter() - initialization_started:.3f}s"
                )

                simulation_started = time.perf_counter()
                try:
                    results = list(checked_batches(parallel, lambda run: delayed(simulation_worker)(
                            run + 1, BASE_PATH, 250,
                            period_for_worker, rain_duration_for_worker,
                            SIM_DURATION, START_TIME_STR,
                            use_real_data=USE_REAL_DATA,
                            real_db_name=DB_NAME,
                            real_flow_data=None,
                            speed_reduction_thresholds=speed_thresholds,
                            random_seed=scenario_seed_base + run + 1,
                            shared_context=shared_context,
                            use_fast_shortest_path=use_fast_shortest_path,
                            validate_fast_path=False,
                            route_choice_mode=route_choice_mode,
                            route_tie_rtol=route_tie_rtol,
                        ), CHECKPOINT_ROOT / f'{scenario_label}_{threshold_index:02d}', MC_ITERATIONS, BATCH_SIZE, SIM_DURATION))
                except Exception as exc:
                    raise RuntimeError(f"阈值 {speed_thresholds} 并行计算失败") from exc
                simulation_elapsed = time.perf_counter() - simulation_started
                print(f"🏁 阈值 {speed_thresholds} 仿真完成，计算耗时 {simulation_elapsed:.1f}s")

                aggregation_started = time.perf_counter()
                validation_failures = []
                for result in results:
                    if result[1] is None:
                        message = str(result[2])
                        print(f"❌ [Error] 阈值 {speed_thresholds} / Run {result[0]} failed: {message}")
                        if '[FAST_PATH_VALIDATION_FAILED]' in message:
                            validation_failures.append(message)
                    else:
                        recorder.add_run_data(*result)
                if validation_failures:
                    raise RuntimeError(validation_failures[0])
                print(
                    f"⏱️ 阈值 {speed_thresholds} 汇总耗时: "
                    f"{time.perf_counter() - aggregation_started:.3f}s"
                )

                print(f"💾 阈值 {speed_thresholds} 开始保存数据...")
                recorder.save_to_excel(OUTPUT_DIR, speed_thresholds, scenario_label)
                # [两步选路 20261118] 两步口径下 E_rain/E_base 理论上可以 > 1，这里如实记录，不做截断
                if recorder.run_data:
                    efficiency_matrix = np.vstack([recorder.run_data[key][0] for key in sorted(recorder.run_data)])
                    above_one = efficiency_matrix > 1.0 + 1e-12
                    efficiency_diagnostics.append({
                        'scenario': scenario_label, 'slow_mm_h': slow_threshold, 'stop_mm_h': stop_threshold,
                        'route_choice_mode': route_choice_mode, 'runs': int(efficiency_matrix.shape[0]),
                        'min_efficiency': float(np.min(efficiency_matrix)),
                        'max_efficiency': float(np.max(efficiency_matrix)),
                        'cells_above_one': int(above_one.sum()),
                        'share_cells_above_one': float(above_one.mean()),
                        'runs_with_any_above_one': int(above_one.any(axis=1).sum()),
                    })
                    if above_one.any():
                        print(f"⚠️ 阈值 {speed_thresholds}: {int(above_one.sum())} 个 (Run, 分钟) 的相对效率 > 1，"
                              f"最大 {np.max(efficiency_matrix):.6f}")
                print(f"⏱️ 阈值 {speed_thresholds} 总耗时: {time.perf_counter() - threshold_started:.1f}s")

    diagnostics_file = os.path.join(OUTPUT_DIR, f"两步选路_相对效率诊断_{scenario_label}.csv")
    pd.DataFrame(efficiency_diagnostics).to_csv(diagnostics_file, index=False, encoding='utf-8-sig')
    with open(os.path.join(OUTPUT_DIR, f"两步选路_运行配置_{scenario_label}.json"), 'w', encoding='utf-8') as handle:
        json.dump({
            'build': '20261118-TWO-STEP-ROUTE', 'route_choice_mode': route_choice_mode,
            'route_tie_rtol': route_tie_rtol, 'scenario': scenario_label,
            'mc_iterations': MC_ITERATIONS, 'random_seed_base': random_seed_base,
            'period_seed_id': period_seed_id, 'threshold_pairs': threshold_pairs,
            'sim_duration_min': SIM_DURATION, 'start_time': START_TIME_STR, 'base_path': BASE_PATH,
        }, handle, ensure_ascii=False, indent=2)
    print(f"📝 相对效率诊断已写入: {diagnostics_file}")
if __name__ == "__main__":
    main()