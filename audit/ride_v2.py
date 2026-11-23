import random, numpy as np, sys, json
from pathlib import Path
GEN=Path('/home/claude/audit/out_v2/gen'); sys.path.insert(0,str(GEN))
ns={'__name__':'m','__file__':str(GEN/'run_P100_ning.py')}
exec(compile((GEN/'run_P100_ning.py').read_text(encoding='utf-8'),'x','exec'),ns)
BASE='/home/claude/metro-rainfall-efficiency-mc/inputs'
random.seed(0); np.random.seed(0)
s0=ns['ExtremeRainfall'](BASE,max_intensity=250,P=100,rain_duration=180,speed_reduction_thresholds=(5,50))
S=ns['SubwayDIIMSystem'](BASE,s0,sim_start_time='09:00')
rt=np.array([sd['riding_base_time'] for sd in S.fast_edge_data if sd['type']=='riding'])
q=lambda a:{p:round(float(np.quantile(a,p/100)),1) for p in (0,5,25,50,75,95,100)}
print('riding_base_time_s',q(rt))
l4=[sd for sd in S.fast_edge_data if str(sd['line_id'])=='4']
print('line4 edges with rules', sum(1 for sd in l4 if sd['riding_applicable_rules']), 'of', len(l4))
