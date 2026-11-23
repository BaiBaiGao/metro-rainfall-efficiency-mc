import random, numpy as np, sys, json
from collections import defaultdict
from pathlib import Path
GEN=Path('/home/claude/audit/out/gen'); sys.path.insert(0,str(GEN))
ns={'__name__':'m','__file__':str(GEN/'run_P100_ning.py')}
exec(compile((GEN/'run_P100_ning.py').read_text(encoding='utf-8'),'x','exec'),ns)
BASE='/home/claude/metro-rainfall-efficiency-mc/inputs'
random.seed(0); np.random.seed(0)
s0=ns['ExtremeRainfall'](BASE,max_intensity=250,P=100,rain_duration=180,speed_reduction_thresholds=(5,50))
S=ns['SubwayDIIMSystem'](BASE,s0,sim_start_time='09:00')
agg=defaultdict(lambda:[0,0])
for sd in S.fast_edge_data:
    for t in range(240):
        c=(540+t)%1440
        ok=any(r['start_min']<=c<r['end_min'] for r in sd['riding_applicable_rules'])
        k=(str(sd['line_id']),sd['type']); agg[k][0]+=0 if ok else 1; agg[k][1]+=1
res={f'{k[0]}|{k[1]}':round(v[0]/v[1],3) for k,v in sorted(agg.items()) if v[0]}
print(json.dumps(res,ensure_ascii=False))
