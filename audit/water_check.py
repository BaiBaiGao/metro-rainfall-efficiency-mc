import random, numpy as np, pandas as pd, sys, json
from pathlib import Path
GEN=Path('/home/claude/audit/out/gen'); sys.path.insert(0,str(GEN))
ns={'__name__':'m','__file__':str(GEN/'run_P100_ning.py')}
exec(compile((GEN/'run_P100_ning.py').read_text(encoding='utf-8'),'x','exec'),ns)
BASE='/home/claude/metro-rainfall-efficiency-mc/inputs'
random.seed(0); np.random.seed(0)
s0=ns['ExtremeRainfall'](BASE,max_intensity=250,P=100,rain_duration=180,speed_reduction_thresholds=(5,50))
sysm=ns['SubwayDIIMSystem'](BASE,s0,sim_start_time='09:00')
pos=[sysm.G.nodes[n]['pos'] for n in sysm.station_nodes]
out={}
for P,pid in ((2,1),(100,6)):
    mx=[];over250=0;over450=0;tot=0;cum_peak=[]
    for run in range(1,21):
        seed=20260208+pid*1_000_000+4*10_000+run
        random.seed(seed); np.random.seed(seed)
        st=ns['ExtremeRainfall'](BASE,max_intensity=250,P=P,rain_duration=180,speed_reduction_thresholds=(5,50))
        w=np.zeros((240,len(pos)))
        for t in range(240):
            st.move_step()
            for i,p in enumerate(pos):
                d=st._haversine(st.center_loc[0],st.center_loc[1],p[0],p[1])
                w[t,i]=st.get_intensity(t,p,precomputed_dist=d)
        mx.append(w.max()); over250+=(w>250).sum(); over450+=(w>450).sum(); tot+=w.size
        cum_peak.append(float(st.chicago_profile[min(239,len(st.chicago_profile)-1)]))
    out[f'P{P}']={'max_station_water_mm_per_run':[round(x,1) for x in mx],'station_minutes_gt250':int(over250),'station_minutes_gt450':int(over450),'station_minutes_total':int(tot),'center_cumulative_rain_mm_240min':round(cum_peak[0],1)}
print(json.dumps(out,indent=1))
json.dump(out,open('/home/claude/audit/out/water_check.json','w'),indent=1)
