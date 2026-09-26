"""Online comparison diagnostics (Section 5.6, Table 12): error split, class recalls, distance of
ARF errors to the preceding transition, and run-level (block) bootstrap intervals for Kappa-Temporal.
Recomputed from results/TONIOT_online_learners_perrow_predictions.csv; seed 42, 2,000 resamples.
No dataset access required.
Outputs: TONIOT_online_error_split.csv, TONIOT_online_kappaT_bootstrap.csv"""
import os, sys, numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else 'results'
df = pd.read_csv(os.path.join(R, 'TONIOT_online_learners_perrow_predictions.csv'))
y = df['y_true'].to_numpy()
prev = df['persistence W=1'].to_numpy()   # previous true label
bm = y != prev; wi = ~bm
methods = [c for c in df.columns if c != 'y_true']
rows=[]
for m in methods:
    p = df[m].to_numpy(); err = p != y
    r = dict(method=m, acc=100*(1-err.mean()), errors=int(err.sum()),
             within_err=int((err&wi).sum()), boundary_err=int((err&bm).sum()),
             share_within=100*(err&wi).sum()/max(err.sum(),1),
             recall_attack=100*((p==1)&(y==1)).sum()/(y==1).sum(),
             recall_normal=100*((p==0)&(y==0)).sum()/(y==0).sum(),
             bdry_acc=100*(p[bm]==y[bm]).mean(), within_acc=100*(p[wi]==y[wi]).mean())
    rows.append(r)
res = pd.DataFrame(rows); print(res.round(3).to_string())
print('eval rows', len(y), 'boundary rows', bm.sum(), 'attack share', y.mean(), 'normal rows', (y==0).sum())
# persistence accuracy and kappa_T
pa = (prev==y).mean()
def kt(p): return ((p==y).mean()-pa)/(1-pa)
# run-level (block) bootstrap for kappa_T: resample whole class runs of the true label
run_id = np.concatenate([[0], np.cumsum(bm[1:])])  # new run at each boundary (first row starts run 0)
# note: row 0 of eval region may itself be a boundary; treat each boundary as start of a new run
starts = np.flatnonzero(np.r_[True, y[1:]!=y[:-1]])
nr = len(starts); ends = np.r_[starts[1:], len(y)]
print('runs in eval region', nr)
rng = np.random.default_rng(42)
B=2000
out={}
corr = {m:(df[m].to_numpy()==y).astype(np.int64) for m in methods}
pcorr = (prev==y).astype(np.int64)
# per-run sums
def runsums(v): return np.add.reduceat(v, starts)
lens = ends-starts
S = {m: runsums(corr[m]) for m in methods}; SP = runsums(pcorr)
for m in methods:
    if m=='persistence W=1': continue
    point = kt(df[m].to_numpy())
    bs=np.empty(B); bsi=np.empty(B)
    for b in range(B):
        idx = rng.integers(0,nr,nr)
        n = lens[idx].sum(); acc = S[m][idx].sum()/n; pac = SP[idx].sum()/n
        bs[b] = (acc-pac)/(1-pac) if pac<1 else np.nan
        ii = rng.integers(0,len(y),len(y))
        a2 = corr[m][ii].mean(); p2 = pcorr[ii].mean(); bsi[b]=(a2-p2)/(1-p2)
    out[m]=(point, *np.nanpercentile(bs,[2.5,97.5]), *np.percentile(bsi,[2.5,97.5]))
    print(f'{m:20s} kT={point:+.3f}  run-bootstrap 95% CI [{out[m][1]:+.3f}, {out[m][2]:+.3f}]  row-bootstrap [{out[m][3]:+.3f}, {out[m][4]:+.3f}]')
# ARF within-block errors: distance to preceding boundary
p = df['ARF (ADWIN)'].to_numpy(); err=(p!=y)&wi
bidx = np.flatnonzero(bm)
e_idx = np.flatnonzero(err)
last_b = np.searchsorted(bidx, e_idx, side='right')-1
dist = np.where(last_b>=0, e_idx - bidx[np.maximum(last_b,0)], 10**9)
print('ARF within-block errors', err.sum(), '<=10 rows after a boundary', (dist<=10).sum(), '<=100', (dist<=100).sum())
res.to_csv(os.path.join(R, 'TONIOT_online_error_split.csv'), index=False)
pd.DataFrame([(k,*v) for k,v in out.items()], columns=['method','kappaT','run_lo','run_hi','row_lo','row_hi']).to_csv(os.path.join(R, 'TONIOT_online_kappaT_bootstrap.csv'), index=False)
