# ============================================================================
# LABEL-PERSISTENCE CHARACTERISATION  (Table 4)
#
# Characterises label persistence on the full chronological label sequence of one dataset:
#   (1) consecutive-same-class rate,
#   (2) mutual information I(y_t; y_{t-1}) and uncertainty coefficient U = I / H(y_t),
#   (3) lag-1 label autocorrelation, and
#   (4) per-class run-length distributions.
# Only the timestamp and label columns are read (no features, no model).
# Set DATASET to "ton", "botiot" or "unsw".
# Outputs: persistence_characterization_<DATASET>.csv, runlength_summary_<DATASET>.csv
# ============================================================================
import time, functools, numpy as np, pandas as pd

import os
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print = functools.partial(print, flush=True)

DATASET = "ton"     # "ton" | "botiot" | "unsw"
CFG = {
 "ton":    dict(path="/kaggle/input/datasets/fadiabuzwayed/ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv",
                label="label", ts="ts",  name="TON-IoT"),
 "botiot": dict(path="/kaggle/input/datasets/ndayisabae/nf-bot-iot-v3/NF-BoT-IoT-v3.csv",
                label="Label", ts="FLOW_START_MILLISECONDS", name="NF-BoT-IoT-v3"),
 "unsw":   dict(path="/kaggle/input/datasets/ndayisabae/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv",
                label="Label", ts="FLOW_START_MILLISECONDS", name="NF-UNSW-NB15-v3"),
}

def load_labels(cfg):
    df = pd.read_csv(cfg["path"], usecols=[cfg["ts"], cfg["label"]], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_numeric(df[cfg["ts"]], errors="coerce").ffill().fillna(0).to_numpy(np.float64)
    y  = pd.to_numeric(df[cfg["label"]], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    o  = np.argsort(ts, kind="mergesort")          # stable chronological sort
    return y[o]

def run_lengths(y):
    """Return list of (class, run_length) for maximal same-class runs."""
    if len(y)==0: return np.array([]), np.array([])
    chg = np.where(np.diff(y) != 0)[0] + 1
    bounds = np.concatenate([[0], chg, [len(y)]])
    lens = np.diff(bounds)
    cls  = y[bounds[:-1]]
    return cls, lens

def mutual_information(y):
    """I(y_t; y_{t-1}) in bits, H(y_t) in bits, U = I/H, lag-1 autocorr, 2x2 transition."""
    prev, cur = y[:-1], y[1:]
    K = int(max(y.max(), 1)) + 1
    N = np.zeros((K, K), float)
    for i in range(K):
        pi = prev == i
        for j in range(K):
            N[i, j] = np.sum(pi & (cur == j))
    tot = N.sum(); P = N / tot
    pi = P.sum(1); pj = P.sum(0)                    # marginals of prev, cur
    I = 0.0
    for i in range(K):
        for j in range(K):
            if P[i, j] > 0 and pi[i] > 0 and pj[j] > 0:
                I += P[i, j] * np.log2(P[i, j] / (pi[i] * pj[j]))
    H = -np.sum([p * np.log2(p) for p in pj if p > 0])   # H(y_t)
    U = I / H if H > 0 else float("nan")            # uncertainty coefficient
    ac = np.corrcoef(prev.astype(float), cur.astype(float))[0, 1]  # lag-1 autocorr
    Prow = N / N.sum(1, keepdims=True)              # P(y_t | y_{t-1})
    return I, H, U, ac, N, Prow

def pct_rows_in_long_runs(cls, lens, thresh):
    tot = lens.sum()
    return 100.0 * lens[lens >= thresh].sum() / tot if tot else 0.0

def main():
    t0=time.time(); cfg=CFG[DATASET]
    print(f">> characterising {cfg['name']}"); y=load_labels(cfg); n=len(y)
    persist = np.mean(y[1:]==y[:-1])*100
    attack  = np.mean(y==1)*100
    print(f"rows={n:,}  attack={attack:.2f}%  consecutive-same-class={persist:.2f}%")

    I,H,U,ac,N,Prow = mutual_information(y)
    print("\n--- TRANSITION MATRIX  (rows = y_{t-1}, cols = y_t) ---")
    print("counts:        ->0            ->1")
    for i in range(N.shape[0]):
        print(f"  from {i}:  {N[i,0]:>14,.0f} {N[i,1]:>14,.0f}")
    print("P(y_t | y_{t-1}):")
    for i in range(Prow.shape[0]):
        print(f"  from {i}:  {Prow[i,0]:>8.5f}      {Prow[i,1]:>8.5f}")
    print(f"\n--- INFORMATION-THEORETIC PERSISTENCE ---")
    print(f"  I(y_t; y_(t-1))      = {I:.4f} bits")
    print(f"  H(y_t)              = {H:.4f} bits")
    print(f"  U = I/H(y_t)        = {U:.4f}   (fraction of current-label entropy fixed by previous label)")
    print(f"  lag-1 autocorr      = {ac:.4f}")

    print("\n--- RUN-LENGTH DISTRIBUTION (per class) ---")
    cls,lens = run_lengths(y)
    rows=[]
    for c in sorted(set(cls.tolist())):
        L=lens[cls==c]; name={0:"normal",1:"attack"}.get(c,str(c))
        r=dict(dataset=cfg['name'],cls=name,n_runs=len(L),mean=L.mean(),median=np.median(L),
               p90=np.percentile(L,90),p99=np.percentile(L,99),max=int(L.max()),
               pct_rows_in_runs_ge100=pct_rows_in_long_runs(cls[cls==c],L,100),
               pct_rows_in_runs_ge1000=pct_rows_in_long_runs(cls[cls==c],L,1000))
        rows.append(r)
        print(f"  {name:>6}: runs={len(L):>7,}  mean={L.mean():>9.1f}  median={np.median(L):>6.0f}"
              f"  p90={np.percentile(L,90):>7.0f}  p99={np.percentile(L,99):>8.0f}  max={int(L.max()):>8,}")
        print(f"          rows in runs >=100: {r['pct_rows_in_runs_ge100']:.1f}%  "
              f">=1000: {r['pct_rows_in_runs_ge1000']:.1f}%")

    # log-binned histogram (text) over ALL runs
    print("\n--- RUN-LENGTH HISTOGRAM (all runs, log bins) ---")
    edges=np.array([1,2,5,10,50,100,500,1000,5000,10000,50000,10**9])
    h,_=np.histogram(lens,bins=edges)
    for k in range(len(h)):
        lo,hi=edges[k],edges[k+1]-1 if edges[k+1]<10**9 else "inf"
        print(f"  [{lo:>6}..{str(hi):>6}]: {h[k]:>7,} runs")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"runlength_summary_{DATASET}.csv"),index=False)
    pd.DataFrame([dict(dataset=cfg['name'],rows=n,attack_pct=attack,persistence_pct=persist,
                       MI_bits=I,H_yt_bits=H,U_uncertainty_coef=U,lag1_autocorr=ac,
                       n_runs=len(lens),mean_run_len=lens.mean(),max_run_len=int(lens.max()))]
                 ).to_csv(os.path.join(OUT_DIR, f"persistence_characterization_{DATASET}.csv"),index=False)
    print(f"\nSaved runlength_summary_{DATASET}.csv + persistence_characterization_{DATASET}.csv "
          f"({time.time()-t0:.0f}s)")
    print(">> U close to 1 and long right-tailed runs indicate strong label persistence.")

if __name__=="__main__":
    main()
