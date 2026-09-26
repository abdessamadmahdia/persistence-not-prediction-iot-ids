# ============================================================================
# EFFECT OF DE-DUPLICATION ON LABEL PERSISTENCE  (Section 4.2)
#
# Applies the NF-v3 loader selection (timestamp-threshold tail, IPv4 address columns skipped)
# and the same drop_duplicates() as the experiments, then reports the consecutive-same-class
# rate before and after de-duplication and the number of rows removed.
# Set DATASET to "unsw" or "botiot".
# ============================================================================
import numpy as np, pandas as pd

DATASET = "unsw"     # "unsw" | "botiot"
KEEP_TAIL = 500_000  # same as the experiment's subsample
CHUNK = 1_000_000

CFG = {
 "unsw":   dict(path="/kaggle/input/datasets/ndayisabae/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv",
                name="NF-UNSW-NB15-v3"),
 "botiot": dict(path="/kaggle/input/datasets/ndayisabae/nf-bot-iot-v3/NF-BoT-IoT-v3.csv",
                name="NF-BoT-IoT-v3"),
}
TS_CANDS    = ["FLOW_START_MILLISECONDS","FLOW_END_MILLISECONDS"]
LABEL_CANDS = ["Label","label"]
SKIP_STR    = ["IPV4_SRC_ADDR","IPV4_DST_ADDR"]

def first(cands, colset):
    for c in cands:
        if c in colset: return c
    return None

def persist(y):
    return 100.0*np.mean(y[1:]==y[:-1]) if len(y)>1 else float("nan")

def run(cfg):
    path=cfg["path"]
    head=pd.read_csv(path, nrows=0); cols=[c.strip() for c in head.columns]; colset=set(cols)
    tsc=first(TS_CANDS,colset); lb=first(LABEL_CANDS,colset)
    print(f">> {cfg['name']}  ts={tsc}  label={lb}")

    # PASS 1: full-file timestamps to find the tail threshold (cheap, 1 col)
    ts_all=[]
    for ch in pd.read_csv(path, usecols=[tsc], low_memory=False, chunksize=CHUNK):
        ts_all.append(pd.to_numeric(ch[tsc.strip() if tsc.strip() in ch.columns else tsc],
                                    errors="coerce").to_numpy(np.float64))
    ts_all=np.nan_to_num(np.concatenate(ts_all), nan=0.0); n_total=ts_all.size
    thr = np.partition(ts_all, n_total-KEEP_TAIL)[n_total-KEEP_TAIL] if KEEP_TAIL<n_total else -np.inf
    print(f"   full rows={n_total:,}  tail threshold set to keep most-recent {KEEP_TAIL:,}")
    del ts_all

    # PASS 2: read the tail rows with the SAME columns the loader keeps (skip IPV4 strings)
    skip=[c for c in SKIP_STR if c in colset]
    usecols=[c for c in cols if c not in skip]
    keep=[]
    for ch in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=CHUNK):
        ch.columns=[c.strip() for c in ch.columns]
        t=pd.to_numeric(ch[tsc], errors="coerce").fillna(0.0)
        keep.append(ch[t.to_numpy()>=thr])
    df=pd.concat(keep, ignore_index=True); del keep

    # ----- BEFORE dedup (on the exact tail selection) -----
    df_sorted_before = df.copy()
    tsb=pd.to_numeric(df_sorted_before[tsc], errors="coerce").ffill().fillna(0).to_numpy(np.float64)
    ob=np.argsort(tsb, kind="mergesort")
    yb=df_sorted_before[lb].astype(int).to_numpy()[ob]
    print(f"   PRE-dedup : rows={len(yb):,}  persist%={persist(yb):.2f}  attack={yb.mean()*100:.2f}%")

    # ----- the SAME dedup the experiment uses (full-row, IPV4 skipped) -----
    df=df.drop_duplicates().reset_index(drop=True)
    y=df[lb].astype(int).to_numpy()
    ts=pd.to_numeric(df[tsc], errors="coerce").ffill().fillna(0).to_numpy(np.float64)
    o=np.argsort(ts, kind="mergesort"); y=y[o]
    print(f"   POST-dedup: rows={len(y):,}  persist%={persist(y):.2f}  attack={y.mean()*100:.2f}%  bincount={np.bincount(y)}")

    dp=persist(yb)-persist(y)
    print(f"\n   >> persistence change from de-duplication: {dp:+.2f} points")
    print(f"   >> rows removed: {len(yb)-len(y):,}")

if __name__=="__main__":
    run(CFG[DATASET])
