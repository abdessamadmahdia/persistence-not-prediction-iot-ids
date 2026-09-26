# ============================================================================
# NF-BoT-IoT-v3 BOUNDARY CONFUSION  (Section 5.7)
#
# Same protocol as the TON-IoT runs: 10-fold expanding-window walk-forward, fixed RandomForest
# probe, raw X | static | leaked prototype ("oracle") | shuffled | causal recency (label feedback)
# | window = 1 persistence baseline, pooled Kappa-Temporal, boundary-stratified accuracy.
# load_nfv3(): full-file persistence from [timestamp, label]; classification on the most recent
# ~500,000 rows selected by a timestamp threshold; exact duplicates removed on all columns except
# the IPv4 address columns; infinities/NaNs set to 0, zero-variance columns dropped, values
# clipped to +/-1e9. Never sub-sample randomly: that would destroy the temporal order.
# The paired Wilcoxon statistics printed at the end are descriptive only and are not used.
# Runs run_boundary_confusion(): per-class precision/recall on class-transition rows for raw X,
# the leaked prototype (B = 100) and recency W = 50.
# Output: BOTIOT_boundary_confusion.csv
# ============================================================================

import os, glob, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef)
from scipy.stats import wilcoxon

import os
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

# ============================================================================
# CHOOSE DATASET HERE
# ============================================================================
DATASET = "botiot"          # "botiot" or "unsw"

FAST_MODE = False           # True -> reduced check (150k rows, 5 folds, 150 trees); reported runs use False

DATASETS = {
    "botiot": dict(name="NF-BoT-IoT-v3",  hint="bot-iot",
                   path="/kaggle/input/datasets/ndayisabae/nf-bot-iot-v3/NF-BoT-IoT-v3.csv",
                   subsample=500_000),
    "unsw":   dict(name="NF-UNSW-NB15-v3", hint="unsw",
                   path="/kaggle/input/datasets/ndayisabae/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv",
                   subsample=500_000),
    # subsample = number of most recent rows kept (timestamp threshold); both reported runs use
    # 500,000. The full-dataset persistence statistic is always measured on all rows first.
    # If a path does not exist, the CSV is searched under /kaggle/input using `hint`.
}

# ---- protocol (matches TON-IoT run) ----
SEED            = 42
N_FOLDS         = 10
TEST_BLOCK      = 0.10
N_ORACLE_BINS   = 100
RECENCY_WINDOWS = [50, 250, 1000]
RF_N_TREES      = 300
RF_MAX_DEPTH    = 12
RF_MIN_SAMPLES  = 5

if FAST_MODE:
    N_FOLDS = 5; RF_N_TREES = 150; RECENCY_WINDOWS = [250]

rng = np.random.default_rng(SEED)

# ---- NF-v3 schema ----
TS_CANDIDATES   = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]
LABEL_BIN_CANDS = ["Label", "label", "LABEL"]
LABEL_MULTI_CANDS = ["Attack", "attack", "Attack_Type", "ATTACK"]
# Columns that must never be features (identifiers + raw absolute times):
DROP_ALWAYS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
               "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]

# ============================================================================
# LOADER  (all NF-v3 feature columns are numeric codes/counts -> treat as numeric)
# ============================================================================
def _first(cands, cols):
    for c in cands:
        if c in cols: return c
    return None

def find_csv(hint, explicit=None):
    if explicit and os.path.exists(explicit): return explicit
    cands = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    pref = [c for c in cands if hint.lower() in c.lower()]
    picked = (pref or cands)
    if not picked:
        raise FileNotFoundError(
            f"No CSV found under /kaggle/input for hint '{hint}'. "
            f"Attach the dataset or set an explicit path.")
    return picked[0]

def load_nfv3(path, keep_tail=None, chunksize=1_000_000, hard_cap=1_500_000):
    """Memory-safe loader for large NF-v3 CSVs (e.g. NF-BoT-IoT-v3 ~17M rows).
    Pass 1: read ONLY [timestamp, label] for all rows -> true full-dataset
            consecutive-same-class persistence (cheap, ~2 columns).
    Pass 2: chunk-read features, skipping the heavy IPV4 string columns, and
            keep ONLY the most-recent `keep_tail` rows by timestamp.
    keep_tail=None -> use `hard_cap` (the full 17M cannot fit in RAM)."""
    # --- resolve column names from the header only ---
    head = pd.read_csv(path, nrows=0)
    cols = [c.strip() for c in head.columns]
    colset = set(cols)
    lb  = _first(LABEL_BIN_CANDS, colset)
    tsc = _first(TS_CANDIDATES, colset)
    lm  = _first(LABEL_MULTI_CANDS, colset)
    if lb is None:  raise ValueError(f"No binary label column. Columns: {cols[:15]}")
    if tsc is None: raise ValueError("No FLOW_START/END_MILLISECONDS (likely NF-v2 -> use v3).")

    cap = keep_tail if keep_tail is not None else hard_cap

    # --- PASS 1: cheap full-file persistence on ts-sorted labels ---
    print(f"Pass 1 (persistence, [{tsc},{lb}] only): {path}")
    ts_all, y_all = [], []
    for ch in pd.read_csv(path, usecols=[tsc, lb], low_memory=False, chunksize=chunksize):
        ts_all.append(pd.to_numeric(ch[tsc], errors="coerce").to_numpy(np.float64))
        y_all.append(ch[lb].to_numpy())
    ts_all = np.concatenate(ts_all); y_all = np.concatenate(y_all).astype(int)
    ts_all = np.nan_to_num(ts_all, nan=0.0)
    o = np.argsort(ts_all, kind="mergesort")
    full_persist = float(np.mean(y_all[o][1:] == y_all[o][:-1]) * 100.0)
    n_total = len(y_all)
    print(f"  FULL dataset: rows={n_total:,}  attack={y_all.mean():.4f}  "
          f"consecutive same-class = {full_persist:.2f}%")

    # timestamp threshold that keeps the most-recent `cap` rows (genuine time-tail)
    if cap < n_total:
        thr = np.partition(ts_all, n_total - cap)[n_total - cap]
    else:
        thr = -np.inf
    del ts_all, y_all, o

    # --- PASS 2: features for the time-tail only, skipping heavy string IDs ---
    skip_strings = [c for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR") if c in colset]
    usecols = [c for c in cols if c not in skip_strings]
    print(f"Pass 2 (features, ts >= threshold; skipping {skip_strings}) ...")
    keep = []
    for ch in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=chunksize):
        ch.columns = [c.strip() for c in ch.columns]
        t = pd.to_numeric(ch[tsc], errors="coerce").fillna(0.0)
        keep.append(ch[t.to_numpy() >= thr])
    df = pd.concat(keep, ignore_index=True); del keep

    df = df.drop_duplicates().reset_index(drop=True)
    y  = df[lb].astype(int).to_numpy()
    ts = pd.to_numeric(df[tsc], errors="coerce").ffill().fillna(0).to_numpy(np.float64)

    drop = [c for c in (DROP_ALWAYS + [lb] + ([lm] if lm else [])) if c in df.columns]
    X_df = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_df = X_df.loc[:, ~X_df.columns.duplicated()]
    # drop zero-variance (constant-in-tail) columns and clip float32 overflow
    nunique_ok = X_df.std(axis=0, numeric_only=True) > 0
    X_df = X_df.loc[:, nunique_ok.index[nunique_ok.values]]
    X_df = X_df.clip(lower=-1e9, upper=1e9).astype(np.float32)
    if X_df.shape[1] == 0:
        raise ValueError("All feature columns were constant/invalid after cleaning.")

    order = np.argsort(ts, kind="mergesort")
    X = X_df.to_numpy(np.float32)[order]
    y, ts = y[order], ts[order]

    tail_persist = np.mean(y[1:] == y[:-1]) * 100.0
    print(f"  LOADED time-tail: rows={X.shape[0]:,}  feat={X.shape[1]}  "
          f"attack={y.mean():.4f}  tail persistence={tail_persist:.2f}%")
    print(f"  label='{lb}'  time='{tsc}'")
    return X, y, ts, full_persist

def contiguous_tail(X, y, ts, n_keep):
    n = X.shape[0]
    if n <= n_keep: return X, y, ts
    sl = slice(n - n_keep, n)
    Xs, ys, tss = X[sl], y[sl], ts[sl]
    same = np.mean(ys[1:] == ys[:-1]) * 100.0
    print(f"  further contiguous tail {n_keep:,}: rows={Xs.shape[0]:,}  "
          f"attack={ys.mean():.4f}  persistence={same:.2f}%")
    return Xs, ys, tss

# ============================================================================
# PROTOTYPES + RELATIONSHIP  (identical to the TON-IoT scripts)
# ============================================================================
def make_relationship(X, a, b):
    da, db = X - a, X - b
    return np.hstack([da, db, np.abs(da), np.abs(db),
                      np.linalg.norm(da,axis=1)[:,None],
                      np.linalg.norm(db,axis=1)[:,None]]).astype(np.float32)

def compute_oracle(Xs, y, ts, n_bins):
    n, d = Xs.shape
    edges = np.quantile(ts, np.linspace(0,1,n_bins+1)); edges[0]-=1; edges[-1]+=1
    bid = np.clip(np.searchsorted(edges, ts, side="right")-1, 0, n_bins-1)
    ga = Xs[y==1].mean(0) if (y==1).any() else Xs.mean(0)
    gn = Xs[y==0].mean(0) if (y==0).any() else Xs.mean(0)
    aM = np.tile(ga,(n_bins,1)).astype(np.float32); nM = np.tile(gn,(n_bins,1)).astype(np.float32)
    for k in range(n_bins):
        m = bid==k; ma, mn = m&(y==1), m&(y==0)
        if ma.any(): aM[k]=Xs[ma].mean(0)
        if mn.any(): nM[k]=Xs[mn].mean(0)
    return aM[bid], nM[bid]

def compute_recency(Xs, y, W, fb_a, fb_n):
    n, d = Xs.shape
    yA=(y==1).astype(np.float64); yN=(y==0).astype(np.float64)
    csA=np.vstack([np.zeros((1,d)), np.cumsum(Xs*yA[:,None],0)])
    csN=np.vstack([np.zeros((1,d)), np.cumsum(Xs*yN[:,None],0)])
    cA=np.concatenate([[0],np.cumsum(yA)]); cN=np.concatenate([[0],np.cumsum(yN)])
    hi=np.arange(n); lo=np.maximum(0,hi-W)
    def build(cs,cnt,fb):
        sw=cs[hi]-cs[lo]; cw=cnt[hi]-cnt[lo]; se=cs[hi]; ce=cnt[hi]
        out=np.tile(fb,(n,1)).astype(np.float64)
        mw=cw>0; out[mw]=sw[mw]/cw[mw][:,None]
        me=(~mw)&(ce>0); out[me]=se[me]/ce[me][:,None]
        return out.astype(np.float32)
    return build(csA,cA,fb_a), build(csN,cN,fb_n)

# ============================================================================
# METRICS + STATS
# ============================================================================
def cmet(yt,yp): return dict(
    f1=f1_score(yt,yp,average="macro",zero_division=0)*100,
    bal=balanced_accuracy_score(yt,yp)*100,
    acc=accuracy_score(yt,yp)*100, mcc=matthews_corrcoef(yt,yp))

def make_rf(): return RandomForestClassifier(
    n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH, min_samples_leaf=RF_MIN_SAMPLES,
    random_state=SEED, n_jobs=-1, class_weight="balanced_subsample", max_features="sqrt")

def aut(vals):
    v=np.asarray(vals,float)/100.0
    if len(v)<2: return float(v.mean()) if len(v) else float("nan")
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x / 1.x
    return float(_trap(v, np.linspace(0,1,len(v))))

def paired(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); d=a-b
    if np.allclose(d,0): return (float("nan"),0.0,float(np.mean(d)))
    try: p=wilcoxon(a,b,zero_method="wilcox").pvalue
    except Exception: p=float("nan")
    pos=(d>0).sum(); neg=(d<0).sum(); tot=pos+neg
    return (p,(pos-neg)/tot if tot else 0.0,float(np.mean(d)))

# ============================================================================
# BOUNDARY CONFUSION (NF-BoT-IoT-v3): per-class precision/recall on class-transition rows
# for raw X, the leaked prototype (oracle@100) and recency_W50.
# Uses the NF-v3 loader above (streaming, sanitised); three RandomForest fits per fold.
# ============================================================================
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

CFG = dict(name="NF-BoT-IoT-v3", hint="bot-iot",
           path="/kaggle/input/datasets/ndayisabae/nf-bot-iot-v3/NF-BoT-IoT-v3.csv",
           subsample=500_000)
ORACLE_B = 100
REC_W    = 50

def run_boundary_confusion():
    t0=time.time()
    path=find_csv(CFG["hint"], CFG.get("path"))
    X,y,ts,full_persist=load_nfv3(path, keep_tail=CFG["subsample"])
    n,d=X.shape
    print("="*72+f"\n{CFG['name']} boundary confusion  rows={n:,} feat={d} attack={y.mean():.4f} "
          f"full_persistence={full_persist:.2f}%\n"+"="*72)
    conds=["raw X","oracle@%d"%ORACLE_B,"recency_W%d"%REC_W]
    conf={c:{"p1":[],"r1":[],"p0":[],"r0":[],"bacc":[],"nb":[]} for c in conds}
    starts=np.linspace(0.50,1.0-TEST_BLOCK,N_FOLDS)
    for f,frac in enumerate(starts,1):
        te0=int(round(n*frac)); te1=min(n,int(round(te0+n*TEST_BLOCK)))
        tr=np.zeros(n,bool);te=np.zeros(n,bool);tr[:te0]=True;te[te0:te1]=True
        tri,tei=np.where(tr)[0],np.where(te)[0]
        if (y[tr]==1).sum()==0 or (y[tr]==0).sum()==0:
            print(f"fold {f}: skipped (single-class train)"); continue
        sc=StandardScaler().fit(X[tr]);Xs=sc.transform(X).astype(np.float32)
        aM=Xs[tr&(y==1)].mean(0).astype(np.float32);nM=Xs[tr&(y==0)].mean(0).astype(np.float32)
        yte=y[tei];prev=y[tei-1];bm=(yte!=prev)
        preds={}
        mr=make_rf();mr.fit(Xs[tri],y[tri]);preds["raw X"]=mr.predict(Xs[tei])
        a,b=compute_oracle(Xs,y,ts,ORACLE_B);Zo=make_relationship(Xs,a,b)
        mo=make_rf();mo.fit(Zo[tri],y[tri]);preds["oracle@%d"%ORACLE_B]=mo.predict(Zo[tei])
        ra,rb=compute_recency(Xs,y,REC_W,aM,nM);Zr=make_relationship(Xs,ra,rb)
        mc=make_rf();mc.fit(Zr[tri],y[tri]);preds["recency_W%d"%REC_W]=mc.predict(Zr[tei])
        for c in conds:
            pr=preds[c]; d0=conf[c]; d0["nb"].append(int(bm.sum()))
            if bm.sum()>0:
                d0["bacc"].append(accuracy_score(yte[bm],pr[bm])*100)
                if len(np.unique(yte[bm]))>1:
                    d0["p1"].append(precision_score(yte[bm],pr[bm],pos_label=1,zero_division=0)*100)
                    d0["r1"].append(recall_score(yte[bm],pr[bm],pos_label=1,zero_division=0)*100)
                    d0["p0"].append(precision_score(yte[bm],pr[bm],pos_label=0,zero_division=0)*100)
                    d0["r0"].append(recall_score(yte[bm],pr[bm],pos_label=0,zero_division=0)*100)
        print(f"fold {f}/{N_FOLDS} boundary_rows={int(bm.sum())} ({time.time()-t0:.0f}s)")
    print("\n=== PER-CLASS PRECISION/RECALL AT BOUNDARIES (NF-BoT-IoT-v3, mean over folds) ===")
    print(f"{'condition':>16}{'bdry_acc':>10}{'P(atk)':>9}{'R(atk)':>9}{'P(nrm)':>9}{'R(nrm)':>9}")
    import numpy as _np
    rows=[]
    for c in conds:
        d0=conf[c]
        def m(k): return _np.mean(d0[k]) if len(d0[k]) else float('nan')
        print(f"{c:>16}{m('bacc'):>10.1f}{m('p1'):>9.1f}{m('r1'):>9.1f}{m('p0'):>9.1f}{m('r0'):>9.1f}")
        rows.append(dict(condition=c,boundary_acc=m('bacc'),P_attack=m('p1'),R_attack=m('r1'),
                         P_normal=m('p0'),R_normal=m('r0'),avg_boundary_rows=_np.mean(d0['nb'])))
    print(f"(avg boundary rows/fold: {_np.mean(conf['raw X']['nb']):.0f}  "
          f"-- at 99.8% attack these are few; interpret with caution)")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "BOTIOT_boundary_confusion.csv"),index=False)
    print(f"Saved BOTIOT_boundary_confusion.csv  ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    run_boundary_confusion()
