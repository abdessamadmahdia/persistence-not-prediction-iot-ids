# ============================================================================
# NF-v3 REPLICATION  (Section 5.7, Table 13: NF-BoT-IoT-v3 and NF-UNSW-NB15-v3)
#
# Same protocol as the TON-IoT runs: 10-fold expanding-window walk-forward, fixed RandomForest
# probe, raw X | static | leaked prototype ("oracle") | shuffled | causal recency (label feedback)
# | window = 1 persistence baseline, pooled Kappa-Temporal, boundary-stratified accuracy.
# load_nfv3(): full-file persistence from [timestamp, label]; classification on the most recent
# ~500,000 rows selected by a timestamp threshold; exact duplicates removed on all columns except
# the IPv4 address columns; infinities/NaNs set to 0, zero-variance columns dropped, values
# clipped to +/-1e9. Never sub-sample randomly: that would destroy the temporal order.
# The paired Wilcoxon statistics printed at the end are descriptive only and are not used.
# Run once with DATASET = "botiot" and once with DATASET = "unsw".
# Outputs: confound_NF_BoT_IoT_v3.csv, confound_NF_UNSW_NB15_v3.csv
# ============================================================================

import os, glob, time, warnings, functools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef)
from scipy.stats import wilcoxon
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
# EXPERIMENT
# ============================================================================
def run(cfg):
    t0=time.time()
    path = find_csv(cfg["hint"], cfg.get("path"))
    sub = 150_000 if FAST_MODE else cfg.get("subsample")
    X, y, ts, full_persist = load_nfv3(path, keep_tail=sub)

    if len(np.unique(y)) < 2:
        raise ValueError("Only one class present after loading; check label column.")

    n, d = X.shape
    print("="*78+f"\n{cfg['name']}  |  rows={n:,}  feat={d}  attack={y.mean():.4f}  "
          f"full_persistence={full_persist:.2f}%\n"+"="*78)
    print(f"folds={N_FOLDS} trees={RF_N_TREES} recency={RECENCY_WINDOWS} FAST={FAST_MODE}")

    conds=["raw X","static","oracle","oracle_shuf"]+\
          [f"recency_W{w}" for w in RECENCY_WINDOWS]+\
          [f"recency_shuf_W{w}" for w in RECENCY_WINDOWS]
    res={c:[] for c in conds}; bnd={c:[] for c in conds}; per=[]

    starts=np.linspace(0.50,1.0-TEST_BLOCK,N_FOLDS)
    for fold,frac in enumerate(starts,1):
        tr_end=int(round(n*frac)); te_end=min(n,int(round(tr_end+n*TEST_BLOCK)))
        tr=np.zeros(n,bool); te=np.zeros(n,bool); tr[:tr_end]=True; te[tr_end:te_end]=True
        tr_idx,te_idx=np.where(tr)[0],np.where(te)[0]
        if (y[tr]==1).sum()==0 or (y[tr]==0).sum()==0 or len(te_idx)==0:
            print(f"FOLD {fold}: skipped (a class missing in train)"); continue

        sc=StandardScaler().fit(X[tr]); Xs=sc.transform(X).astype(np.float32)
        aM=Xs[tr&(y==1)].mean(0).astype(np.float32); nM=Xs[tr&(y==0)].mean(0).astype(np.float32)
        a_or,b_or=compute_oracle(Xs,y,ts,N_ORACLE_BINS)
        recency={w:compute_recency(Xs,y,w,aM,nM) for w in RECENCY_WINDOWS}

        Z={"raw X":Xs,
           "static":make_relationship(Xs,np.tile(aM,(n,1)),np.tile(nM,(n,1))),
           "oracle":make_relationship(Xs,a_or,b_or)}
        for w in RECENCY_WINDOWS:
            ra,rb=recency[w]; Z[f"recency_W{w}"]=make_relationship(Xs,ra,rb)
        def shuf(a_all,b_all):
            ap,bp=np.empty_like(a_all),np.empty_like(b_all)
            ap[tr_idx]=a_all[rng.permutation(tr_idx)]; bp[tr_idx]=b_all[rng.permutation(tr_idx)]
            ap[te_idx]=a_all[rng.permutation(te_idx)]; bp[te_idx]=b_all[rng.permutation(te_idx)]
            return make_relationship(Xs,ap,bp)
        Z["oracle_shuf"]=shuf(a_or,b_or)
        for w in RECENCY_WINDOWS:
            ra,rb=recency[w]; Z[f"recency_shuf_W{w}"]=shuf(ra,rb)

        y_tr,y_te=y[tr_idx],y[te_idx]
        prev=y[te_idx-1]; bm=(y_te!=prev); wi=~bm
        pm=cmet(y_te,prev)
        pm["bdry"]=accuracy_score(y_te[bm],prev[bm])*100 if bm.any() else np.nan
        pm["within"]=accuracy_score(y_te[wi],prev[wi])*100 if wi.any() else np.nan
        pm["nb"]=int(bm.sum()); per.append(pm)

        for c in conds:
            rf=make_rf(); rf.fit(Z[c][tr_idx],y_tr); pr=rf.predict(Z[c][te_idx])
            res[c].append(cmet(y_te,pr))
            ab=accuracy_score(y_te[bm],pr[bm])*100 if bm.any() else np.nan
            aw=accuracy_score(y_te[wi],pr[wi])*100 if wi.any() else np.nan
            fb=f1_score(y_te[bm],pr[bm],average="macro",zero_division=0)*100 if bm.any() else np.nan
            bnd[c].append((ab,aw,fb))
        print(f"FOLD {fold}/{N_FOLDS} tr={tr.sum():,} te={te.sum():,}  "
              f"raw={res['raw X'][-1]['f1']:.1f} oracle={res['oracle'][-1]['f1']:.1f} "
              f"rec50={res.get('recency_W50',res[conds[4]])[-1]['f1']:.1f} persist={pm['f1']:.1f}")

    # ---- summary ----
    def col(c,k): return np.array([r[k] for r in res[c]])
    print("\n"+"="*78+"\nOVERALL (mean +/- std) + AUT\n"+"="*78)
    print(f"{'condition':20s}{'macroF1':>16s}{'AUT':>10s}{'MCC':>10s}")
    for c in conds:
        f1=col(c,"f1")
        print(f"{c:20s}{f1.mean():7.2f} +/-{f1.std():5.2f}{aut(f1):10.3f}{col(c,'mcc').mean():10.3f}")
    pf=np.array([p["f1"] for p in per])
    print(f"{'persistence W=1':20s}{pf.mean():7.2f} +/-{pf.std():5.2f}{aut(pf):10.3f}")

    # ---- Kappa-Temporal (pooled + fold-mean; report pooled) ----
    print("\n"+"="*78+"\nKAPPA-TEMPORAL (vs window=1 persistent; <0 = worse than persistence)\n"+"="*78)
    pa=np.array([p["acc"] for p in per])/100.0; valid=pa<1.0
    print(f"{'condition':20s}{'kappaT_pooled':>16s}{'kappaT_foldmean':>18s}")
    kt_store={}
    for c in conds:
        ac=col(c,"acc")/100.0
        ktp=(ac.mean()-pa.mean())/(1-pa.mean()) if pa.mean()<1 else float("nan")
        ktf=(ac[valid]-pa[valid])/(1-pa[valid]) if valid.any() else np.array([np.nan])
        kt_store[c]=ktp
        print(f"{c:20s}{ktp:16.2f}{np.nanmean(ktf):18.2f}")
    print(f"(persistent baseline accuracy = {pa.mean()*100:.2f}% ; 1-acc = {(1-pa.mean())*100:.3f}%)")

    # ---- boundary ----
    print("\n"+"="*78+"\nBOUNDARY-STRATIFIED (mean over folds)\n"+"="*78)
    print(f"{'condition':20s}{'bdry_acc':>12s}{'within':>12s}{'bdry_F1':>12s}")
    for c in conds:
        a=np.array(bnd[c]); print(f"{c:20s}{np.nanmean(a[:,0]):12.2f}{np.nanmean(a[:,1]):12.2f}{np.nanmean(a[:,2]):12.2f}")
    pb=np.array([[p["bdry"],p["within"]] for p in per])
    print(f"{'persistence W=1':20s}{np.nanmean(pb[:,0]):12.2f}{np.nanmean(pb[:,1]):12.2f}{'--':>12s}")
    print(f"(avg boundary rows/fold: {np.mean([p['nb'] for p in per]):.0f})")

    # ---- paired stats (vs raw X, the meaningful baseline) ----
    print("\n"+"="*78+f"\nPAIRED WILCOXON, descriptive (n={len(per)} overlapping folds)  p | sign effect | mean diff\n"+"="*78)
    def show(nm,a,b):
        p,rb,md=paired(a,b); print(f"  {nm:34s} p={p:6.4f}  rb={rb:+.2f}  mean_diff={md:+.2f}")
    show("oracle vs static (F1)", col("oracle","f1"), col("static","f1"))
    show("oracle vs oracle_shuf (F1)", col("oracle","f1"), col("oracle_shuf","f1"))
    bw=max(RECENCY_WINDOWS,key=lambda w:col(f"recency_W{w}","f1").mean())
    show(f"recency_W{bw} vs raw (F1)", col(f"recency_W{bw}","f1"), col("raw X","f1"))
    show(f"recency_W{bw} vs its shuffle (F1)", col(f"recency_W{bw}","f1"), col(f"recency_shuf_W{bw}","f1"))
    rec_bd=np.array(bnd[f"recency_W{bw}"])[:,0]; orc_bd=np.array(bnd["oracle"])[:,0]; raw_bd=np.array(bnd["raw X"])[:,0]
    show(f"recency_W{bw} vs raw X (BOUNDARY)", rec_bd, raw_bd)
    show("oracle vs raw X (BOUNDARY)", orc_bd, raw_bd)

    # ---- summary ----
    print("\n"+"="*78+"\nSUMMARY\n"+"="*78)
    print(f"full-dataset persistence = {full_persist:.2f}%")
    print(f"persistence W=1 F1 = {pf.mean():.2f} (AUT {aut(pf):.3f})")
    print(f"oracle F1 = {col('oracle','f1').mean():.2f} "
          f"(oracle-static {col('oracle','f1').mean()-col('static','f1').mean():+.2f})")
    print(f"BOUNDARY acc: raw={np.nanmean(raw_bd):.2f} oracle={np.nanmean(orc_bd):.2f} "
          f"recency_W{bw}={np.nanmean(rec_bd):.2f} "
          f"recency_shuf={np.nanmean(np.array(bnd[f'recency_shuf_W{bw}'])[:,0]):.2f}")
    real=(np.nanmean(rec_bd)>np.nanmean(raw_bd)+3.0) or (np.nanmean(orc_bd)>np.nanmean(raw_bd)+3.0)
    print(">> "+("A prototype condition exceeds raw X at boundaries by more than 3 points." if real else
                 "Neither the leaked prototype nor recency exceeds raw X at boundaries."))

    # ---- save ----
    tag=cfg["name"].replace("-","_")
    rows=[dict(dataset=cfg["name"], condition=c, macro_f1=col(c,"f1").mean(),
               aut=aut(col(c,"f1")), kappa_temporal=kt_store.get(c,np.nan),
               boundary_acc=np.nanmean(np.array(bnd[c])[:,0]),
               within_acc=np.nanmean(np.array(bnd[c])[:,1])) for c in conds]
    rows.append(dict(dataset=cfg["name"], condition="persistence_W1", macro_f1=pf.mean(),
                     aut=aut(pf), kappa_temporal=0.0,
                     boundary_acc=np.nanmean(pb[:,0]), within_acc=np.nanmean(pb[:,1])))
    rows.append(dict(dataset=cfg["name"], condition="_FULL_PERSISTENCE_%",
                     macro_f1=full_persist, aut=np.nan, kappa_temporal=np.nan,
                     boundary_acc=np.nan, within_acc=np.nan))
    out_dir = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out=f"{out_dir}/confound_{tag}.csv"
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"\nSaved: {out}   Runtime: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    print = functools.partial(print, flush=True)
    run(DATASETS[DATASET])
