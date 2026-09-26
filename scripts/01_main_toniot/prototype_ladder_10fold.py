# ============================================================================
# PROTOTYPE LADDER, PERSISTENCE BASELINE AND KAPPA-TEMPORAL  (Sections 5.3-5.4; Tables 7, 9)
#
# TON-IoT, 10-fold expanding-window walk-forward protocol, fixed RandomForest probe.
# Conditions: raw X | static | leaked prototype ("oracle", B = 100 quantile bins over the whole
# series) | shuffled leaked prototype | causal recency W in {50, 250, 1000} (uses the true labels
# of the W preceding rows) | shuffled recency | window = 1 persistence baseline.
# Reports macro-F1, AUT, MCC, pooled Kappa-Temporal, boundary and within-block accuracy.
# The paired Wilcoxon statistics printed at the end are descriptive only: the folds overlap
# (Section 4.2) and these statistics are not used in the paper.
# RECENT_TRAIN_CAP (default None) can cap the training block to its most recent rows to reduce
# runtime; the reported run used None.
# Output: TONIOT_prototype_ladder_10fold.csv
# ============================================================================

import os, time, warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
)
from scipy.stats import wilcoxon

import os
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


warnings.filterwarnings("ignore")

# ================================================================
# CONFIG
# ================================================================
SEED            = 42
N_FOLDS         = 10
TEST_BLOCK      = 0.10
N_ORACLE_BINS   = 100
RECENCY_WINDOWS = [50, 250, 1000]

RF_N_TREES      = 300
RF_MAX_DEPTH    = 12
RF_MIN_SAMPLES  = 5

RECENT_TRAIN_CAP = None      # None = full training block (reported run)

rng = np.random.default_rng(SEED)
DATA_PATH = ("/kaggle/input/datasets/fadiabuzwayed/"
             "ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv")

# ================================================================
# COLUMNS  (shared TON-IoT preprocessing)
# ================================================================
ID_COLS   = ["src_ip", "src_port", "dst_ip", "dst_port"]
TS_COL, LABEL_BIN, LABEL_MULTI = "ts", "label", "type"
MISS = {"-", "n/a", "", " ", "(empty)", "nan", "none"}
ONEHOT = ["proto","service","conn_state","http_method","http_version","dns_AA",
          "dns_RD","dns_RA","dns_rejected","ssl_resumed","ssl_established"]
PRESENCE = ["dns_query","ssl_version","ssl_cipher","ssl_subject","ssl_issuer",
            "http_uri","http_referrer","http_user_agent","http_orig_mime_types",
            "http_resp_mime_types","weird_name","weird_addl","weird_notice",
            "http_trans_depth"]
CONTINUOUS = ["duration","src_bytes","dst_bytes","missed_bytes","src_pkts",
              "src_ip_bytes","dst_pkts","dst_ip_bytes","dns_qclass","dns_qtype",
              "dns_rcode","http_request_body_len","http_response_body_len","http_status_code"]

# ================================================================
# PREPROCESS  (identical)
# ================================================================
def load_and_preprocess(path):
    print(f"Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    obj = df.select_dtypes(include="object").columns
    if len(obj) > 0:
        df[obj] = df[obj].apply(lambda s: s.astype(str).str.strip()).replace({"-": "n/a"})
    df = df.drop_duplicates().reset_index(drop=True)
    y  = df[LABEL_BIN].astype(int).to_numpy()
    ts = pd.to_numeric(df[TS_COL], errors="coerce").ffill().fillna(0).to_numpy(np.float64)
    remove = [c for c in [TS_COL, LABEL_BIN, LABEL_MULTI, *ID_COLS] if c in df.columns]
    feats = df.drop(columns=remove); present = set(feats.columns); blocks = []
    ncols = [c for c in CONTINUOUS if c in present]
    if ncols: blocks.append(feats[ncols].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    pcols = [c for c in PRESENCE if c in present]
    if pcols:
        pres = feats[pcols].apply(lambda s: (~s.astype(str).str.strip().str.lower().isin(MISS)).astype(int))
        pres.columns = [f"has_{c}" for c in pcols]; blocks.append(pres)
    ocols = [c for c in ONEHOT if c in present]
    if ocols:
        blocks.append(pd.get_dummies(feats[ocols].astype(str), columns=ocols, prefix=ocols, dtype=np.int8))
    handled = set(ncols) | set(pcols) | set(ocols)
    rem = [c for c in feats.columns if c not in handled]
    if rem: blocks.append(feats[rem].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    X_df = pd.concat(blocks, axis=1); X_df = X_df.loc[:, ~X_df.columns.duplicated()].astype(float)
    order = np.argsort(ts, kind="mergesort")
    X_df = X_df.iloc[order].reset_index(drop=True); y, ts = y[order], ts[order]
    return X_df.to_numpy(np.float32), y, ts

# ================================================================
# PROTOTYPES + RELATIONSHIP
# ================================================================
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

# ================================================================
# METRICS + STATS
# ================================================================
def cm(yt,yp): return dict(
    f1=f1_score(yt,yp,average="macro",zero_division=0)*100,
    bal=balanced_accuracy_score(yt,yp)*100,
    acc=accuracy_score(yt,yp)*100, mcc=matthews_corrcoef(yt,yp))

def make_rf(): return RandomForestClassifier(
    n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH, min_samples_leaf=RF_MIN_SAMPLES,
    random_state=SEED, n_jobs=-1, class_weight="balanced_subsample", max_features="sqrt")

def aut(vals):
    # Area Under Time: trapezoid of per-fold F1 (in [0,1]) over the normalised fold axis.
    v = np.asarray(vals, float)/100.0
    if len(v) < 2: return float(v.mean()) if len(v) else float("nan")
    x = np.linspace(0,1,len(v))
    return float(getattr(np, "trapezoid", getattr(np, "trapz", None))(v, x))

def paired(a, b):
    # Wilcoxon signed-rank p-value + sign-based effect size (pos - neg) / n.
    a, b = np.asarray(a,float), np.asarray(b,float)
    d = a - b
    if np.allclose(d, 0): return (float("nan"), 0.0, float(np.mean(d)))
    try: p = wilcoxon(a, b, zero_method="wilcox").pvalue
    except Exception: p = float("nan")
    pos = (d > 0).sum(); neg = (d < 0).sum(); tot = pos + neg
    rb = (pos - neg)/tot if tot else 0.0        # sign-based effect size
    return (p, rb, float(np.mean(d)))

# ================================================================
# MAIN
# ================================================================
def run():
    t0 = time.time()
    X, y, ts = load_and_preprocess(DATA_PATH)
    n, d = X.shape
    same = np.mean(y[1:]==y[:-1])*100
    print("\n"+"="*78+f"\nDATASET  rows={n:,}  feat={d}  attack={y.mean():.3f}  "
          f"persistence={same:.2f}%\n"+"="*78)
    print(f"folds={N_FOLDS}  RF_trees={RF_N_TREES}  recency={RECENCY_WINDOWS}  "
          f"cap={RECENT_TRAIN_CAP}")

    conds = ["raw X","static","oracle","oracle_shuf"] + \
            [f"recency_W{w}" for w in RECENCY_WINDOWS] + \
            [f"recency_shuf_W{w}" for w in RECENCY_WINDOWS]
    res = {c: [] for c in conds}
    bnd = {c: [] for c in conds}          # (bdry_acc, within_acc, bdry_f1)
    per = []                              # persistence window=1

    starts = np.linspace(0.50, 1.0-TEST_BLOCK, N_FOLDS)
    for fold, frac in enumerate(starts, 1):
        tr_end = int(round(n*frac)); te_end = min(n, int(round(tr_end+n*TEST_BLOCK)))
        tr = np.zeros(n,bool); te = np.zeros(n,bool); tr[:tr_end]=True; te[tr_end:te_end]=True
        tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
        print(f"\nFOLD {fold}/{N_FOLDS}  train={tr.sum():,}  test={te.sum():,}")

        scaler = StandardScaler().fit(X[tr]); Xs = scaler.transform(X).astype(np.float32)
        aM = Xs[tr&(y==1)].mean(0).astype(np.float32)
        nM = Xs[tr&(y==0)].mean(0).astype(np.float32)

        a_orac, b_orac = compute_oracle(Xs, y, ts, N_ORACLE_BINS)
        recency = {w: compute_recency(Xs, y, w, aM, nM) for w in RECENCY_WINDOWS}

        Z = {"raw X": Xs,
             "static": make_relationship(Xs, np.tile(aM,(n,1)), np.tile(nM,(n,1))),
             "oracle": make_relationship(Xs, a_orac, b_orac)}
        for w in RECENCY_WINDOWS:
            ra, rb = recency[w]; Z[f"recency_W{w}"] = make_relationship(Xs, ra, rb)

        def shuf(a_all, b_all):
            ap, bp = np.empty_like(a_all), np.empty_like(b_all)
            ap[tr_idx]=a_all[rng.permutation(tr_idx)]; bp[tr_idx]=b_all[rng.permutation(tr_idx)]
            ap[te_idx]=a_all[rng.permutation(te_idx)]; bp[te_idx]=b_all[rng.permutation(te_idx)]
            return make_relationship(Xs, ap, bp)
        Z["oracle_shuf"] = shuf(a_orac, b_orac)
        for w in RECENCY_WINDOWS:
            ra, rb = recency[w]; Z[f"recency_shuf_W{w}"] = shuf(ra, rb)

        tr_use = tr_idx
        if RECENT_TRAIN_CAP and len(tr_idx) > RECENT_TRAIN_CAP:
            tr_use = tr_idx[-RECENT_TRAIN_CAP:]
        y_tr, y_te = y[tr_use], y[te_idx]

        prev = y[te_idx-1]; bm = (y_te != prev); wi = ~bm
        pm = cm(y_te, prev)
        pm["bdry"] = accuracy_score(y_te[bm], prev[bm])*100 if bm.any() else np.nan
        pm["within"] = accuracy_score(y_te[wi], prev[wi])*100 if wi.any() else np.nan
        pm["nb"] = int(bm.sum()); per.append(pm)

        for c in conds:
            rf = make_rf(); rf.fit(Z[c][tr_use], y_tr); pr = rf.predict(Z[c][te_idx])
            res[c].append(cm(y_te, pr))
            ab = accuracy_score(y_te[bm], pr[bm])*100 if bm.any() else np.nan
            aw = accuracy_score(y_te[wi], pr[wi])*100 if wi.any() else np.nan
            fb = f1_score(y_te[bm], pr[bm], average="macro", zero_division=0)*100 if bm.any() else np.nan
            bnd[c].append((ab, aw, fb))
        print(f"  raw={res['raw X'][-1]['f1']:.1f} static={res['static'][-1]['f1']:.1f} "
              f"oracle={res['oracle'][-1]['f1']:.1f} "
              f"rec250={res['recency_W250'][-1]['f1']:.1f} persist={pm['f1']:.1f}")

    # ---------- summary ----------
    def col(c,k): return np.array([r[k] for r in res[c]])
    print("\n"+"="*78+"\nOVERALL (mean +/- std)   +   AUT\n"+"="*78)
    print(f"{'condition':20s}{'macroF1':>16s}{'AUT':>10s}{'MCC':>10s}")
    for c in conds:
        f1 = col(c,"f1")
        print(f"{c:20s}{f1.mean():7.2f} +/-{f1.std():5.2f}{aut(f1):10.3f}{col(c,'mcc').mean():10.3f}")
    pf = np.array([p["f1"] for p in per])
    print(f"{'persistence W=1':20s}{pf.mean():7.2f} +/-{pf.std():5.2f}{aut(pf):10.3f}")

    # ---------- Kappa-Temporal (Bifet 2013; Zliobaite et al. 2015) ----------
    # kappa_t = (acc_model - acc_persistent) / (1 - acc_persistent)
    #   = 0  -> no better than predicting the previous label
    #   < 0  -> worse than the persistence baseline
    #   = 1  -> perfect. Uses ACCURACY, not macro-F1, by definition.
    print("\n"+"="*78+"\nKAPPA-TEMPORAL  (vs window=1 persistent baseline; <0 = worse than persistence)\n"+"="*78)
    pa = np.array([p["acc"] for p in per])/100.0          # persistent-baseline accuracy per fold
    valid = pa < 1.0                                       # folds where 1-acc_persist > 0 (else undefined)
    print(f"{'condition':20s}{'kappaT_pooled':>16s}{'kappaT_foldmean':>18s}{'n_valid':>9s}")
    kt_store = {}
    for c in conds:
        ac = col(c,"acc")/100.0
        kt_pool = (ac.mean() - pa.mean())/(1 - pa.mean()) if pa.mean() < 1 else float("nan")
        kt_fold = (ac[valid]-pa[valid])/(1-pa[valid]) if valid.any() else np.array([np.nan])
        kt_store[c] = kt_pool
        print(f"{c:20s}{kt_pool:16.2f}{np.nanmean(kt_fold):18.2f}{int(valid.sum()):9d}")
    print(f"(persistent baseline accuracy = {pa.mean()*100:.2f}% ; 1-acc = {(1-pa.mean())*100:.2f}%)")

    print("\n"+"="*78+"\nBOUNDARY-STRATIFIED (mean over folds)\n"+"="*78)
    print(f"{'condition':20s}{'bdry_acc':>12s}{'within':>12s}{'bdry_F1':>12s}")
    for c in conds:
        a = np.array(bnd[c]); print(f"{c:20s}{np.nanmean(a[:,0]):12.2f}{np.nanmean(a[:,1]):12.2f}{np.nanmean(a[:,2]):12.2f}")
    pb = np.array([[p["bdry"], p["within"]] for p in per])
    print(f"{'persistence W=1':20s}{np.nanmean(pb[:,0]):12.2f}{np.nanmean(pb[:,1]):12.2f}{'--':>12s}")
    print(f"(avg boundary rows/fold: {np.mean([p['nb'] for p in per]):.0f})")

    # ---------- paired stats ----------
    print("\n"+"="*78+"\nPAIRED WILCOXON, descriptive (n=%d overlapping folds)  p | sign effect | mean diff\n"%N_FOLDS+"="*78)
    def show(name, a, b):
        p, rb, md = paired(a, b)
        print(f"  {name:34s} p={p:6.4f}  rb={rb:+.2f}  mean_diff={md:+.2f}")
    show("learned/oracle vs raw (F1)", col("oracle","f1"), col("raw X","f1"))
    show("oracle vs static (F1)",       col("oracle","f1"), col("static","f1"))
    show("oracle vs oracle_shuf (F1)",  col("oracle","f1"), col("oracle_shuf","f1"))
    bw = max(RECENCY_WINDOWS, key=lambda w: col(f"recency_W{w}","f1").mean())
    show(f"recency_W{bw} vs raw (F1)",   col(f"recency_W{bw}","f1"), col("raw X","f1"))
    show(f"recency_W{bw} vs its shuffle (F1)", col(f"recency_W{bw}","f1"), col(f"recency_shuf_W{bw}","f1"))
    # Boundary test: compare against RAW X (the meaningful baseline).
    # NB: persistence boundary-acc is a constant 0.00 by construction, so a
    # Wilcoxon against it is degenerate (zero variance -> nan) and meaningless.
    rec_bd = np.array(bnd[f"recency_W{bw}"])[:,0]
    orc_bd = np.array(bnd["oracle"])[:,0]
    raw_bd = np.array(bnd["raw X"])[:,0]
    show(f"recency_W{bw} vs raw X (BOUNDARY acc)", rec_bd, raw_bd)
    show("oracle vs raw X (BOUNDARY acc)",        orc_bd, raw_bd)

    # ---------- summary ----------
    gap = col("oracle","f1").mean() - col("static","f1").mean()
    print("\n"+"="*78+"\nSUMMARY\n"+"="*78)
    print(f"oracle F1 = {col('oracle','f1').mean():.2f} (oracle-static "
          f"= {gap:+.2f}, oracle-oracle_shuf = "
          f"{col('oracle','f1').mean()-col('oracle_shuf','f1').mean():+.2f})")
    print(f"persistence W=1 F1 = {pf.mean():.2f} (window = 1 baseline)")
    print(f"BOUNDARY acc: raw={np.nanmean(raw_bd):.2f}  oracle={np.nanmean(orc_bd):.2f}  "
          f"recency_W{bw}={np.nanmean(rec_bd):.2f}  "
          f"recency_shuf={np.nanmean(np.array(bnd[f'recency_shuf_W{bw}'])[:,0]):.2f}")
    # Does any prototype condition exceed raw X at boundaries by more than 3 points?
    boundary_real = (np.nanmean(rec_bd) > np.nanmean(raw_bd) + 3.0) or \
                    (np.nanmean(orc_bd) > np.nanmean(raw_bd) + 3.0)
    print(">> " + ("A prototype condition exceeds raw X at boundaries by more than 3 points."
                   if boundary_real else
                   "Neither the leaked prototype nor recency exceeds raw X at boundaries."))

    # save
    rows = [dict(condition=c, macro_f1=col(c,"f1").mean(), aut=aut(col(c,"f1")),
                 kappa_temporal=kt_store.get(c, float("nan")),
                 boundary_acc=np.nanmean(np.array(bnd[c])[:,0]),
                 within_acc=np.nanmean(np.array(bnd[c])[:,1])) for c in conds]
    rows.append(dict(condition="persistence_W1", macro_f1=pf.mean(), aut=aut(pf),
                     boundary_acc=np.nanmean(pb[:,0]), within_acc=np.nanmean(pb[:,1])))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "TONIOT_prototype_ladder_10fold.csv"), index=False)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'TONIOT_prototype_ladder_10fold.csv')}   Runtime: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    run()
