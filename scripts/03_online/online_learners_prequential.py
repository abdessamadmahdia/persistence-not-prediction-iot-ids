# ============================================================================
# ONLINE LEARNERS VS BATCH RANDOMFOREST, PREQUENTIAL  (Section 5.6, Table 12)
#
# Adaptive Random Forest (ARF), Hoeffding tree with ADWIN-triggered reset and plain Hoeffding
# tree, evaluated prequentially (predict, then learn from the true label) over the TON-IoT
# stream; metrics on the chronological second half. The batch RandomForest is trained once on
# the first half and receives no labels afterwards. Features are standardised on the first half.
# Reports macro-F1, boundary and within-block accuracy, Kappa-Temporal vs the window = 1
# baseline, and 95% bootstrap intervals for boundary accuracy (paired differences vs batch RF).
# Outputs: TONIOT_online_learners.csv, TONIOT_online_learners_perrow_predictions.csv
# ============================================================================
import time, warnings, subprocess, sys
warnings.filterwarnings("ignore")

# Import the environment-critical libs FIRST, before touching river.
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Install river with --no-deps so it CANNOT upgrade NumPy/SciPy/scikit-learn
# (a plain `pip install river` pulls a newer NumPy and breaks Kaggle's sklearn:
#  AttributeError: _multiarray_umath has no attribute '_blas_supports_fpe').
try:
    import river
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","--no-deps","river"])
    import river

# river imports (handle version differences)
try:    from river.forest import ARFClassifier
except Exception: from river.ensemble import AdaptiveRandomForestClassifier as ARFClassifier
from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN

import os
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


SEED=42
EVAL_FROM=0.50          # evaluate on the chronological second half (matches walk-forward test range)
SUBSAMPLE=None          # e.g. 200_000 keeps a contiguous tail for a faster run; None = full file (reported)
ARF_MODELS=10
DATA_PATH=("/kaggle/input/datasets/fadiabuzwayed/"
           "ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv")

# ---- TON-IoT columns (same preprocessing as the batch experiments) ----
ID_COLS=["src_ip","src_port","dst_ip","dst_port"]; TS_COL="ts"
LABEL_BIN="label"; LABEL_MULTI="type"
MISS={"-","n/a",""," ","(empty)","nan","none"}
ONEHOT=["proto","service","conn_state","http_method","http_version","dns_AA",
        "dns_RD","dns_RA","dns_rejected","ssl_resumed","ssl_established"]
PRESENCE=["dns_query","ssl_version","ssl_cipher","ssl_subject","ssl_issuer",
          "http_uri","http_referrer","http_user_agent","http_orig_mime_types",
          "http_resp_mime_types","weird_name","weird_addl","weird_notice","http_trans_depth"]
CONTINUOUS=["duration","src_bytes","dst_bytes","missed_bytes","src_pkts","src_ip_bytes",
            "dst_pkts","dst_ip_bytes","dns_qclass","dns_qtype","dns_rcode",
            "http_request_body_len","http_response_body_len","http_status_code"]

def load(path):
    df=pd.read_csv(path,low_memory=False); df.columns=[c.strip() for c in df.columns]
    obj=df.select_dtypes(include="object").columns
    if len(obj): df[obj]=df[obj].apply(lambda s:s.astype(str).str.strip()).replace({"-":"n/a"})
    df=df.drop_duplicates().reset_index(drop=True)
    y=df[LABEL_BIN].astype(int).to_numpy()
    ts=pd.to_numeric(df[TS_COL],errors="coerce").ffill().fillna(0).to_numpy(np.float64)
    rm=[c for c in [TS_COL,LABEL_BIN,LABEL_MULTI,*ID_COLS] if c in df.columns]
    feats=df.drop(columns=rm); present=set(feats.columns); blocks=[]
    nc=[c for c in CONTINUOUS if c in present]
    if nc: blocks.append(feats[nc].apply(pd.to_numeric,errors="coerce").fillna(0.0))
    pc=[c for c in PRESENCE if c in present]
    if pc:
        pr=feats[pc].apply(lambda s:(~s.astype(str).str.strip().str.lower().isin(MISS)).astype(int))
        pr.columns=[f"has_{c}" for c in pc]; blocks.append(pr)
    oc=[c for c in ONEHOT if c in present]
    if oc: blocks.append(pd.get_dummies(feats[oc].astype(str),columns=oc,prefix=oc,dtype=np.int8))
    handled=set(nc)|set(pc)|set(oc); rem=[c for c in feats.columns if c not in handled]
    if rem: blocks.append(feats[rem].apply(pd.to_numeric,errors="coerce").fillna(0.0))
    X=pd.concat(blocks,axis=1); X=X.loc[:,~X.columns.duplicated()].astype(float)
    o=np.argsort(ts,kind="mergesort"); X=X.iloc[o].reset_index(drop=True); y=y[o]
    return X.to_numpy(np.float32), y, list(X.columns)

def make_arf():
    try:    return ARFClassifier(n_models=ARF_MODELS, seed=SEED)
    except TypeError: return ARFClassifier(n_models=ARF_MODELS, random_state=SEED)

def prequential(model, Xd, y, feat, eval_from_idx, adwin_retrain=False):
    """Predict-then-learn over the stream; collect predictions on eval region.
    If adwin_retrain, an ADWIN monitors errors and resets the model on drift
    (a simple ADWIN-triggered adaptive baseline around a Hoeffding tree)."""
    n=len(y); preds=np.full(n,-1,dtype=int); adw=ADWIN() if adwin_retrain else None
    for i in range(n):
        x={feat[j]:float(Xd[i,j]) for j in range(len(feat))}
        yp=model.predict_one(x)
        if yp is None: yp=0
        preds[i]=int(yp)
        model.learn_one(x,int(y[i]))
        if adw is not None:
            adw.update(0.0 if yp==y[i] else 1.0)
            if adw.drift_detected:
                model=HoeffdingTreeClassifier()   # reset on detected drift
    return preds

def diagnostics(name, y, preds, eval_from_idx):
    ev=np.arange(eval_from_idx,len(y))
    yt=y[ev]; yp=preds[ev]; prev=y[ev-1]
    bm=(yt!=prev); wi=~bm
    persist_acc=accuracy_score(yt,prev)
    acc=accuracy_score(yt,yp)
    f1=f1_score(yt,yp,average="macro",zero_division=0)*100
    bacc=accuracy_score(yt[bm],yp[bm])*100 if bm.any() else np.nan
    wacc=accuracy_score(yt[wi],yp[wi])*100 if wi.any() else np.nan
    kt=(acc-persist_acc)/(1-persist_acc) if persist_acc<1 else float("nan")
    return dict(method=name,macro_f1=f1,boundary_acc=bacc,within_acc=wacc,
                kappa_temporal=kt,eval_rows=len(ev),boundary_rows=int(bm.sum()))

def _eval_preds(y, preds, efi):
    """Return (y_true_eval, y_pred_eval, boundary_mask_eval)."""
    ev=np.arange(efi,len(y)); yt=y[ev]; yp=preds[ev]; prev=y[ev-1]
    return yt, yp, (yt!=prev)

def boundary_bootstrap(yt, preds_dict, bm, base_key, n_boot=5000, seed=42):
    """95% CIs for boundary accuracy per method, and paired diff vs base_key."""
    rng=np.random.default_rng(seed)
    bi=np.where(bm)[0]
    if len(bi)==0:
        print("no boundary rows -> skip bootstrap"); return
    corr={k:(preds_dict[k][bi]==yt[bi]).astype(float) for k in preds_dict}
    print("\n95pct BOOTSTRAP CI - boundary accuracy (n_bdry=%d)"%len(bi))
    print(f"{'method':>20}{'acc':>9}{'95% CI':>20}")
    for k in preds_dict:
        c=corr[k]; pt=c.mean()*100
        bs=np.array([c[rng.integers(0,len(c),len(c))].mean()*100 for _ in range(n_boot)])
        lo,hi=np.percentile(bs,[2.5,97.5])
        print(f"{k:>20}{pt:>9.2f}   [{lo:.2f}, {hi:.2f}]")
    print(f"\nBOUNDARY DIFFERENCE vs '{base_key}' (paired bootstrap; sig if CI excludes 0)")
    cb=corr[base_key]
    for k in preds_dict:
        if k==base_key: continue
        ca=corr[k]; pt=(ca.mean()-cb.mean())*100
        bs=np.empty(n_boot)
        for j in range(n_boot):
            s=rng.integers(0,len(bi),len(bi)); bs[j]=(ca[s].mean()-cb[s].mean())*100
        lo,hi=np.percentile(bs,[2.5,97.5]); sig=(lo>0 or hi<0)
        print(f"  {k:>18} - {base_key}: {pt:+.2f}  [{lo:+.2f}, {hi:+.2f}]  "
              f"{'SIGNIFICANT' if sig else 'not sig (spans 0)'}")

def run():
    t0=time.time(); X,y,feat=load(DATA_PATH)
    if SUBSAMPLE and len(y)>SUBSAMPLE:
        X,y=X[-SUBSAMPLE:],y[-SUBSAMPLE:]
    n=len(y); efi=int(round(n*EVAL_FROM))
    sc=StandardScaler().fit(X[:efi]); Xd=sc.transform(X).astype(np.float32)
    print(f"rows={n:,} feat={len(feat)} persistence={np.mean(y[1:]==y[:-1])*100:.2f}% "
          f"eval_from={efi:,}")
    rows=[]; evalpred={}   # method -> full-length pred array (only eval region used)

    # window=1 persistence baseline
    prev_all=np.empty(n,int); prev_all[0]=y[0]; prev_all[1:]=y[:-1]
    rows.append(diagnostics("persistence W=1", y, prev_all, efi)); evalpred["persistence W=1"]=prev_all
    print(f"[{time.time()-t0:.0f}s] persistence done")

    # BATCH RandomForest raw reference (train on pre-eval half, predict eval region)
    rf=RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=5,
        random_state=SEED,n_jobs=-1,class_weight="balanced_subsample",max_features="sqrt")
    rf.fit(Xd[:efi], y[:efi])
    rawp=np.empty(n,int); rawp[:efi]=y[:efi]; rawp[efi:]=rf.predict(Xd[efi:])
    rows.append(diagnostics("batch RF (raw)", y, rawp, efi)); evalpred["batch RF (raw)"]=rawp
    print(f"[{time.time()-t0:.0f}s] batch RF done")

    # ARF (ADWIN-driven adaptive ensemble)
    parf=prequential(make_arf(),Xd,y,feat,efi)
    rows.append(diagnostics("ARF (ADWIN)", y, parf, efi)); evalpred["ARF (ADWIN)"]=parf
    print(f"[{time.time()-t0:.0f}s] ARF done")
    # ADWIN-triggered Hoeffding tree (reset on drift)
    pah=prequential(HoeffdingTreeClassifier(),Xd,y,feat,efi,adwin_retrain=True)
    rows.append(diagnostics("HT + ADWIN reset", y, pah, efi)); evalpred["HT + ADWIN reset"]=pah
    print(f"[{time.time()-t0:.0f}s] HT+ADWIN done")
    # plain Hoeffding tree
    pht=prequential(HoeffdingTreeClassifier(),Xd,y,feat,efi)
    rows.append(diagnostics("Hoeffding tree", y, pht, efi)); evalpred["Hoeffding tree"]=pht
    print(f"[{time.time()-t0:.0f}s] HT done")

    print("\n=== ADAPTIVE DRIFT-METHOD BASELINE (TON-IoT, prequential on 2nd half) ===")
    print(f"{'method':>20}{'macroF1':>10}{'bdry_acc':>10}{'within':>10}{'kappaT':>10}")
    for r in rows:
        print(f"{r['method']:>20}{r['macro_f1']:>10.2f}{r['boundary_acc']:>10.2f}"
              f"{r['within_acc']:>10.2f}{r['kappa_temporal']:>10.3f}")
    print(f"(eval rows: {rows[0]['eval_rows']:,} ; boundary rows: {rows[0]['boundary_rows']:,})")

    # ---- bootstrap CIs on boundary accuracy, ARF & others vs batch RF (raw) ----
    yt,_,bm=_eval_preds(y, prev_all, efi)
    evalpred_eval={k:v for k,v in evalpred.items()}
    boundary_bootstrap(yt, {k:evalpred[k][np.arange(efi,n)] for k in evalpred}, bm,
                       base_key="batch RF (raw)")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "TONIOT_online_learners.csv"),index=False)
    # also dump per-row eval predictions so CIs can be recomputed offline
    ev=np.arange(efi,n)
    perrow=pd.DataFrame({"y_true":y[ev], **{k:evalpred[k][ev] for k in evalpred}})
    perrow.to_csv(os.path.join(OUT_DIR, "TONIOT_online_learners_perrow_predictions.csv"),index=False)
    print(f"Saved TONIOT_online_learners.csv + TONIOT_online_learners_perrow_predictions.csv ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    run()
