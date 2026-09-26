# ============================================================================
# TRANSITION-UPWEIGHTED TRAINING  (Section 5.5, Table 11)
#
# Up-weights class-transition training rows (y[j] != y[j-1]) by 1, 5, 20 and 100 and reports
# overall macro-F1, boundary accuracy and per-class boundary precision/recall.
# TON-IoT, 10-fold expanding-window walk-forward protocol, fixed RandomForest probe.
# Output: TONIOT_transition_upweighting.csv
# ============================================================================
import time, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score)

import os
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

SEED=42; N_FOLDS=10; TEST_BLOCK=0.10
RF_N_TREES=300; RF_MAX_DEPTH=12; RF_MIN_SAMPLES=5
MULTIPLIERS=[1,5,20,100]        # 1 == plain raw-X baseline; others up-weight transitions
DATA_PATH=("/kaggle/input/datasets/fadiabuzwayed/"
           "ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv")

# ---- columns (TON-IoT) ----
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
    o=np.argsort(ts,kind="mergesort"); X=X.iloc[o].reset_index(drop=True); y,ts=y[o],ts[o]
    return X.to_numpy(np.float32),y,ts

def make_rf():
    return RandomForestClassifier(n_estimators=RF_N_TREES,max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES,random_state=SEED,n_jobs=-1,
        class_weight="balanced_subsample",max_features="sqrt")

def run():
    t0=time.time(); X,y,ts=load(DATA_PATH); n=len(y)
    print(f"rows={n:,} feat={X.shape[1]} persistence={np.mean(y[1:]==y[:-1])*100:.2f}%")
    # transition flag in global index space (row j is a transition if y[j]!=y[j-1])
    is_trans=np.zeros(n,bool); is_trans[1:]=y[1:]!=y[:-1]
    res={m:{"f1":[],"bacc":[],"p1":[],"r1":[],"p0":[],"r0":[],"nb":[]} for m in MULTIPLIERS}
    starts=np.linspace(0.50,1.0-TEST_BLOCK,N_FOLDS)
    for f,frac in enumerate(starts,1):
        te0=int(round(n*frac)); te1=min(n,int(round(te0+n*TEST_BLOCK)))
        tr=np.zeros(n,bool);te=np.zeros(n,bool);tr[:te0]=True;te[te0:te1]=True
        tri,tei=np.where(tr)[0],np.where(te)[0]
        if (y[tr]==1).sum()==0 or (y[tr]==0).sum()==0:
            print(f"fold {f}: skipped"); continue
        sc=StandardScaler().fit(X[tr]); Xs=sc.transform(X).astype(np.float32)
        yte=y[tei]; prev=y[tei-1]; bm=(yte!=prev)
        for m in MULTIPLIERS:
            w=np.ones(len(tri),dtype=float)
            w[is_trans[tri]]=float(m)               # up-weight transition rows in TRAIN
            clf=make_rf(); clf.fit(Xs[tri],y[tri],sample_weight=w); pr=clf.predict(Xs[tei])
            d=res[m]; d["f1"].append(f1_score(yte,pr,average="macro",zero_division=0)*100)
            d["nb"].append(int(bm.sum()))
            if bm.sum()>0:
                d["bacc"].append(accuracy_score(yte[bm],pr[bm])*100)
                if len(np.unique(yte[bm]))>1:
                    d["p1"].append(precision_score(yte[bm],pr[bm],pos_label=1,zero_division=0)*100)
                    d["r1"].append(recall_score(yte[bm],pr[bm],pos_label=1,zero_division=0)*100)
                    d["p0"].append(precision_score(yte[bm],pr[bm],pos_label=0,zero_division=0)*100)
                    d["r0"].append(recall_score(yte[bm],pr[bm],pos_label=0,zero_division=0)*100)
        print(f"fold {f}/{N_FOLDS} boundary_rows={int(bm.sum())} ({time.time()-t0:.0f}s)")
    print("\n=== TRANSITION-UPWEIGHTED BASELINE (TON-IoT, mean over folds) ===")
    print(f"{'trans_weight':>12}{'macroF1':>10}{'bdry_acc':>10}{'P(atk)':>9}{'R(atk)':>9}{'P(nrm)':>9}{'R(nrm)':>9}")
    def m(d,k): return np.mean(d[k]) if len(d[k]) else float('nan')
    rows=[]
    for mm in MULTIPLIERS:
        d=res[mm]; tag="x%d%s"%(mm," (raw)" if mm==1 else "")
        print(f"{tag:>12}{m(d,'f1'):>10.2f}{m(d,'bacc'):>10.2f}{m(d,'p1'):>9.1f}{m(d,'r1'):>9.1f}{m(d,'p0'):>9.1f}{m(d,'r0'):>9.1f}")
        rows.append(dict(trans_weight=mm,macro_f1=m(d,'f1'),boundary_acc=m(d,'bacc'),
                         P_attack=m(d,'p1'),R_attack=m(d,'r1'),P_normal=m(d,'p0'),R_normal=m(d,'r0')))
    base_bd=m(res[1],'bacc'); best=max(MULTIPLIERS[1:],key=lambda mm:m(res[mm],'bacc'))
    print(f"\nraw boundary_acc={base_bd:.2f} ; best up-weighted (x{best}) boundary_acc={m(res[best],'bacc'):.2f}")
    print(">> "+("Up-weighting raises boundary accuracy by more than 3 points." if m(res[best],'bacc')>base_bd+3.0
                 else "Up-weighting does not raise boundary accuracy by more than 3 points."))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "TONIOT_transition_upweighting.csv"),index=False)
    print(f"Saved TONIOT_transition_upweighting.csv  ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    run()
