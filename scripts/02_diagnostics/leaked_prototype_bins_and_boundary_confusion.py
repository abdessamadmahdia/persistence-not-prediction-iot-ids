# ============================================================================
# LEAKED-PROTOTYPE BIN SETTINGS AND BOUNDARY CONFUSION  (Section 5.3, Table 8; Section 5.5, Table 10)
#
# (a) Leaked prototype ("oracle") with B in {50, 100, 200} timestamp-quantile bins computed over
#     the whole series (about B/10 bins per test block); Table 8.
# (b) Per-class (attack/normal) precision and recall on class-transition rows for raw X,
#     the leaked prototype (B = 100) and recency W = 50; Table 10.
# TON-IoT, 10-fold expanding-window walk-forward protocol, fixed RandomForest probe;
# loader, prototypes and relationship representation as in prototype_ladder_10fold.py.
# Outputs: TONIOT_leaked_prototype_bins.csv, TONIOT_boundary_confusion.csv
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
ORACLE_BINS=[50,100,200]        # bin settings (Table 8)
RECENCY_W=50                    # for the boundary-confusion recency condition
rng=np.random.default_rng(SEED)

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

def rel(X,a,b):
    da,db=X-a,X-b
    return np.hstack([da,db,np.abs(da),np.abs(db),
                      np.linalg.norm(da,axis=1)[:,None],np.linalg.norm(db,axis=1)[:,None]]).astype(np.float32)

def oracle(Xs,y,ts,B):
    n,d=Xs.shape; e=np.quantile(ts,np.linspace(0,1,B+1)); e[0]-=1;e[-1]+=1
    bid=np.clip(np.searchsorted(e,ts,side="right")-1,0,B-1)
    ga=Xs[y==1].mean(0) if (y==1).any() else Xs.mean(0)
    gn=Xs[y==0].mean(0) if (y==0).any() else Xs.mean(0)
    aM=np.tile(ga,(B,1)).astype(np.float32); nM=np.tile(gn,(B,1)).astype(np.float32)
    for k in range(B):
        m=bid==k; ma,mn=m&(y==1),m&(y==0)
        if ma.any():aM[k]=Xs[ma].mean(0)
        if mn.any():nM[k]=Xs[mn].mean(0)
    return aM[bid],nM[bid]

def recency(Xs,y,W,fa,fb):
    n,d=Xs.shape; yA=(y==1).astype(np.float64);yN=(y==0).astype(np.float64)
    csA=np.vstack([np.zeros((1,d)),np.cumsum(Xs*yA[:,None],0)]);csN=np.vstack([np.zeros((1,d)),np.cumsum(Xs*yN[:,None],0)])
    cA=np.concatenate([[0],np.cumsum(yA)]);cN=np.concatenate([[0],np.cumsum(yN)])
    hi=np.arange(n);lo=np.maximum(0,hi-W)
    def bd(cs,cnt,fb):
        sw=cs[hi]-cs[lo];cw=cnt[hi]-cnt[lo];se=cs[hi];ce=cnt[hi]
        o=np.tile(fb,(n,1)).astype(np.float64);mw=cw>0;o[mw]=sw[mw]/cw[mw][:,None]
        me=(~mw)&(ce>0);o[me]=se[me]/ce[me][:,None];return o.astype(np.float32)
    return bd(csA,cA,fa),bd(csN,cN,fb)

def rf(): return RandomForestClassifier(n_estimators=RF_N_TREES,max_depth=RF_MAX_DEPTH,
    min_samples_leaf=RF_MIN_SAMPLES,random_state=SEED,n_jobs=-1,
    class_weight="balanced_subsample",max_features="sqrt")

def run():
    t0=time.time(); X,y,ts=load(DATA_PATH); n=len(y)
    print(f"rows={n:,} feat={X.shape[1]} persistence={np.mean(y[1:]==y[:-1])*100:.2f}%")
    binF1={B:[] for B in ORACLE_BINS}; binBd={B:[] for B in ORACLE_BINS}
    # boundary confusion: per-class precision/recall at boundaries
    conf={c:{"p1":[],"r1":[],"p0":[],"r0":[]} for c in ["raw X","oracle@100","recency_W%d"%RECENCY_W]}
    starts=np.linspace(0.50,1.0-TEST_BLOCK,N_FOLDS)
    for f,frac in enumerate(starts,1):
        te0=int(round(n*frac)); te1=min(n,int(round(te0+n*TEST_BLOCK)))
        tr=np.zeros(n,bool);te=np.zeros(n,bool);tr[:te0]=True;te[te0:te1]=True
        tri,tei=np.where(tr)[0],np.where(te)[0]
        sc=StandardScaler().fit(X[tr]);Xs=sc.transform(X).astype(np.float32)
        aM=Xs[tr&(y==1)].mean(0).astype(np.float32);nM=Xs[tr&(y==0)].mean(0).astype(np.float32)
        yte=y[tei];prev=y[tei-1];bm=(yte!=prev)   # boundary rows
        # ---- leaked prototype at each bin setting ----
        for B in ORACLE_BINS:
            a,b=oracle(Xs,y,ts,B);Z=rel(Xs,a,b)
            m=rf();m.fit(Z[tri],y[tri]);pr=m.predict(Z[tei])
            binF1[B].append(f1_score(yte,pr,average="macro",zero_division=0)*100)
            binBd[B].append(accuracy_score(yte[bm],pr[bm])*100 if bm.any() else np.nan)
            if B==100: pred_oracle=pr
        # ---- raw X and recency for boundary confusion ----
        mr=rf();mr.fit(Xs[tri],y[tri]);pred_raw=mr.predict(Xs[tei])
        ra,rb=recency(Xs,y,RECENCY_W,aM,nM);Zr=rel(Xs,ra,rb)
        mc=rf();mc.fit(Zr[tri],y[tri]);pred_rec=mc.predict(Zr[tei])
        # ---- per-class precision/recall on boundary rows ----
        for cname,pred in [("raw X",pred_raw),("oracle@100",pred_oracle),
                           ("recency_W%d"%RECENCY_W,pred_rec)]:
            if bm.sum()>0 and len(np.unique(yte[bm]))>1:
                yb,pb=yte[bm],pred[bm]
                conf[cname]["p1"].append(precision_score(yb,pb,pos_label=1,zero_division=0)*100)
                conf[cname]["r1"].append(recall_score(yb,pb,pos_label=1,zero_division=0)*100)
                conf[cname]["p0"].append(precision_score(yb,pb,pos_label=0,zero_division=0)*100)
                conf[cname]["r0"].append(recall_score(yb,pb,pos_label=0,zero_division=0)*100)
        print(f"fold {f}/{N_FOLDS} done ({time.time()-t0:.0f}s)  boundary_rows={int(bm.sum())}")

    print("\n=== LEAKED-PROTOTYPE BIN SETTINGS (mean over folds) ===")
    print(f"{'B (bins)':>10}{'oracle F1':>12}{'oracle boundary_acc':>22}")
    for B in ORACLE_BINS:
        print(f"{B:>10}{np.mean(binF1[B]):>12.2f}{np.nanmean(binBd[B]):>22.2f}")
    print("(interpretation: overall F1 may rise with B, but boundary_acc stays "
          "below raw X at each tested bin setting.)")

    print("\n=== PER-CLASS PRECISION/RECALL AT BOUNDARIES (mean over folds) ===")
    print(f"{'condition':>16}{'P(attack)':>11}{'R(attack)':>11}{'P(normal)':>11}{'R(normal)':>11}")
    for c in conf:
        d=conf[c]
        print(f"{c:>16}{np.mean(d['p1']):>11.1f}{np.mean(d['r1']):>11.1f}"
              f"{np.mean(d['p0']):>11.1f}{np.mean(d['r0']):>11.1f}")
    print("(reading: compare R(normal) vs R(attack) at boundaries to see whether "
          "missed regime-changes are false negatives on the newly-arriving class.)")

    pd.DataFrame([{"B":B,"oracle_f1":np.mean(binF1[B]),"oracle_boundary_acc":np.nanmean(binBd[B])}
                  for B in ORACLE_BINS]).to_csv(os.path.join(OUT_DIR, "TONIOT_leaked_prototype_bins.csv"),index=False)
    pd.DataFrame([{"condition":c,**{k:np.mean(v) for k,v in conf[c].items()}} for c in conf]
                 ).to_csv(os.path.join(OUT_DIR, "TONIOT_boundary_confusion.csv"),index=False)
    print(f"\nSaved TONIOT_leaked_prototype_bins.csv and TONIOT_boundary_confusion.csv  ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    run()
