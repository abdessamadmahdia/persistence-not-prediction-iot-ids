# ============================================================================
# CLASS-CONDITIONAL FORECASTING METHOD  (Section 5.1, Table 5; Section 5.2 MAE)
#
# TON-IoT, 77 features, 10-fold expanding-window walk-forward protocol (tau = 0.10,
# phi_k = 0.50 + 0.40(k-1)/9).
# Probes: kNN (k = 5) and RandomForest (300 trees, depth 12, min leaf 5).
# Conditions: raw X | static prototype | learned (phi(u) = [u, u^2]) | shuffled.
# Also reports per-fold MCC and the test-period R^2 / MAE of the learned regressors.
# Output: TONIOT_forecasting_method_10fold.csv
# ============================================================================
import time, warnings, functools
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, r2_score, mean_absolute_error

import os
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


SEED=42; N_FOLDS=10; TEST_BLOCK=0.10
RF_TREES=300; RF_DEPTH=12; RF_LEAF=5; KNN_K=5
DATA=("/kaggle/input/datasets/fadiabuzwayed/"
      "ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv")

ID_COLS=["src_ip","src_port","dst_ip","dst_port"]; TS="ts"; LB="label"; LM="type"
MISS={"-","n/a",""," ","(empty)","nan","none"}
ONEHOT=["proto","service","conn_state","http_method","http_version","dns_AA","dns_RD",
        "dns_RA","dns_rejected","ssl_resumed","ssl_established"]
PRESENCE=["dns_query","ssl_version","ssl_cipher","ssl_subject","ssl_issuer","http_uri",
          "http_referrer","http_user_agent","http_orig_mime_types","http_resp_mime_types",
          "weird_name","weird_addl","weird_notice","http_trans_depth"]
CONT=["duration","src_bytes","dst_bytes","missed_bytes","src_pkts","src_ip_bytes","dst_pkts",
      "dst_ip_bytes","dns_qclass","dns_qtype","dns_rcode","http_request_body_len",
      "http_response_body_len","http_status_code"]

def load():
    df=pd.read_csv(DATA,low_memory=False); df.columns=[c.strip() for c in df.columns]
    o=df.select_dtypes(include="object").columns
    if len(o): df[o]=df[o].apply(lambda s:s.astype(str).str.strip()).replace({"-":"n/a"})
    df=df.drop_duplicates().reset_index(drop=True)
    y=df[LB].astype(int).to_numpy()
    ts=pd.to_numeric(df[TS],errors="coerce").ffill().fillna(0).to_numpy(np.float64)
    rm=[c for c in [TS,LB,LM,*ID_COLS] if c in df.columns]
    f=df.drop(columns=rm); pres=set(f.columns); blocks=[]
    nc=[c for c in CONT if c in pres]
    if nc: blocks.append(f[nc].apply(pd.to_numeric,errors="coerce").fillna(0.0))
    pc=[c for c in PRESENCE if c in pres]
    if pc:
        pr=f[pc].apply(lambda s:(~s.astype(str).str.strip().str.lower().isin(MISS)).astype(int))
        pr.columns=[f"has_{c}" for c in pc]; blocks.append(pr)
    oc=[c for c in ONEHOT if c in pres]
    if oc: blocks.append(pd.get_dummies(f[oc].astype(str),columns=oc,prefix=oc,dtype=np.int8))
    handled=set(nc)|set(pc)|set(oc); rem=[c for c in f.columns if c not in handled]
    if rem: blocks.append(f[rem].apply(pd.to_numeric,errors="coerce").fillna(0.0))
    X=pd.concat(blocks,axis=1); X=X.loc[:,~X.columns.duplicated()].astype(float)
    idx=np.argsort(ts,kind="mergesort")
    return X.to_numpy(np.float32)[idx], y[idx], ts[idx]

def rel(X,a,b):
    da,db=X-a,X-b
    return np.hstack([da,db,np.abs(da),np.abs(db),
        np.linalg.norm(da,axis=1)[:,None],np.linalg.norm(db,axis=1)[:,None]]).astype(np.float32)

def probes():
    return {"kNN":KNeighborsClassifier(n_neighbors=KNN_K,n_jobs=-1),
            "RF":RandomForestClassifier(n_estimators=RF_TREES,max_depth=RF_DEPTH,
                 min_samples_leaf=RF_LEAF,random_state=SEED,n_jobs=-1,
                 class_weight="balanced_subsample",max_features="sqrt")}

def run():
    t0=time.time(); X,y,ts=load(); n=len(y)
    print(f"rows={n:,} feat={X.shape[1]} attack={100*y.mean():.2f}% "
          f"persistence={100*np.mean(y[1:]==y[:-1]):.2f}%")
    conds=["raw X","static","learned","shuffled"]
    F1={p:{c:[] for c in conds} for p in ["kNN","RF"]}
    MCC={p:{c:[] for c in conds} for p in ["kNN","RF"]}
    R2={"attack_learned":[], "attack_static":[], "normal_learned":[], "normal_static":[]}
    MAE={"attack_learned":[], "attack_static":[], "normal_learned":[], "normal_static":[]}
    rng=np.random.default_rng(SEED)
    for f,frac in enumerate(np.linspace(0.50,1.0-TEST_BLOCK,N_FOLDS),1):
        te0=int(round(n*frac)); te1=min(n,int(round(te0+n*TEST_BLOCK)))
        tr=np.zeros(n,bool); te=np.zeros(n,bool); tr[:te0]=True; te[te0:te1]=True
        tri,tei=np.where(tr)[0],np.where(te)[0]
        if (y[tr]==1).sum()==0 or (y[tr]==0).sum()==0: print(f"fold {f}: skip"); continue
        sc=StandardScaler().fit(X[tr]); Xs=sc.transform(X).astype(np.float32)
        tmin,tmax=ts[tr].min(),ts[tr].max(); u=((ts-tmin)/max(tmax-tmin,1e-9)).astype(np.float64)
        aM=Xs[tr&(y==1)].mean(0); nM=Xs[tr&(y==0)].mean(0)
        # learned time-regressors  phi(t)=[u,u^2]
        def fit(mask):
            P=np.column_stack([u[mask],u[mask]**2])
            m=Ridge(alpha=1.0).fit(P,Xs[mask])
            return m.predict(np.column_stack([u,u**2])).astype(np.float32)
        aL=fit(tr&(y==1)); bL=fit(tr&(y==0))
        # Section 5.2 diagnostics on the test period
        teA=tei[y[tei]==1]; teN=tei[y[tei]==0]
        if len(teA)>1:
            R2["attack_learned"].append(r2_score(Xs[teA],aL[teA]))
            R2["attack_static"].append(r2_score(Xs[teA],np.tile(aM,(len(teA),1))))
            MAE["attack_learned"].append(mean_absolute_error(Xs[teA],aL[teA]))
            MAE["attack_static"].append(mean_absolute_error(Xs[teA],np.tile(aM,(len(teA),1))))
        if len(teN)>1:
            R2["normal_learned"].append(r2_score(Xs[teN],bL[teN]))
            R2["normal_static"].append(r2_score(Xs[teN],np.tile(nM,(len(teN),1))))
            MAE["normal_learned"].append(mean_absolute_error(Xs[teN],bL[teN]))
            MAE["normal_static"].append(mean_absolute_error(Xs[teN],np.tile(nM,(len(teN),1))))
        # shuffled control: permute prototype PAIRS across rows (within train / within test)
        aS=aL.copy(); bS=bL.copy()
        for m in (tri,tei):
            p=rng.permutation(len(m)); aS[m]=aL[m][p]; bS[m]=bL[m][p]
        Z={"raw X":Xs,
           "static":rel(Xs,np.tile(aM,(n,1)),np.tile(nM,(n,1))),
           "learned":rel(Xs,aL,bL),
           "shuffled":rel(Xs,aS,bS)}
        line=[]
        for pname,clf in probes().items():
            for c in conds:
                m=clf.__class__(**clf.get_params()); m.fit(Z[c][tri],y[tri]); pr=m.predict(Z[c][tei])
                f1=f1_score(y[tei],pr,average="macro",zero_division=0)*100
                mc=matthews_corrcoef(y[tei],pr)
                F1[pname][c].append(f1); MCC[pname][c].append(mc)
                if pname=="RF": line.append(f"{c}:{f1:.1f}(mcc{mc:.2f})")
        print(f"fold {f:>2}/{N_FOLDS}  RF-> "+"  ".join(line)+f"   [{time.time()-t0:.0f}s]")

    print("\n=== 5.1  TON-IoT, 10-fold walk-forward, full 77 features ===")
    print(f"{'Condition':>22}{'kNN macro-F1':>20}{'RF macro-F1':>20}{'RF MCC':>10}")
    for c in conds:
        print(f"{c:>22}{np.mean(F1['kNN'][c]):>13.2f}\u00B1{np.std(F1['kNN'][c]):<6.2f}"
              f"{np.mean(F1['RF'][c]):>13.2f}\u00B1{np.std(F1['RF'][c]):<6.2f}"
              f"{np.mean(MCC['RF'][c]):>10.3f}")
    print("\n>> An MCC close to 0 indicates near-constant predictions; read macro-F1 together with MCC.")

    print("\n=== 5.2  Feature-on-time prediction quality (test regime, 10 folds) ===")
    print(f"{'':>18}{'R^2 learned':>14}{'R^2 static':>14}{'MAE learned':>14}{'MAE static':>14}")
    for cl in ["attack","normal"]:
        print(f"{cl:>18}{np.mean(R2[cl+'_learned']):>14.3f}{np.mean(R2[cl+'_static']):>14.3f}"
              f"{np.mean(MAE[cl+'_learned']):>14.4f}{np.mean(MAE[cl+'_static']):>14.4f}")

    pd.DataFrame([{"condition":c,
                   "knn_f1":np.mean(F1['kNN'][c]),"knn_sd":np.std(F1['kNN'][c]),
                   "rf_f1":np.mean(F1['RF'][c]),"rf_sd":np.std(F1['RF'][c]),
                   "rf_mcc":np.mean(MCC['RF'][c])} for c in conds]
                ).to_csv(os.path.join(OUT_DIR, "TONIOT_forecasting_method_10fold.csv"),index=False)
    print(f"\nSaved TONIOT_forecasting_method_10fold.csv  ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    run()
