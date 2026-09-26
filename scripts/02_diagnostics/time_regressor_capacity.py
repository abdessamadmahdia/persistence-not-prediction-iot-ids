# ============================================================================
# TIME-REGRESSOR CAPACITY  (Section 5.2, Table 6)
#
# Replaces the quadratic time basis of the class-conditional regressors with higher-capacity
# regressors, keeping the rest of the pipeline unchanged:
#   - quadratic phi(u) = [u, u^2] + Ridge (default method)
#   - cubic B-spline basis (degree 3, 25 knots, constant extrapolation) + Ridge
#   - RandomForest-on-time (60 trees, depth 10, min leaf 20)
# Reports macro-F1, boundary accuracy, pooled Kappa-Temporal and the test-period R^2 of each
# regressor per class (uniform average over features).
# TON-IoT, 10-fold expanding-window walk-forward protocol, fixed RandomForest probe.
# Output: TONIOT_time_regressor_capacity.csv
# ============================================================================
import time, warnings, functools
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, accuracy_score, r2_score

import os
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


SEED=42; N_FOLDS=10; TEST_BLOCK=0.10
RF_N_TREES=300; RF_MAX_DEPTH=12; RF_MIN_SAMPLES=5
SPLINE_KNOTS=25; SPLINE_DEGREE=3
RFT_TREES=60; RFT_DEPTH=10          # RandomForest-on-time (kept light)
RUN_RF_TIME=True                    # set False to skip the RF-on-time regressor (faster)
DATA_PATH=("/kaggle/input/datasets/fadiabuzwayed/"
           "ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv")

# ---- TON-IoT columns (identical preprocessing to the main experiments) ----
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
    o=np.argsort(ts,kind="mergesort"); X=X.iloc[o].reset_index(drop=True); y=y[o]; ts=ts[o]
    return X.to_numpy(np.float32), y, ts

def make_relationship(X,a,b):
    da,db=X-a,X-b
    return np.hstack([da,db,np.abs(da),np.abs(db),
                      np.linalg.norm(da,axis=1)[:,None],
                      np.linalg.norm(db,axis=1)[:,None]]).astype(np.float32)

def make_rf():
    return RandomForestClassifier(n_estimators=RF_N_TREES,max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES,random_state=SEED,n_jobs=-1,
        class_weight="balanced_subsample",max_features="sqrt")

# ---- time-regressor factories: each returns fit(u,X)->predict(u)->X_hat ----
def fit_quadratic(u_tr, X_tr):
    Phi=np.column_stack([u_tr,u_tr**2]); m=Ridge(alpha=1.0).fit(Phi,X_tr)
    return lambda u: m.predict(np.column_stack([u,u**2]))

def fit_spline(u_tr, X_tr):
    st=SplineTransformer(n_knots=SPLINE_KNOTS,degree=SPLINE_DEGREE,
                         extrapolation="constant").fit(u_tr[:,None])
    m=Ridge(alpha=1.0).fit(st.transform(u_tr[:,None]),X_tr)
    return lambda u: m.predict(st.transform(np.asarray(u)[:,None]))

def fit_rf_time(u_tr, X_tr):
    m=RandomForestRegressor(n_estimators=RFT_TREES,max_depth=RFT_DEPTH,
        min_samples_leaf=20,random_state=SEED,n_jobs=-1).fit(u_tr[:,None],X_tr)
    return lambda u: m.predict(np.asarray(u)[:,None])

REGRESSORS={"quadratic":fit_quadratic, "spline(25 knots)":fit_spline}
if RUN_RF_TIME: REGRESSORS["rf-on-time"]=fit_rf_time

def run():
    t0=time.time(); X,y,ts=load(DATA_PATH); n=len(y)
    print(f"rows={n:,} feat={X.shape[1]} persistence={np.mean(y[1:]==y[:-1])*100:.2f}%")
    conds=["raw X","static"]+list(REGRESSORS.keys())
    F1={c:[] for c in conds}; BD={c:[] for c in conds}
    ACC={c:[] for c in conds}; PACC=[]                       # for pooled kappa_T
    R2={r:{"attack":[],"normal":[]} for r in REGRESSORS}
    starts=np.linspace(0.50,1.0-TEST_BLOCK,N_FOLDS)
    for f,frac in enumerate(starts,1):
        te0=int(round(n*frac)); te1=min(n,int(round(te0+n*TEST_BLOCK)))
        tr=np.zeros(n,bool); te=np.zeros(n,bool); tr[:te0]=True; te[te0:te1]=True
        tri,tei=np.where(tr)[0],np.where(te)[0]
        if (y[tr]==1).sum()==0 or (y[tr]==0).sum()==0: print(f"fold {f}: skip"); continue
        sc=StandardScaler().fit(X[tr]); Xs=sc.transform(X).astype(np.float32)
        # scaled time in [0,1] on TRAIN range (test extrapolates beyond 1)
        tmin,tmax=ts[tr].min(),ts[tr].max(); rng=max(tmax-tmin,1e-9)
        u=((ts-tmin)/rng).astype(np.float64)
        aM=Xs[tr&(y==1)].mean(0); nM=Xs[tr&(y==0)].mean(0)
        yte=y[tei]; prev=y[tei-1]; bm=(yte!=prev)
        persist_acc=accuracy_score(yte,prev); PACC.append(persist_acc)

        # references: raw X and static prototype
        for cond,Z in [("raw X",Xs),
                       ("static",make_relationship(Xs,np.tile(aM,(n,1)),np.tile(nM,(n,1))))]:
            m=make_rf(); m.fit(Z[tri],y[tri]); pr=m.predict(Z[tei])
            F1[cond].append(f1_score(yte,pr,average="macro",zero_division=0)*100)
            ACC[cond].append(accuracy_score(yte,pr))
            BD[cond].append(accuracy_score(yte[bm],pr[bm])*100 if bm.any() else np.nan)

        # each time-regressor: fit C_A,C_N on train class rows; predict prototypes
        uA,XA=u[tr&(y==1)],Xs[tr&(y==1)]; uN,XN=u[tr&(y==0)],Xs[tr&(y==0)]
        for rname,fit in REGRESSORS.items():
            cA=fit(uA,XA); cN=fit(uN,XN)
            a_all=cA(u).astype(np.float32); b_all=cN(u).astype(np.float32)
            # out-of-sample R^2: predict class feature-state on TEST rows of that class
            teA=tei[yte==1]; teN=tei[yte==0]
            if len(teA)>1: R2[rname]["attack"].append(r2_score(Xs[teA],a_all[teA]))
            if len(teN)>1: R2[rname]["normal"].append(r2_score(Xs[teN],b_all[teN]))
            Z=make_relationship(Xs,a_all,b_all)
            m=make_rf(); m.fit(Z[tri],y[tri]); pr=m.predict(Z[tei])
            F1[rname].append(f1_score(yte,pr,average="macro",zero_division=0)*100)
            ACC[rname].append(accuracy_score(yte,pr))
            BD[rname].append(accuracy_score(yte[bm],pr[bm])*100 if bm.any() else np.nan)
        print(f"fold {f}/{N_FOLDS} done ({time.time()-t0:.0f}s)")

    pa=np.mean(PACC)
    def kt(c): 
        a=np.mean(ACC[c]); return (a-pa)/(1-pa) if pa<1 else float('nan')
    print("\n=== NONPARAMETRIC TIME-REGRESSOR TEST (TON-IoT, 10 folds) ===")
    print(f"{'condition':>18}{'macroF1':>12}{'boundary':>12}{'kappaT':>10}")
    for c in conds:
        print(f"{c:>18}{np.mean(F1[c]):>8.2f}\u00B1{np.std(F1[c]):>4.1f}"
              f"{np.nanmean(BD[c]):>12.2f}{kt(c):>10.2f}")
    print(f"(persistence W=1 accuracy = {pa*100:.2f}%)")

    print("\n=== OUT-OF-SAMPLE R^2 OF THE TIME-REGRESSORS (test regime; <=0 = no trend) ===")
    print(f"{'regressor':>18}{'attack R^2':>14}{'normal R^2':>14}")
    for r in REGRESSORS:
        aR=np.mean(R2[r]['attack']) if R2[r]['attack'] else float('nan')
        nR=np.mean(R2[r]['normal']) if R2[r]['normal'] else float('nan')
        print(f"{r:>18}{aR:>14.3f}{nR:>14.3f}")
    print(">> R^2 is a uniform average over features; features with near-zero test-period "
          "variance can make it very large in magnitude.")

    rows=[]
    for c in conds:
        rows.append(dict(condition=c,macro_f1=np.mean(F1[c]),macro_f1_std=np.std(F1[c]),
                         boundary_acc=np.nanmean(BD[c]),kappa_temporal=kt(c),
                         attack_r2=(np.mean(R2[c]['attack']) if c in R2 and R2[c]['attack'] else np.nan),
                         normal_r2=(np.mean(R2[c]['normal']) if c in R2 and R2[c]['normal'] else np.nan)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "TONIOT_time_regressor_capacity.csv"),index=False)
    print(f"\nSaved TONIOT_time_regressor_capacity.csv ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    run()
