# ============================================================================
# CLASS COMPOSITION OF THE ROWS USED  (Table 3)
#
# Reproduces the row selection of the experiment loaders and reports attack/normal counts:
#   - TON-IoT: exact duplicates removed on the raw rows (all columns), as in the main loader.
#   - NF-BoT-IoT-v3 / NF-UNSW-NB15-v3: most recent ~500,000 rows selected by a timestamp
#     threshold, then exact duplicates removed on all columns except the IPv4 address
#     columns, as in load_nfv3().
# The NF-v3 files are streamed in chunks.
# ============================================================================
import numpy as np, pandas as pd

TON_PATH  = "/kaggle/input/datasets/fadiabuzwayed/ton-iot-train-test-network/TON_IoT_Train_Test_Network.csv"
BOT_PATH  = "/kaggle/input/datasets/ndayisabae/nf-bot-iot-v3/NF-BoT-IoT-v3.csv"
UNSW_PATH = "/kaggle/input/datasets/ndayisabae/nf-unsw-nb15-v3/NF-UNSW-NB15-v3.csv"
NF_TAIL   = 500_000
NF_TS     = "FLOW_START_MILLISECONDS"
NF_LABEL  = "Label"

def ton_counts():
    df = pd.read_csv(TON_PATH, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    obj = df.select_dtypes(include="object").columns
    if len(obj) > 0:
        df[obj] = df[obj].apply(lambda s: s.astype(str).str.strip()).replace({"-": "n/a"})
    df = df.drop_duplicates().reset_index(drop=True)   # full-frame dedup (all columns)
    y = df["label"].astype(int).to_numpy()
    bc = np.bincount(y, minlength=2)
    return len(y), int(bc[1]), int(bc[0])

SKIP_STR  = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

def nfv3_counts(path, chunksize=1_000_000):
    # pass 1: timestamps only -> threshold keeping the most recent NF_TAIL rows
    ts_all=[]
    for chunk in pd.read_csv(path, usecols=[NF_TS], chunksize=chunksize, low_memory=False):
        chunk.columns=[c.strip() for c in chunk.columns]
        ts_all.append(pd.to_numeric(chunk[NF_TS], errors="coerce").to_numpy(np.float64))
    ts_all=np.nan_to_num(np.concatenate(ts_all), nan=0.0); n=ts_all.size
    thr=-np.inf if n<=NF_TAIL else np.partition(ts_all, n-NF_TAIL)[n-NF_TAIL]
    del ts_all
    # pass 2: tail rows with the same columns as the loader (IPv4 strings skipped), then de-duplicate
    cols=[c.strip() for c in pd.read_csv(path, nrows=0).columns]
    usecols=[c for c in cols if c not in SKIP_STR]
    keep=[]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk.columns=[c.strip() for c in chunk.columns]
        t=pd.to_numeric(chunk[NF_TS], errors="coerce").fillna(0.0).to_numpy()
        keep.append(chunk[t>=thr])
    df=pd.concat(keep, ignore_index=True).drop_duplicates().reset_index(drop=True)
    yv=df[NF_LABEL].astype(int).to_numpy()
    return len(yv), int((yv==1).sum()), int((yv==0).sum())

print("="*70)
print(f"{'Dataset':<20}{'Rows':>11}{'Attack':>12}{'Normal':>12}{'Attack%':>10}")
print("="*70)
rows=[]
tt,ta,tn = ton_counts(); rows.append(("TON-IoT",tt,ta,tn))
print(f"{'TON-IoT':<20}{tt:>11,}{ta:>12,}{tn:>12,}{100*ta/tt:>9.2f}%", flush=True)
for name,path in [("NF-BoT-IoT-v3",BOT_PATH),("NF-UNSW-NB15-v3",UNSW_PATH)]:
    k,a,nn = nfv3_counts(path); rows.append((name,k,a,nn))
    print(f"{name:<20}{k:>11,}{a:>12,}{nn:>12,}{100*a/k:>9.2f}%", flush=True)
print("="*70)
print("\nClass composition of the rows used (Table 3):")
for name,tot,att,nor in rows:
    print(f"  {name} ({tot:,}): {att:,} attack / {nor:,} normal ({100*att/tot:.2f}%)")
print("\nExpected: TON-IoT 449,972; NF-BoT-IoT-v3 500,141; NF-UNSW-NB15-v3 494,732 rows.")
