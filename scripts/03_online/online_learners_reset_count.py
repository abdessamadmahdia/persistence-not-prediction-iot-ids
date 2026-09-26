# ============================================================================
# ONLINE LEARNERS, SUPPLEMENTARY RUN WITH RESET COUNTING  (Section 5.6)
#
# Same learners and prequential protocol as online_learners_prequential.py, with a counter for
# the ADWIN-triggered resets of the Hoeffding tree (ADWIN is re-initialised after each reset).
# Differences from the main run: features are not standardised and no batch model is fitted.
# ARF internal drift detections are not counted.
# River is installed with --no-deps so that it does not upgrade NumPy/SciPy/scikit-learn.
# Output: TONIOT_online_learners_reset_count.csv
# ============================================================================

import os
import time
import warnings
import subprocess
import sys
import importlib

warnings.filterwarnings("ignore")


# ============================================================================
# 1. INSTALL / IMPORT RIVER SAFELY
# ============================================================================

try:
    import river
except ImportError:
    print("River not found. Installing River without dependencies...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "river"
    ])
    import river


# ============================================================================
# 2. IMPORTS
# ============================================================================

import numpy as np
import pandas as pd


# River API compatibility
try:
    from river.forest import ARFClassifier
except Exception:
    try:
        from river.ensemble import AdaptiveRandomForestClassifier as ARFClassifier
    except Exception as e:
        raise ImportError(
            "Could not import Adaptive Random Forest from the installed "
            f"River version ({river.__version__})."
        ) from e

from river.tree import HoeffdingTreeClassifier
from river.drift import ADWIN


# ============================================================================
# 3. ENVIRONMENT CHECK
# ============================================================================

print("=" * 78)
print("ENVIRONMENT")
print("=" * 78)

print("Python :", sys.version.split()[0])
print("NumPy  :", np.__version__)
print("Pandas :", pd.__version__)
print("River  :", river.__version__)

print("River / NumPy import: OK")
print()


# ============================================================================
# 4. CONFIGURATION
# ============================================================================

SEED = 42

# Evaluate only the chronological second half.
EVAL_FROM = 0.50

# None = full dataset.
# Example: 200_000 = use only the final 200k chronological rows.
SUBSAMPLE = None

# Number of ARF trees.
ARF_MODELS = 10

DATA_PATH = (
    "/kaggle/input/datasets/fadiabuzwayed/"
    "ton-iot-train-test-network/"
    "TON_IoT_Train_Test_Network.csv"
)

OUTPUT_PATH = os.path.join(os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"),
                           "TONIOT_online_learners_reset_count.csv")


# ============================================================================
# 5. TON-IoT COLUMNS
# ============================================================================

ID_COLS = [
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port"
]

TS_COL = "ts"

LABEL_BIN = "label"
LABEL_MULTI = "type"

MISS = {
    "-",
    "n/a",
    "",
    " ",
    "(empty)",
    "nan",
    "none"
}

ONEHOT = [
    "proto",
    "service",
    "conn_state",
    "http_method",
    "http_version",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_resumed",
    "ssl_established"
]

PRESENCE = [
    "dns_query",
    "ssl_version",
    "ssl_cipher",
    "ssl_subject",
    "ssl_issuer",
    "http_uri",
    "http_referrer",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",
    "weird_addl",
    "weird_notice",
    "http_trans_depth"
]

CONTINUOUS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code"
]


# ============================================================================
# 6. LOAD + PREPROCESS
# ============================================================================

def load(path):

    print("=" * 78)
    print("LOADING DATA")
    print("=" * 78)

    t0 = time.time()

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print(f"Raw rows: {len(df):,}")
    print(f"Raw columns: {len(df.columns):,}")

    # Clean column names
    df.columns = [
        c.strip()
        for c in df.columns
    ]

    # Clean object columns
    obj = df.select_dtypes(
        include="object"
    ).columns

    if len(obj):

        df[obj] = (
            df[obj]
            .apply(
                lambda s:
                s.astype(str)
                .str.strip()
            )
            .replace(
                {
                    "-": "n/a"
                }
            )
        )

    # Remove duplicate observations
    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Duplicates removed: "
        f"{before - len(df):,}"
    )

    # ------------------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------------------

    y = (
        df[LABEL_BIN]
        .astype(int)
        .to_numpy()
    )

    # ------------------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------------------

    ts = (
        pd.to_numeric(
            df[TS_COL],
            errors="coerce"
        )
        .ffill()
        .fillna(0)
        .to_numpy(
            np.float64
        )
    )

    # ------------------------------------------------------------------------
    # REMOVE NON-FEATURE COLUMNS
    # ------------------------------------------------------------------------

    remove_cols = [
        c
        for c in [
            TS_COL,
            LABEL_BIN,
            LABEL_MULTI,
            *ID_COLS
        ]
        if c in df.columns
    ]

    feats = df.drop(
        columns=remove_cols
    )

    present = set(
        feats.columns
    )

    blocks = []

    # ------------------------------------------------------------------------
    # CONTINUOUS FEATURES
    # ------------------------------------------------------------------------

    nc = [
        c
        for c in CONTINUOUS
        if c in present
    ]

    if nc:

        continuous_block = (
            feats[nc]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .fillna(0.0)
        )

        blocks.append(
            continuous_block
        )

    # ------------------------------------------------------------------------
    # PRESENCE FEATURES
    # ------------------------------------------------------------------------

    pc = [
        c
        for c in PRESENCE
        if c in present
    ]

    if pc:

        presence_block = (
            feats[pc]
            .apply(
                lambda s:
                (
                    ~s.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(MISS)
                ).astype(int)
            )
        )

        presence_block.columns = [
            f"has_{c}"
            for c in pc
        ]

        blocks.append(
            presence_block
        )

    # ------------------------------------------------------------------------
    # ONE-HOT CATEGORICAL FEATURES
    # ------------------------------------------------------------------------

    oc = [
        c
        for c in ONEHOT
        if c in present
    ]

    if oc:

        onehot_block = pd.get_dummies(
            feats[oc].astype(str),
            columns=oc,
            prefix=oc,
            dtype=np.int8
        )

        blocks.append(
            onehot_block
        )

    # ------------------------------------------------------------------------
    # ANY REMAINING NUMERIC COLUMNS
    # ------------------------------------------------------------------------

    handled = (
        set(nc)
        | set(pc)
        | set(oc)
    )

    rem = [
        c
        for c in feats.columns
        if c not in handled
    ]

    if rem:

        remaining_block = (
            feats[rem]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .fillna(0.0)
        )

        blocks.append(
            remaining_block
        )

    # ------------------------------------------------------------------------
    # CONCATENATE FEATURES
    # ------------------------------------------------------------------------

    X = pd.concat(
        blocks,
        axis=1
    )

    # Remove duplicate feature names
    X = X.loc[
        :,
        ~X.columns.duplicated()
    ]

    X = X.astype(float)

    # ------------------------------------------------------------------------
    # CHRONOLOGICAL SORT
    # ------------------------------------------------------------------------

    order = np.argsort(
        ts,
        kind="mergesort"
    )

    X = (
        X.iloc[order]
        .reset_index(drop=True)
    )

    y = y[order]

    X = X.to_numpy(
        dtype=np.float32
    )

    print(
        f"Final rows: {len(y):,}"
    )

    print(
        f"Final features: {X.shape[1]:,}"
    )

    print(
        f"Loading/preprocessing time: "
        f"{time.time() - t0:.1f}s"
    )

    print()

    return X, y, list(
        pd.concat(
            blocks,
            axis=1
        ).loc[
            :,
            ~pd.concat(
                blocks,
                axis=1
            ).columns.duplicated()
        ].columns
    )


# ============================================================================
# 7. ARF CREATION
# ============================================================================

def make_arf():

    # River versions have slightly different constructor signatures.
    try:

        return ARFClassifier(
            n_models=ARF_MODELS,
            seed=SEED
        )

    except TypeError:

        try:

            return ARFClassifier(
                n_models=ARF_MODELS,
                random_state=SEED
            )

        except TypeError:

            return ARFClassifier(
                n_models=ARF_MODELS
            )


# ============================================================================
# 8. CONVERT NUMPY ROW TO RIVER DICTIONARY
# ============================================================================

def row_to_dict(row, feat):

    return {
        feat[j]: float(row[j])
        for j in range(len(feat))
    }


# ============================================================================
# 9. PREQUENTIAL EVALUATION
# ============================================================================

def prequential(
    model,
    X,
    y,
    feat,
    adwin_retrain=False
):

    """
    Strict prequential protocol:

        1. Receive x_t
        2. Predict y_t BEFORE seeing y_t
        3. Observe y_t
        4. Learn from (x_t, y_t)

    If adwin_retrain=True:

        - ADWIN monitors the prediction error.
        - When drift is detected:
              * reset the Hoeffding Tree
              * reset ADWIN
    """

    n = len(y)

    preds = np.full(
        n,
        -1,
        dtype=np.int8
    )

    adw = (
        ADWIN()
        if adwin_retrain
        else None
    )

    drift_count = 0

    t0 = time.time()

    for i in range(n):

        # ------------------------------------------------------------
        # Convert current observation to River format
        # ------------------------------------------------------------

        x = row_to_dict(
            X[i],
            feat
        )

        # ------------------------------------------------------------
        # 1. PREDICT BEFORE LEARNING
        # ------------------------------------------------------------

        yp = model.predict_one(x)

        # River returns None when the model has not yet learned enough.
        if yp is None:

            yp = 0

        preds[i] = int(yp)

        # ------------------------------------------------------------
        # 2. OBSERVE TRUE LABEL
        # ------------------------------------------------------------

        true_y = int(y[i])

        # ------------------------------------------------------------
        # 3. LEARN AFTER PREDICTION
        # ------------------------------------------------------------

        model.learn_one(
            x,
            true_y
        )

        # ------------------------------------------------------------
        # 4. ADWIN DRIFT DETECTION
        # ------------------------------------------------------------

        if adw is not None:

            error = (
                0.0
                if yp == true_y
                else 1.0
            )

            adw.update(error)

            if adw.drift_detected:

                drift_count += 1

                # Reset the learner
                model = HoeffdingTreeClassifier()

                # IMPORTANT:
                # Reset ADWIN as well so that the old drift state
                # does not carry over to the new learner.
                adw = ADWIN()

        # ------------------------------------------------------------
        # Progress report
        # ------------------------------------------------------------

        if (i + 1) % 100_000 == 0:

            elapsed = time.time() - t0

            print(
                f"      processed "
                f"{i + 1:,}/{n:,} "
                f"({100*(i+1)/n:.1f}%) "
                f"| {elapsed/60:.1f} min "
                f"| drifts={drift_count}"
            )

    print(
        f"      Finished: "
        f"{n:,} rows "
        f"| drifts={drift_count} "
        f"| {((time.time()-t0)/60):.1f} min"
    )

    return preds, drift_count


# ============================================================================
# 10. MACRO-F1 WITHOUT SKLEARN
# ============================================================================

def binary_macro_f1(y_true, y_pred):

    """
    Binary macro-F1 implementation.

    Avoids scikit-learn completely.
    """

    classes = [
        0,
        1
    ]

    f1_values = []

    for c in classes:

        tp = np.sum(
            (y_true == c)
            &
            (y_pred == c)
        )

        fp = np.sum(
            (y_true != c)
            &
            (y_pred == c)
        )

        fn = np.sum(
            (y_true == c)
            &
            (y_pred != c)
        )

        precision_den = tp + fp
        recall_den = tp + fn

        if precision_den == 0:

            precision = 0.0

        else:

            precision = (
                tp / precision_den
            )

        if recall_den == 0:

            recall = 0.0

        else:

            recall = (
                tp / recall_den
            )

        if precision + recall == 0:

            f1 = 0.0

        else:

            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        f1_values.append(f1)

    return float(
        np.mean(f1_values)
    )


# ============================================================================
# 11. DIAGNOSTICS
# ============================================================================

def diagnostics(
    name,
    y,
    preds,
    eval_from_idx,
    drift_count=0
):

    # ------------------------------------------------------------------------
    # Evaluation region
    # ------------------------------------------------------------------------

    ev = np.arange(
        eval_from_idx,
        len(y)
    )

    yt = y[ev]
    yp = preds[ev]

    # Previous TRUE label.
    # This is the correct reference for W=1 persistence.
    prev = y[ev - 1]

    # ------------------------------------------------------------------------
    # Boundary mask
    # ------------------------------------------------------------------------

    boundary_mask = (
        yt != prev
    )

    within_mask = (
        ~boundary_mask
    )

    # ------------------------------------------------------------------------
    # OVERALL ACCURACY
    # ------------------------------------------------------------------------

    overall_acc = float(
        np.mean(
            yt == yp
        )
    )

    # ------------------------------------------------------------------------
    # MACRO F1
    # ------------------------------------------------------------------------

    macro_f1 = (
        binary_macro_f1(
            yt,
            yp
        )
        * 100
    )

    # ------------------------------------------------------------------------
    # BOUNDARY ACCURACY
    # ------------------------------------------------------------------------

    if boundary_mask.any():

        boundary_acc = (
            np.mean(
                yp[boundary_mask]
                ==
                yt[boundary_mask]
            )
            * 100
        )

    else:

        boundary_acc = np.nan

    # ------------------------------------------------------------------------
    # WITHIN-BLOCK ACCURACY
    # ------------------------------------------------------------------------

    if within_mask.any():

        within_acc = (
            np.mean(
                yp[within_mask]
                ==
                yt[within_mask]
            )
            * 100
        )

    else:

        within_acc = np.nan

    # ------------------------------------------------------------------------
    # PERSISTENCE BASELINE
    # ------------------------------------------------------------------------

    persistence_correct = (
        yt == prev
    )

    persistence_acc = float(
        np.mean(
            persistence_correct
        )
    )

    # ------------------------------------------------------------------------
    # OVERALL KAPPA-TEMPORAL
    #
    # Relative to the W=1 persistence baseline.
    # ------------------------------------------------------------------------

    if persistence_acc < 1.0:

        kappa_temporal = (
            overall_acc
            - persistence_acc
        ) / (
            1.0
            - persistence_acc
        )

    else:

        kappa_temporal = np.nan

    # ------------------------------------------------------------------------
    # BOUNDARY PERSISTENCE ACCURACY
    #
    # Because boundary is defined as:
    #
    #       yt != prev
    #
    # W=1 persistence necessarily gets 0% at boundaries.
    # ------------------------------------------------------------------------

    if boundary_mask.any():

        boundary_persistence_acc = (
            np.mean(
                prev[boundary_mask]
                ==
                yt[boundary_mask]
            )
            * 100
        )

        boundary_delta = (
            boundary_acc
            - boundary_persistence_acc
        )

    else:

        boundary_persistence_acc = np.nan
        boundary_delta = np.nan

    # ------------------------------------------------------------------------
    # WITHIN-BLOCK PERSISTENCE
    # ------------------------------------------------------------------------

    if within_mask.any():

        within_persistence_acc = (
            np.mean(
                prev[within_mask]
                ==
                yt[within_mask]
            )
            * 100
        )

        within_delta = (
            within_acc
            - within_persistence_acc
        )

    else:

        within_persistence_acc = np.nan
        within_delta = np.nan

    # ------------------------------------------------------------------------
    # RETURN
    # ------------------------------------------------------------------------

    return {

        "method": name,

        "accuracy":
            overall_acc * 100,

        "macro_f1":
            macro_f1,

        "boundary_acc":
            boundary_acc,

        "within_acc":
            within_acc,

        "persistence_acc":
            persistence_acc * 100,

        "kappa_temporal":
            kappa_temporal,

        "boundary_persistence_acc":
            boundary_persistence_acc,

        "boundary_delta_vs_persistence":
            boundary_delta,

        "within_delta_vs_persistence":
            within_delta,

        "eval_rows":
            len(ev),

        "boundary_rows":
            int(
                boundary_mask.sum()
            ),

        "within_rows":
            int(
                within_mask.sum()
            ),

        "drift_count":
            int(drift_count)
    }


# ============================================================================
# 12. PRINT RESULTS
# ============================================================================

def print_results(rows):

    print()
    print("=" * 110)
    print(
        "ADAPTIVE DRIFT-METHOD BASELINE "
        "(TON-IoT / PREQUENTIAL / SECOND HALF)"
    )
    print("=" * 110)

    header = (
        f"{'Method':<24}"
        f"{'Acc':>9}"
        f"{'Macro-F1':>11}"
        f"{'Boundary':>11}"
        f"{'Within':>11}"
        f"{'Kappa-T':>11}"
        f"{'Bdry Δ':>11}"
        f"{'Drifts':>9}"
    )

    print(header)
    print("-" * 110)

    for r in rows:

        print(
            f"{r['method']:<24}"
            f"{r['accuracy']:>9.2f}"
            f"{r['macro_f1']:>11.2f}"
            f"{r['boundary_acc']:>11.2f}"
            f"{r['within_acc']:>11.2f}"
            f"{r['kappa_temporal']:>11.3f}"
            f"{r['boundary_delta_vs_persistence']:>11.2f}"
            f"{r['drift_count']:>9}"
        )

    print("-" * 110)

    print(
        f"Evaluation rows : "
        f"{rows[0]['eval_rows']:,}"
    )

    print(
        f"Boundary rows   : "
        f"{rows[0]['boundary_rows']:,}"
    )

    print(
        f"Within-block    : "
        f"{rows[0]['within_rows']:,}"
    )

    print()

    print(
        "IMPORTANT INTERPRETATION:"
    )

    print(
        "  Boundary accuracy is evaluated only where "
        "the true label changes from the previous row."
    )

    print(
        "  W=1 persistence has exactly 0% boundary accuracy "
        "by definition."
    )

    print(
        "  Therefore boundary_delta_vs_persistence is the "
        "more meaningful boundary comparison."
    )

    print(
        "  Kappa-Temporal is an OVERALL metric relative to "
        "the persistence baseline; it is NOT a boundary-only metric."
    )

    print()


# ============================================================================
# 13. MAIN EXPERIMENT
# ============================================================================

def run():

    total_start = time.time()

    print()
    print("=" * 78)
    print("STARTING EXPERIMENT")
    print("=" * 78)
    print()

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    X, y, feat = load(
        DATA_PATH
    )

    # ------------------------------------------------------------------------
    # OPTIONAL SUBSAMPLE
    # ------------------------------------------------------------------------

    if (
        SUBSAMPLE is not None
        and len(y) > SUBSAMPLE
    ):

        print(
            f"Using final "
            f"{SUBSAMPLE:,} chronological rows."
        )

        X = X[-SUBSAMPLE:]
        y = y[-SUBSAMPLE:]

    # ------------------------------------------------------------------------
    # DATA SIZE
    # ------------------------------------------------------------------------

    n = len(y)

    # Evaluation begins at 50%.
    eval_from_idx = int(
        round(
            n * EVAL_FROM
        )
    )

    # ------------------------------------------------------------------------
    # TEMPORAL PERSISTENCE
    # ------------------------------------------------------------------------

    persistence_rate = (
        np.mean(
            y[1:] == y[:-1]
        )
        * 100
    )

    print(
        f"Rows              : {n:,}"
    )

    print(
        f"Features           : {len(feat):,}"
    )

    print(
        f"Persistence W=1    : "
        f"{persistence_rate:.2f}%"
    )

    print(
        f"Evaluation starts  : "
        f"{eval_from_idx:,}"
    )

    print(
        f"Evaluation fraction: "
        f"{100 * (1 - EVAL_FROM):.1f}%"
    )

    print()

    rows = []

    # =========================================================================
    # METHOD 1: PERSISTENCE W=1
    # =========================================================================

    print("=" * 78)
    print("METHOD 1/4: PERSISTENCE W=1")
    print("=" * 78)

    t0 = time.time()

    prev_all = np.empty(
        n,
        dtype=np.int8
    )

    # First prediction is simply the first true label.
    prev_all[0] = y[0]

    # Thereafter:
    # prediction at t = true label at t-1
    prev_all[1:] = y[:-1]

    row = diagnostics(
        "Persistence W=1",
        y,
        prev_all,
        eval_from_idx,
        drift_count=0
    )

    rows.append(row)

    print(
        f"Persistence done "
        f"in {time.time()-t0:.1f}s"
    )

    print()

    # =========================================================================
    # METHOD 2: ADAPTIVE RANDOM FOREST
    # =========================================================================

    print("=" * 78)
    print("METHOD 2/4: ADAPTIVE RANDOM FOREST")
    print("=" * 78)

    t0 = time.time()

    arf = make_arf()

    arf_preds, arf_drifts = prequential(
        arf,
        X,
        y,
        feat,
        adwin_retrain=False
    )

    row = diagnostics(
        "ARF",
        y,
        arf_preds,
        eval_from_idx,
        drift_count=arf_drifts
    )

    rows.append(row)

    print(
        f"ARF done "
        f"in {(time.time()-t0)/60:.1f} min"
    )

    print()

    # =========================================================================
    # METHOD 3: HOEFFDING TREE + ADWIN RESET
    # =========================================================================

    print("=" * 78)
    print("METHOD 3/4: HOEFFDING TREE + ADWIN RESET")
    print("=" * 78)

    t0 = time.time()

    ht_adwin = HoeffdingTreeClassifier()

    ht_adwin_preds, ht_adwin_drifts = prequential(
        ht_adwin,
        X,
        y,
        feat,
        adwin_retrain=True
    )

    row = diagnostics(
        "HT + ADWIN reset",
        y,
        ht_adwin_preds,
        eval_from_idx,
        drift_count=ht_adwin_drifts
    )

    rows.append(row)

    print(
        f"HT + ADWIN done "
        f"in {(time.time()-t0)/60:.1f} min"
    )

    print()

    # =========================================================================
    # METHOD 4: PLAIN HOEFFDING TREE
    # =========================================================================

    print("=" * 78)
    print("METHOD 4/4: PLAIN HOEFFDING TREE")
    print("=" * 78)

    t0 = time.time()

    ht = HoeffdingTreeClassifier()

    ht_preds, ht_drifts = prequential(
        ht,
        X,
        y,
        feat,
        adwin_retrain=False
    )

    row = diagnostics(
        "Hoeffding tree",
        y,
        ht_preds,
        eval_from_idx,
        drift_count=ht_drifts
    )

    rows.append(row)

    print(
        f"HT done "
        f"in {(time.time()-t0)/60:.1f} min"
    )

    print()

    # =========================================================================
    # RESULTS
    # =========================================================================

    print_results(
        rows
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    results_df = pd.DataFrame(
        rows
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"Saved results to:"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print(
        f"TOTAL RUNTIME: "
        f"{(time.time()-total_start)/60:.1f} minutes"
    )

    print()
    print("=" * 78)
    print("EXPERIMENT FINISHED")
    print("=" * 78)


# ============================================================================
# 14. RUN
# ============================================================================

if __name__ == "__main__":
    run()