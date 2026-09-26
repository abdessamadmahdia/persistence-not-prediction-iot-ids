# Persistence, Not Prediction

Reproducibility package for the paper
**"Persistence, Not Prediction: Label Autocorrelation Confounds Temporal Evaluation of Flow-Based Network Intrusion Detection"**
(A. Mahdia, E.B. Tazi).

On TON-IoT, a trivial baseline that repeats the previous flow's label reaches **99.47 macro-F1**, above every
batch feature-based method in our main protocol. This repository contains the code, configuration, seed, fold
definition and machine-readable results behind that finding and behind the falsified forecasting hypothesis that
led to it.

---

## 1. What this repository supports

1. **A negative result.** A class-conditional feature-on-time forecasting method performs poorly under
   chronological evaluation on TON-IoT. Higher-capacity time regressors do not recover it, and prototypes built
   from true test-period class means (the *leaked prototype*, a label-derived diagnostic) remain less accurate than
   raw features on class-transition rows.
2. **A measurement.** Label persistence confounds chronological evaluation. We quantify it with the
   consecutive-same-class rate, a **window = 1** persistence baseline, **boundary-stratified** accuracy and
   **Kappa-Temporal (κ_T)**. Conditions that receive true labels (window = 1, recency prototypes, leaked prototype,
   prequential online learners) approach the baseline; label-free batch detectors fall below it, mostly through
   errors inside single-class blocks.

---

## 2. Datasets

Datasets are **public but not redistributed here**. Official sources: TON-IoT from UNSW Canberra
(https://research.unsw.edu.au/projects/toniot-datasets); NF-BoT-IoT-v3 and NF-UNSW-NB15-v3 from the University of
Queensland (https://staff.itee.uq.edu.au/marius/NIDS_datasets/). The copies used in the paper are the Kaggle mirrors
listed in [`data/README.md`](data/README.md), together with the expected file names and the row counts to check
your copy against.

| Dataset | Rows used | Features | Selection | Timestamp column |
|---|---|---|---|---|
| TON-IoT (Train_Test Network) | 449,972 | 77 | full release, de-duplicated | `ts` |
| NF-BoT-IoT-v3 | 500,141 | 46–47 | ≈500k most recent rows, de-duplicated | `FLOW_START_MILLISECONDS` |
| NF-UNSW-NB15-v3 | 494,732 | 47 | 500k most recent rows, de-duplicated | `FLOW_START_MILLISECONDS` |

No manual preprocessing is needed: chronological (stable) sorting, encoding, sanitisation, de-duplication and
row selection all happen inside the scripts.

---

## 3. Installation

```bash
git clone https://github.com/abdessamadmahdia/persistence-not-prediction-iot-ids
cd persistence-not-prediction-iot-ids
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # install river with --no-deps if pip upgrades numpy
```

Each script sets its dataset path in a constant near the top. The defaults point at the Kaggle input directories of
the mirrors listed in `data/README.md`; change them to `data/<file>.csv` for local runs. Outputs are written to
`$OUT_DIR` (default: `/kaggle/working` on Kaggle, otherwise `./outputs`).

---

## 4. Experimental protocol

- **Seed:** 42 everywhere.
- **Folds:** 10, expanding-window walk-forward; test fraction τ = 0.10; training fractions
  φ_k = 0.50 + 0.40(k−1)/9 (k = 1…10). Training block = first round(nφ_k) rows, test block = next round(nτ) rows.
  The split is by row position after a stable time sort, so max(train ts) ≤ min(test ts); rows sharing a timestamp
  at the split can fall on either side.
- **Fold dependence:** consecutive test blocks overlap by ≈55.6%; together they cover the second half of each series
  (TON-IoT: 39,998 rows scored once, 144,992 twice, 39,996 three times). No fold-level significance claims are made;
  the paired Wilcoxon statistics printed by some scripts are descriptive only and are not used in the paper.
- **De-duplication:** `drop_duplicates()` on the raw rows *before* identifiers and timestamps are dropped. TON-IoT:
  all raw columns (including `ts`, IP addresses, ports and labels). NF-v3: all columns except the two IPv4 address
  columns (so ports and millisecond timestamps are included). Feature-level duplicates with different timestamps or
  ports are **not** removed.
- **Probe:** RandomForest (300 trees, depth 12, min_samples_leaf 5, `max_features="sqrt"`, `balanced_subsample`,
  seed 42), identical across all prototype conditions; kNN (k = 5) alongside in §5.1.
- **Relationship representation:** `z = [x−a, x−b, |x−a|, |x−b|, ‖x−a‖₂, ‖x−b‖₂]`, dimensionality 4d + 2.
- **Prototype sources:**
  - static — training-block class means (label-free);
  - learned — ridge regression on φ(u) = [u, u²] (label-free);
  - causal recency (W = 50/250/1000) — mean of each class among rows [i−W, i−1]; fallback: running mean of all
    earlier rows of the class, then the training-block mean. **Uses the true labels of preceding test rows (label
    feedback); no future rows.**
  - leaked — B ∈ {50, 100, 200} timestamp-quantile bins over the **whole series** (≈B/10 bins per test block);
    per-bin true class means; whole-series class mean if a class is absent from a bin. **Uses test-period labels.**
  - shuffled — prototype pairs permuted within the training set and within the test set.
- **κ_T:** `(acc − acc_persistent) / (1 − acc_persistent)`, **pooled** over folds.

**Naming in code and result files:** the leaked prototype is labelled `oracle` (e.g. `oracle`, `oracle@100`,
`oracle_shuf`), recency conditions `recency_W<W>`, and the window = 1 baseline `persistence_W1`.

---

## 5. How to reproduce, in order

```bash
# 0) Dataset characterisation (Section 4; Tables 3-4)
python scripts/00_characterization/persistence_characterization.py   # set DATASET = ton | botiot | unsw
python scripts/00_characterization/class_counts.py
python scripts/00_characterization/dedup_persistence_check.py        # set DATASET = unsw | botiot

# 1) Main TON-IoT experiments
python scripts/01_main_toniot/forecasting_method_10fold.py           # Section 5.1 (Table 5), Section 5.2 MAE
python scripts/01_main_toniot/prototype_ladder_10fold.py             # Sections 5.3-5.4 (Tables 7, 9)

# 2) Diagnostics
python scripts/02_diagnostics/time_regressor_capacity.py             # Section 5.2 (Table 6)
python scripts/02_diagnostics/leaked_prototype_bins_and_boundary_confusion.py   # Tables 8, 10
python scripts/02_diagnostics/transition_upweighting.py              # Table 11
python scripts/02_diagnostics/tenfold_error_split_and_overlap.py results        # Section 5.4 error split, Section 4.2 overlap
python scripts/02_diagnostics/prequential_error_split_and_run_bootstrap.py results   # Section 5.6 (Table 12 κ_T CIs)

# 3) Online learners (needs river)
python scripts/03_online/online_learners_prequential.py              # Section 5.6 (Table 12)
python scripts/03_online/online_learners_reset_count.py              # Section 5.6, supplementary reset count

# 4) Cross-dataset
python scripts/04_cross_dataset/nfv3_replication_10fold.py           # Table 13; run with DATASET = botiot and = unsw
python scripts/04_cross_dataset/botiot_boundary_confusion.py         # Section 5.7, NF-BoT-IoT-v3 boundary rows

# 5) Figures 2 and 3 from the committed results
python scripts/05_figures/make_figures_2_3.py results figures
```

Runtimes on a single CPU are hours, not minutes (for example, the Section 5.1 run takes about 3.7 h).

---

## 6. Results

[`results/`](results/) holds the machine-readable outputs behind the manuscript tables;
[`docs/RESULTS_MAPPING.md`](docs/RESULTS_MAPPING.md) maps each file to the table it supports and states which
files were assembled from run logs.

Notes on reading them:

- **`TONIOT_forecasting_method_10fold.csv` includes an `rf_mcc` column.** Under extreme persistence some test blocks
  contain a single class; macro-F1 is then uninformative while MCC is zero by construction. Read macro-F1 together
  with MCC.
- **`TONIOT_time_regressor_capacity.csv` contains `attack_r2` / `normal_r2` values of very large magnitude.** These
  are uniform averages of per-feature R² and are dominated by features whose test-period variance is close to zero
  (R² = 1 − SSE/SST becomes very large in magnitude when SST ≈ 0). Only the RandomForest-on-time normal-class R²
  (≈ 0.24) is discussed in the paper.

---

## 7. Citation

See [`CITATION.cff`](CITATION.cff). Licensed MIT (see [`LICENSE`](LICENSE)); dataset licences are held by
their original providers.
