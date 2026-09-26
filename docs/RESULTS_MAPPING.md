# Results → manuscript mapping

Table and figure numbers refer to the manuscript. In code and result files the leaked prototype is labelled `oracle`.

| Result file | Supports | Key values | Produced by |
|---|---|---|---|
| `persistence_characterization_ALL.csv`, `persistence_characterization_{ton,botiot,unsw}.csv`, `runlength_summary_{ton,botiot,unsw}.csv` | Table 4; §4.1 full-corpus attack shares | TON 99.80% / U 0.978 / attack 34.93%; BoT 99.93% / U 0.807; UNSW 91.03% / U 0.024; longest TON runs 150,000 normal / 80,000 attack | `00_characterization/persistence_characterization.py` (full label sequence, before de-duplication; `_ALL` combines the three per-dataset files) |
| — (printed) | Table 3 | TON 161,043 / 288,929; BoT 499,342 / 799; UNSW 43,694 / 451,038 | `00_characterization/class_counts.py` |
| — (printed) | §4.2 de-duplication effect | NF-UNSW tail: 500,000 → 494,732 rows; same-class rate 85.37% → 85.22% | `00_characterization/dedup_persistence_check.py` |
| `TONIOT_forecasting_method_10fold.csv` | Table 5 | raw 78.27 (kNN) / 77.01 (RF, MCC 0.584); static 78.54 / 79.66 (0.624); learned 46.80 / 52.82 (0.102); shuffled 62.06 / 62.93 (0.326) | `01_main_toniot/forecasting_method_10fold.py` |
| — (printed) | §5.2 MAE | quadratic vs static: attack 0.2716 vs 0.2077; normal 0.2908 vs 0.3153 | `01_main_toniot/forecasting_method_10fold.py` |
| `TONIOT_time_regressor_capacity.csv` | Table 6; §5.2 R² | quadratic 52.82; spline 48.62; RF-on-time 52.98; boundary 57.93 / 54.43 / 58.00; RF-on-time normal R² 0.235 | `02_diagnostics/time_regressor_capacity.py` |
| `TONIOT_prototype_ladder_10fold.csv` | Tables 7, 9 | raw 77.01 / boundary 79.47 / κ_T −27.18; static 79.66 / 79.63 / −22.00; leaked 96.50 / 59.53 / −5.43 (within 97.18) | `01_main_toniot/prototype_ladder_10fold.py` (the std, MCC and boundary-F1 columns come from the same run's printed output) |
| `TONIOT_error_split_estimate.csv` | §5.4, §6.1 error split | within-block share of errors ≈ 99.3% (raw), 99.1% (static), 93.8% (leaked), 93.7% (recency W = 50) | `02_diagnostics/tenfold_error_split_and_overlap.py` (estimated from mean accuracies) |
| `TONIOT_leaked_prototype_bins.csv` | Table 8 | B = 50 → 91.34 / 72.85; B = 100 → 96.50 / 59.53; B = 200 → 97.56 / 67.58 (≈5 / 10 / 20 bins per test block) | `02_diagnostics/leaked_prototype_bins_and_boundary_confusion.py` |
| `TONIOT_boundary_confusion.csv` | Table 10 | leaked R(normal) 49.7, P(attack) 45.7; raw R(normal) 87.9 | same script |
| `TONIOT_transition_upweighting.csv` | Table 11 | boundary 79.47 → 79.68 across ×1/5/20/100 | `02_diagnostics/transition_upweighting.py` |
| `TONIOT_online_learners.csv`, `TONIOT_online_learners_perrow_predictions.csv` | Table 12 | persistence 99.56; batch RF 77.29 / boundary 89.01 / κ_T −54.43; ARF +0.107 / 85.20; HT+ADWIN −0.805; HT −4.081 | `03_online/online_learners_prequential.py` |
| `TONIOT_online_error_split.csv`, `TONIOT_online_kappaT_bootstrap.csv` | §5.6; Table 12 κ_T CI column | ARF run-level κ_T CI [−0.083, +0.278]; batch RF errors 50,837 within / 101 boundary; recall normal 64.9%, attack 99.5%; all ARF within-block errors ≤ 100 rows after a transition | `02_diagnostics/prequential_error_split_and_run_bootstrap.py` |
| `TONIOT_online_learners_reset_count_log.txt`, `TONIOT_online_learners_reset_count.csv` | §5.6 reset count | HT+ADWIN 37 resets; ARF κ_T +0.049; HT+ADWIN −0.736; HT −6.665 | `03_online/online_learners_reset_count.py` (River 0.26.1, unstandardised features); the CSV is transcribed from the log |
| `confound_NF_BoT_IoT_v3.csv`, `confound_NF_UNSW_NB15_v3.csv` | Table 13, Figures 2–3 | UNSW raw 99.97 / leaked 99.96 / κ_T +1.00; BoT raw 88.06 / κ_T −44.95 | `04_cross_dataset/nfv3_replication_10fold.py` (DATASET = unsw / botiot) |
| `CROSS_DATASET_summary.csv` | Table 13, Figures 2–3 | UNSW +1.00; TON −27.18; BoT −44.95 | assembled from `confound_NF_*.csv` and `TONIOT_prototype_ladder_10fold.csv` (attack ratios and label structure as in Table 13) |
| `BOTIOT_boundary_confusion.csv` | §5.7 boundary discussion | about 41 boundary rows per fold; no boundary conclusion drawn | `04_cross_dataset/botiot_boundary_confusion.py` |
| `UNSW_boundary_confusion.csv` | §5.7 | 7,262 boundary rows per fold; all values 100.0 | assembled by hand from the NF-UNSW-NB15-v3 run (boundary accuracy 99.99%, rounded); no script computes its per-class values |
| `figures/Figure_2.*`, `figures/Figure_3.*` | Figures 2–3 | test-block persistence 85.32 / 99.54 / ≈99.92 | `05_figures/make_figures_2_3.py` |

## Test-block persistence

The pooled accuracy of the window = 1 baseline over the test blocks is printed by the 10-fold runs as
"persistent baseline accuracy": 85.32% (NF-UNSW-NB15-v3) and 99.54% (TON-IoT). For NF-BoT-IoT-v3 it is derived as
1 − 41 / 50,014 ≈ 99.92%.

## Online comparison: main run and reset-count run

Table 12 reports the main run (`online_learners_prequential.py`: standardised features, ADWIN not re-initialised
after a reset), whose per-row predictions are committed and reproduce every value in the table. That run does not
count drift events. The supplementary run (`online_learners_reset_count.py`) counts ADWIN-triggered resets of the
Hoeffding tree (37); it uses unstandardised features and re-initialises ADWIN after each detection, so its
accuracies differ slightly. ARF's internal drift detections are not counted in either run.
