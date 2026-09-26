"""Figures 2 and 3 of the manuscript, generated from the committed result files.

Figure 2: pooled Kappa-Temporal versus TEST-BLOCK label persistence (the pooled accuracy
of the window = 1 baseline over the walk-forward test blocks, i.e. the persistence of the
rows actually classified).  Test-block persistence values:
  NF-UNSW-NB15-v3 85.32 % and TON-IoT 99.54 % are printed by the 10-fold runs
  ("persistent baseline accuracy = ..."); NF-BoT-IoT-v3 is derived as
  1 - (41 boundary rows / 50,014 test rows) = 99.92 % (approximate: the boundary count
  per fold is the rounded mean in results/BOTIOT_boundary_confusion.csv).
Figure 3: macro-F1 of raw features and of the window = 1 baseline, labelled with the
full-sequence persistence and full-corpus attack share (results/persistence_characterization_ALL.csv).
Usage: python make_figures_2_3.py [results_dir] [out_dir]
"""
import sys, os
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
NAVY, TEAL, ORANGE = "#0d2645", "#127d78", "#f3a35d"

cross = pd.read_csv(f"{R}/CROSS_DATASET_summary.csv").set_index("dataset")
char = pd.read_csv(f"{R}/persistence_characterization_ALL.csv").set_index("dataset")
bot_nb = pd.read_csv(f"{R}/BOTIOT_boundary_confusion.csv")["avg_boundary_rows_per_fold"].iloc[0]
test_persist = {"NF-UNSW-NB15-v3": 85.32, "TON-IoT": 99.54,
                "NF-BoT-IoT-v3": 100 * (1 - bot_nb / 50014)}
order = ["NF-UNSW-NB15-v3", "TON-IoT", "NF-BoT-IoT-v3"]
labels = {"NF-UNSW-NB15-v3": "NF-UNSW-NB15-v3\n(interleaved)", "TON-IoT": "TON-IoT",
          "NF-BoT-IoT-v3": "NF-BoT-IoT-v3"}

# ---------------- Figure 2 ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.6))
x = [test_persist[d] for d in order]
ax.axhspan(0, 8, color=TEAL, alpha=0.07, lw=0); ax.axhspan(-52, 0, color=ORANGE, alpha=0.07, lw=0)
ax.axhline(0, color="#333", ls="--", lw=0.9)
ax.plot(x, [cross.loc[d, "kappaT_oracle"] for d in order], "-o", color=TEAL, lw=2.2, ms=8,
        label="leaked prototype")
ax.plot(x, [cross.loc[d, "kappaT_raw"] for d in order], "-s", color=NAVY, lw=2.2, ms=8,
        label="raw features ($X$)")
ax.annotate(labels[order[0]], (x[0], 1.0), xytext=(85.0, -33), fontweight="bold", color=NAVY,
            arrowprops=dict(arrowstyle="-", color="#333", lw=0.8))
ax.annotate("TON-IoT", (x[1], cross.loc["TON-IoT", "kappaT_raw"]), xytext=(93.2, -20),
            fontweight="bold", color=NAVY, arrowprops=dict(arrowstyle="-", color="#333", lw=0.8))
ax.annotate("NF-BoT-IoT-v3", (x[2], cross.loc["NF-BoT-IoT-v3", "kappaT_raw"]), xytext=(92.2, -41),
            fontweight="bold", color=NAVY, arrowprops=dict(arrowstyle="-", color="#333", lw=0.8))
ax.text(84.3, 4.5, r"$\kappa_T>0$: better than the persistence baseline", color=TEAL, fontsize=10)
ax.text(84.3, -49.5, r"$\kappa_T<0$: worse than the persistence baseline", color="#b35900", fontsize=10)
ax.set_xlim(84, 100.5); ax.set_ylim(-52, 8)
ax.set_xlabel("Test-block label persistence (% of test rows with the same class as the previous row)")
ax.set_ylabel(r"Kappa-Temporal ($\kappa_T$, pooled)")
ax.grid(alpha=0.3); ax.legend(loc="center left", framealpha=0.95)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/Figure_2.{ext}", dpi=600, metadata={"Software": None} if ext == "png" else {"Creator": None, "Producer": None, "CreationDate": None})
plt.close(fig)

# ---------------- Figure 3 ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.6))
import numpy as np
pos = np.arange(3); w = 0.36
raw = [cross.loc[d, "rawX_macroF1"] for d in order]; per = [cross.loc[d, "window1_macroF1"] for d in order]
b1 = ax.bar(pos - w/2, raw, w, color=NAVY, label="raw features ($X$)")
b2 = ax.bar(pos + w/2, per, w, color=ORANGE, label="window = 1 persistence baseline")
for b, v in zip(b1, raw): ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.2f}", ha="center", color=NAVY)
for b, v in zip(b2, per): ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.2f}", ha="center", color="#b35900")
ax.set_xticks(pos)
ax.set_xticklabels([f"{d}\npersistence {char.loc[d,'persistence_pct']:.2f}%\nattack {char.loc[d,'attack_pct']:.1f}%"
                    for d in order], fontsize=10)
ax.set_ylabel("macro-F1"); ax.set_ylim(0, 120); ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper center", ncol=2, framealpha=0.95)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/Figure_3.{ext}", dpi=600, metadata={"Software": None} if ext == "png" else {"Creator": None, "Producer": None, "CreationDate": None})
plt.close(fig)
print("test-block persistence used:", {k: round(v, 2) for k, v in test_persist.items()})
