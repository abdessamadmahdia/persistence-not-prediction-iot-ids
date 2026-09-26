"""Within-block vs boundary error shares for the 10-fold TON-IoT conditions, ESTIMATED from the
committed mean accuracies (results/TONIOT_prototype_ladder_10fold.csv), plus the walk-forward
test-block overlap.  Per-row 10-fold predictions were not saved, so the shares are estimates:
errors_within ~ (1 - within_acc) * (T - B), errors_boundary ~ (1 - boundary_acc) * B,
with T = 44,997 test rows per fold and B = 206 mean boundary rows per fold."""
import sys, os, numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "results"
d = pd.read_csv(os.path.join(R, "TONIOT_prototype_ladder_10fold.csv"))
n, T, B = 449_972, 44_997, 206
d["within_err_est"] = (1 - d.within_acc / 100) * (T - B)
d["boundary_err_est"] = (1 - d.boundary_acc / 100) * B
d["share_within_pct"] = 100 * d.within_err_est / (d.within_err_est + d.boundary_err_est)
out = d[["condition", "within_acc", "boundary_acc", "within_err_est", "boundary_err_est", "share_within_pct"]]
out.round(2).to_csv(os.path.join(R, "TONIOT_error_split_estimate.csv"), index=False)
print(out.round(1).to_string(index=False))
starts = np.linspace(0.50, 0.90, 10)
step = starts[1] - starts[0]
print(f"test-block overlap between consecutive folds: {(0.10 - step) / 0.10 * 100:.1f}%")
cover = np.zeros(n, int)
for f in starts:
    a = int(round(n * f)); b = min(n, int(round(a + n * 0.10))); cover[a:b] += 1
print({k: int((cover == k).sum()) for k in range(4)})
