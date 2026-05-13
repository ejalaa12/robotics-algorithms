import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from kalman import KalmanFilter as SimpleKF  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Innovation statistics / likelihood
# ─────────────────────────────────────────────────────────────────────────────
# PROPOSAL: where should this function live?
#
#   Option A – instance method on KalmanFilter in kalman.py:
#                   kf.innovation_stats(z, H, R)  →  (y, S, mahal_sq, likelihood)
#               Pro: natural call-site, KF owns its own statistics.
#               Con: adds measurement-space logic to the base filter class.
#
#   Option B – module-level function in kalman.py:
#                   from kalman import innovation_stats
#               Pro: no class change, stays reusable, easy to import.
#               Con: slightly more verbose call site.
#
#   Option C – kept here in mhkf.py (current)
#               Pro: self-contained.
#               Con: not reusable by other filter files.
# ─────────────────────────────────────────────────────────────────────────────

def innovation_stats(kf, z, H, R):
    """
    Compute innovation statistics for kf against measurement z.

    Parameters
    ----------
    kf : KalmanFilter (kalman.py)  – must expose .mu (n,) and .cov (n,n)
    z  : (m,) array                – measurement vector
    H  : (m, n) array              – observation matrix
    R  : (m, m) array              – measurement noise covariance

    Returns
    -------
    y         : (m,) innovation vector
    S         : (m, m) innovation covariance
    mahal_sq  : scalar, squared Mahalanobis distance  (chi-sq distributed, df=m)
    likelihood: scalar Gaussian likelihood  p(z | predicted state)
    """
    y = z - H @ kf.mu
    S = H @ kf.cov @ H.T + R
    S_inv = np.linalg.inv(S)
    mahal_sq = float(y @ S_inv @ y)
    m = len(z)
    likelihood = float(
        np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * np.linalg.det(S))
    )
    return y, S, mahal_sq, likelihood


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis
# ─────────────────────────────────────────────────────────────────────────────

class Hypothesis:
    """
    A single tracking hypothesis backed by a KalmanFilter from kalman.py.

    Parameters
    ----------
    kf              : SimpleKF instance
    label           : display name
    chi2_threshold  : float or None
        Gate threshold on the squared Mahalanobis distance (chi-squared, df = m).
        None means always fuse every measurement.
        Use the class constants below for common 1-D gate values, or compute via
        scipy.stats.chi2.ppf(confidence, df).
    """

    # Common chi-squared thresholds for 1-D measurements (df = 1)
    GATE_90  = 2.706
    GATE_95  = 3.841
    GATE_99  = 6.635
    GATE_999 = 10.828

    def __init__(self, kf, label, chi2_threshold=None):
        self.kf = kf
        self.label = label
        self.chi2_threshold = chi2_threshold
        self.weight = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# MHKF
# ─────────────────────────────────────────────────────────────────────────────

class MHKF:
    """
    Multiple-Hypothesis Kalman Filter as a bank of independent KFs.

    Each hypothesis decides whether to fuse a measurement independently via
    chi-squared gating.  The fused MHKF estimate is the weighted mean of all
    hypothesis states, with weights proportional to the measurement likelihood
    evaluated at the *predicted* state (before any update).

    This means that when an outlier arrives:
    - The "always fuse" hypothesis updates toward it  → low predicted-state
      likelihood next step → low weight in the fused estimate.
    - Strict-gated hypotheses that rejected the outlier maintain a clean state
      → high likelihood for subsequent normal measurements → dominate the blend.
    """

    def __init__(self, hypotheses):
        self.hypotheses = hypotheses

    def step(self, z, H, R, F, Q):
        likelihoods = []
        for hyp in self.hypotheses:
            hyp.kf.predict(F, Q)
            _, _, mahal_sq, lik = innovation_stats(hyp.kf, z, H, R)
            likelihoods.append(lik)
            if hyp.chi2_threshold is None or mahal_sq <= hyp.chi2_threshold:
                hyp.kf.correct(z, H, R)

        total = sum(likelihoods)
        for hyp, lik in zip(self.hypotheses, likelihoods):
            hyp.weight = lik / total if total > 0 else 1.0 / len(self.hypotheses)

    def estimate(self):
        return sum(h.weight * h.kf.mu for h in self.hypotheses)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)

dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
Q = np.eye(2) * 0.05
R = np.array([[0.5]])

x0 = np.array([0.0, 1.0])
P0 = np.eye(2)

# Simple KF reference – always fuses, no gating (kalman.py, untouched)
simple_kf = SimpleKF(x0.copy(), P0.copy())


def _make_hyp(label, threshold=None):
    return Hypothesis(SimpleKF(x0.copy(), P0.copy()), label, threshold)


mhkf = MHKF([
    _make_hyp("always fuse"),
    _make_hyp(f"gate χ²≤{Hypothesis.GATE_95:.2f}  (95 %)",  Hypothesis.GATE_95),
    _make_hyp(f"gate χ²≤{Hypothesis.GATE_99:.2f}  (99 %)",  Hypothesis.GATE_99),
    _make_hyp(f"gate χ²≤{Hypothesis.GATE_999:.2f} (99.9 %)", Hypothesis.GATE_999),
])

steps = 50
outlier_prob  = 0.20   # fraction of steps with an outlier
outlier_scale = 8.0    # outlier noise = outlier_scale × normal noise std

true_states      = []
measurements     = []
is_outlier_flags = []
estimates_mhkf   = []
estimates_simple = []
hyp_history      = []  # list of dicts: {t, pos, weight, label}

x_true = x0.copy()

for t in range(steps):
    x_true = F @ x_true + np.random.multivariate_normal([0, 0], Q)

    outlier = np.random.rand() < outlier_prob
    noise_std = np.sqrt(R[0, 0]) * (outlier_scale if outlier else 1.0)
    z = np.array([float((H @ x_true)[0]) + np.random.normal(0, noise_std)])

    mhkf.step(z, H, R, F, Q)
    simple_kf.predict(F, Q)
    simple_kf.correct(z, H, R)

    true_states.append(float(x_true[0]))
    measurements.append(float(z[0]))
    is_outlier_flags.append(outlier)
    estimates_mhkf.append(float(mhkf.estimate()[0]))
    estimates_simple.append(float(simple_kf.mu[0]))

    for hyp in mhkf.hypotheses:
        hyp_history.append(
            dict(t=t, pos=float(hyp.kf.mu[0]), weight=hyp.weight, label=hyp.label)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

time_axis  = np.arange(steps)
true_arr   = np.array(true_states)
mhkf_err   = np.abs(np.array(estimates_mhkf)   - true_arr)
simple_err = np.abs(np.array(estimates_simple) - true_arr)

all_labels  = list(dict.fromkeys(d["label"] for d in hyp_history))  # insertion order
cmap        = plt.colormaps["tab10"]
hyp_colors  = {lbl: cmap(i / max(len(all_labels), 1)) for i, lbl in enumerate(all_labels)}

outlier_ts   = [t for t, o in enumerate(is_outlier_flags) if o]
normal_ts    = [t for t, o in enumerate(is_outlier_flags) if not o]
outlier_vals = [measurements[t] for t in outlier_ts]
normal_vals  = [measurements[t] for t in normal_ts]

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.suptitle(
    "Multi-Hypothesis KF  vs  Simple KF  —  with measurement outliers",
    fontsize=13, fontweight="bold",
)

# ── 1 · Position ──────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(time_axis, true_states, "k-", lw=2, zorder=5, label="True position")
ax1.scatter(normal_ts,  normal_vals,  c="gray", marker="x", s=40, zorder=4, label="Measurements")
ax1.scatter(outlier_ts, outlier_vals, c="red",  marker="x", s=90, zorder=4, label="Outliers")
ax1.plot(time_axis, estimates_simple, "k--", lw=1.5, alpha=0.7, label="Simple KF")
ax1.plot(time_axis, estimates_mhkf,   "b-",  lw=2.0,           label="MHKF fused")
for lbl in all_labels:
    ts = [d["t"]   for d in hyp_history if d["label"] == lbl]
    ps = [d["pos"] for d in hyp_history if d["label"] == lbl]
    ax1.plot(ts, ps, lw=1.0, alpha=0.55, color=hyp_colors[lbl],
             linestyle=":", label=f"{lbl}")
ax1.set_ylabel("Position")
ax1.legend(fontsize=7, loc="upper left", ncol=2)
ax1.grid(True, alpha=0.4)

# ── 2 · Absolute error ────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(time_axis, simple_err, "k--", lw=1.5, alpha=0.8,
         label=f"Simple KF   RMSE = {np.sqrt(np.mean(simple_err**2)):.3f}")
ax2.plot(time_axis, mhkf_err,   "b-",  lw=1.5,
         label=f"MHKF fused  RMSE = {np.sqrt(np.mean(mhkf_err**2)):.3f}")
ax2.fill_between(time_axis, simple_err, mhkf_err,
                 where=(mhkf_err  < simple_err), alpha=0.15, color="blue",  label="MHKF better")
ax2.fill_between(time_axis, simple_err, mhkf_err,
                 where=(mhkf_err >= simple_err), alpha=0.15, color="gray",  label="Simple KF better")
for t in outlier_ts:
    ax2.axvline(t, color="red", alpha=0.18, lw=1)
ax2.set_ylabel("|error|  (position)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.4)

# ── 3 · Hypothesis weights ────────────────────────────────────────────────────
ax3 = axes[2]
for lbl in all_labels:
    ts = [d["t"]      for d in hyp_history if d["label"] == lbl]
    ws = [d["weight"] for d in hyp_history if d["label"] == lbl]
    ax3.plot(ts, ws, lw=1.5, color=hyp_colors[lbl], label=lbl)
for t in outlier_ts:
    ax3.axvline(t, color="red", alpha=0.18, lw=1)
ax3.set_ylabel("Hypothesis weight")
ax3.set_xlabel("Time step")
ax3.set_ylim(0, 1.05)
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

