"""
mhkf.py — Multi-Hypothesis Kalman Filter (MHKF)
================================================

KEY CONCEPTS
------------

1. HYPOTHESIS = one possible "association history"
   ─────────────────────────────────────────────
   At every time step we receive a measurement z.  We don't know whether
   z actually came from the target or is spurious (clutter / outlier).
   A hypothesis is one specific answer to the question:
       "For every past step, was that measurement real or clutter?"
   Each hypothesis carries its own independent Kalman filter whose state
   reflects the updates that were accepted along that particular history.

2. BRANCHING = the hypothesis tree
   ─────────────────────────────────
   At each step every surviving hypothesis spawns exactly two children:

     ┌─ "associate" child  ──  assumes z is real  →  KF updates
     │                         weight ∝ likelihood p(z | predicted state)
   H ─┤
     └─ "clutter" child    ──  assumes z is noise  →  KF coasts
                               weight ∝ constant clutter density λ

   Without pruning the tree has 2^t leaves after t steps.

3. WEIGHTS = posterior probability of each history
   ─────────────────────────────────────────────────
   The weight of a hypothesis is proportional to the joint probability
   of all its past association decisions given the data received so far.

       w_h ∝  ∏_{t: A} p(z_t | predicted_state)  ×  ∏_{t: C} λ

   After normalising across all hypotheses the weights form a proper
   posterior over association histories.

4. PRUNING = keeping the tree tractable
   ────────────────────────────────────
   We keep only the top max_hypotheses hypotheses by weight after each
   branching + normalisation step.  Low-probability histories are pruned.
   This is the KEY approximation that makes MHKF practical.

5. FUSED ESTIMATE = probability-weighted mean
   ──────────────────────────────────────────
   We never commit to a single history.  The final state estimate is:

       x̂ = Σ_h  w_h · μ_h

   where μ_h is the KF mean of hypothesis h.  Hypotheses that accepted
   outliers have low weights and contribute little to the fused estimate.

WHY IS THIS BETTER THAN A SIMPLE KF?
─────────────────────────────────────
A simple KF always incorporates every measurement.  When an outlier
arrives, the innovation y = z − H·μ is large, the Kalman gain pulls the
state toward the outlier, and the error spikes.  The MHKF maintains the
hypothesis that z was clutter — keeping one or more filters clean — and
the fused estimate stays robust because clean hypotheses dominate by weight.
"""

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
#                   kf.innovation_stats(z, H, R)
#               Pro: KF owns its statistics.
#               Con: adds measurement-space logic to the base class.
#
#   Option B – module-level function in kalman.py:
#                   from kalman import innovation_stats
#               Pro: no class change, easily reusable.
#
#   Option C – kept here (current)
#               Pro: self-contained learning example.
# ─────────────────────────────────────────────────────────────────────────────


def innovation_stats(kf, z, H, R):
    """
    Compute innovation statistics for kf against measurement z.

    Parameters
    ----------
    kf : KalmanFilter (kalman.py)  – must expose .mu (n,) and .cov (n,n)
    z  : (m,) array                – measurement vector
    H  : (m, n) array
    R  : (m, m) array

    Returns
    -------
    y         : (m,) innovation vector
    S         : (m, m) innovation covariance
    mahal_sq  : scalar, squared Mahalanobis distance (chi-sq, df = m)
    likelihood: scalar Gaussian likelihood p(z | predicted state)
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _clone_kf(kf):
    """Return an independent copy of a KalmanFilter (mu and cov only)."""
    return SimpleKF(kf.mu.copy(), kf.cov.copy())


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis
# ─────────────────────────────────────────────────────────────────────────────


class Hypothesis:
    """
    One node in the MHKF hypothesis tree.

    Attributes
    ----------
    kf      : KalmanFilter whose state reflects the updates accepted along
              this hypothesis's association history.
    weight  : posterior probability (normalised each step).
    history : list of 'A' (associate) / 'C' (clutter) decisions, one per step.
              Records the "story" this hypothesis tells about the data so far.
    """

    def __init__(self, kf, weight, history=None):
        self.kf = kf
        self.weight = weight
        self.history = history if history is not None else []


# ─────────────────────────────────────────────────────────────────────────────
# MHKF
# ─────────────────────────────────────────────────────────────────────────────


class MHKF:
    """
    Multi-Hypothesis Kalman Filter.

    Each call to step() grows the hypothesis tree by branching every
    surviving hypothesis into two children, then prunes to max_hypotheses.
    """

    def __init__(self, hypotheses, max_hypotheses=8, clutter_weight=0.05):
        """
        Parameters
        ----------
        hypotheses      : list[Hypothesis]  – initial (single) hypothesis
        max_hypotheses  : int  – tree size cap; the key pruning parameter
        clutter_weight  : float
            Constant weight factor for the clutter branch, representing the
            prior probability density of receiving a clutter measurement.
            Smaller values bias the filter toward fusing measurements.
        """
        self.hypotheses = hypotheses
        self.max_hypotheses = max_hypotheses
        self.clutter_weight = clutter_weight

    def step(self, z, H, R, F, Q):
        """
        One MHKF step.

        For each surviving hypothesis:
          1. Predict  – advance KF state with motion model F, Q.
          2. Branch   – create associate child (KF updates) and
                        clutter child (KF coasts).
          3. Assign weights based on measurement likelihood / clutter density.
        Then: normalise all weights, prune to max_hypotheses.
        """
        new_hypotheses = []

        for hyp in self.hypotheses:
            # ── Predict ──────────────────────────────────────────────────────
            # Both children share the same predicted state, so we predict in
            # place and clone before each update.
            hyp.kf.predict(F, Q)

            # ── Associate branch ─────────────────────────────────────────────
            # Interpretation: z came from the target.
            # The KF incorporates z; weight reflects how well z fits the
            # prediction (Gaussian likelihood).
            kf_a = _clone_kf(hyp.kf)
            _, _, _, lik = innovation_stats(kf_a, z, H, R)
            kf_a.correct(z, H, R)
            new_hypotheses.append(
                Hypothesis(kf_a, hyp.weight * lik, hyp.history + ["A"])
            )

            # ── Clutter branch ───────────────────────────────────────────────
            # Interpretation: z is spurious noise; target was not detected.
            # The KF coasts (no update); weight uses a constant clutter density.
            kf_c = _clone_kf(hyp.kf)
            new_hypotheses.append(
                Hypothesis(kf_c, hyp.weight * self.clutter_weight, hyp.history + ["C"])
            )

        # ── Normalise ─────────────────────────────────────────────────────────
        # Turn unnormalised scores into a proper posterior over histories.
        total = sum(h.weight for h in new_hypotheses)
        for h in new_hypotheses:
            h.weight /= total if total > 0 else 1.0 / len(new_hypotheses)

        # ── Prune ─────────────────────────────────────────────────────────────
        # Keep only the most probable hypotheses.  This is the approximation
        # that prevents exponential growth of the tree.
        new_hypotheses.sort(key=lambda h: h.weight, reverse=True)
        self.hypotheses = new_hypotheses[: self.max_hypotheses]

    def estimate(self):
        """Probability-weighted mean state across all surviving hypotheses."""
        return sum(h.weight * h.kf.mu for h in self.hypotheses)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(0)

dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])   # constant-velocity motion model
H = np.array([[1.0, 0.0]])              # position-only measurement
Q = np.eye(2) * 0.01                    # small process noise → S stays near R
R = np.array([[1.0]])                   # measurement noise  (std = 1.0)

x0 = np.array([0.0, 1.0])              # initial state [position, velocity]
P0 = np.eye(2)

# ── Simple KF (reference) ────────────────────────────────────────────────────
# Always fuses every measurement, no outlier handling.
simple_kf = SimpleKF(x0.copy(), P0.copy())

# ── MHKF ─────────────────────────────────────────────────────────────────────
# clutter_weight calibration:
#   With small Q, steady-state S ≈ R = 1.0.
#   Peak Gaussian likelihood = 1/sqrt(2π·S) ≈ 0.40.
#   clutter_weight = 0.002 is ~200× smaller → 'A' wins decisively on normal
#   steps while 'C' wins whenever the likelihood collapses at outlier steps.
mhkf = MHKF(
    [Hypothesis(SimpleKF(x0.copy(), P0.copy()), weight=1.0)],
    max_hypotheses=10,
    clutter_weight=0.002,
)

steps = 40

# Fixed, evenly-spaced outlier steps — deterministic for a clear demo.
# Random outliers risk "chained" outliers in the same direction, which lets
# a corrupted hypothesis accumulate weight and look plausible to the filter.
# Fixed spacing guarantees the filter settles between outliers.
outlier_steps = {8, 16, 24, 32}

# Outliers are a fixed ±10σ displacement — NOT a scaled Gaussian draw.
# Random-magnitude "outliers" can accidentally be tiny (e.g., 0.1σ), making
# them indistinguishable from a normal measurement and breaking the demo.
outlier_magnitude = 10.0 * np.sqrt(R[0, 0])   # 10 std-devs from truth

true_states = []
measurements = []
is_outlier_flags = []
estimates_mhkf = []
estimates_simple = []

# Per-step: weight of rank-k hypothesis (for weight-evolution subplot)
MAX_RANK = 4
rank_weights = [
    [] for _ in range(MAX_RANK)
]  # rank_weights[k][t] = weight of (k+1)-th hypothesis at step t

x_true = x0.copy()

for t in range(steps):
    x_true = F @ x_true + np.random.multivariate_normal([0, 0], Q)

    outlier = t in outlier_steps
    if outlier:
        sign = np.random.choice([-1.0, 1.0])
        z = np.array([float((H @ x_true)[0]) + sign * outlier_magnitude])
    else:
        z = np.array([float((H @ x_true)[0]) + np.random.normal(0, np.sqrt(R[0, 0]))])

    mhkf.step(z, H, R, F, Q)
    simple_kf.predict(F, Q)
    simple_kf.correct(z, H, R)

    true_states.append(float(x_true[0]))
    measurements.append(float(z[0]))
    is_outlier_flags.append(outlier)
    estimates_mhkf.append(float(mhkf.estimate()[0]))
    estimates_simple.append(float(simple_kf.mu[0]))

    for k in range(MAX_RANK):
        rank_weights[k].append(
            mhkf.hypotheses[k].weight if k < len(mhkf.hypotheses) else 0.0
        )


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

time_axis = np.arange(steps)
true_arr = np.array(true_states)
mhkf_err = np.abs(np.array(estimates_mhkf) - true_arr)
simple_err = np.abs(np.array(estimates_simple) - true_arr)

outlier_ts = [t for t, o in enumerate(is_outlier_flags) if o]
normal_ts = [t for t, o in enumerate(is_outlier_flags) if not o]
outlier_vals = [measurements[t] for t in outlier_ts]
normal_vals = [measurements[t] for t in normal_ts]

# Build history matrix from final surviving hypotheses (already sorted by weight)
final_hyps = mhkf.hypotheses
n_hyps = len(final_hyps)
history_matrix = np.array(
    [[1 if dec == "A" else 0 for dec in hyp.history] for hyp in final_hyps],
    dtype=float,
)
final_weights = [hyp.weight for hyp in final_hyps]

fig, axes = plt.subplots(
    4,
    1,
    figsize=(14, 14),
    gridspec_kw={"height_ratios": [3, 2, 2, 2]},
    sharex=False,
)
fig.suptitle(
    "Multi-Hypothesis Kalman Filter  —  true MHKF with hypothesis tree",
    fontsize=13,
    fontweight="bold",
)

# ── 1 · Position ──────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(time_axis, true_states, "k-", lw=2, zorder=5, label="True position")
ax1.scatter(
    normal_ts, normal_vals, c="gray", marker="x", s=40, zorder=4, label="Measurements"
)
ax1.scatter(
    outlier_ts, outlier_vals, c="red", marker="x", s=90, zorder=4, label="Outliers"
)
ax1.plot(
    time_axis,
    estimates_simple,
    "k--",
    lw=1.5,
    alpha=0.7,
    label=f"Simple KF  (RMSE={np.sqrt(np.mean(simple_err**2)):.2f})",
)
ax1.plot(
    time_axis,
    estimates_mhkf,
    "b-",
    lw=2,
    label=f"MHKF fused (RMSE={np.sqrt(np.mean(mhkf_err**2)):.2f})",
)
ax1.set_ylabel("Position")
ax1.set_xlabel("Time step")
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.4)

# ── 2 · Absolute error ────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(time_axis, simple_err, "k--", lw=1.5, alpha=0.8, label="Simple KF")
ax2.plot(time_axis, mhkf_err, "b-", lw=1.5, label="MHKF fused")
ax2.fill_between(
    time_axis,
    simple_err,
    mhkf_err,
    where=(mhkf_err < simple_err),
    alpha=0.15,
    color="blue",
    label="MHKF better",
)
ax2.fill_between(
    time_axis,
    simple_err,
    mhkf_err,
    where=(mhkf_err >= simple_err),
    alpha=0.15,
    color="gray",
    label="Simple KF better",
)
for t in outlier_ts:
    ax2.axvline(t, color="red", alpha=0.18, lw=1)
ax2.set_ylabel("|error|")
ax2.set_xlabel("Time step")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.4)

# ── 3 · Weight evolution of top-k hypotheses ─────────────────────────────────
# Shows how quickly the top hypothesis separates and how pruning concentrates
# probability mass on a small number of consistent histories.
ax3 = axes[2]
colors = plt.colormaps["tab10"](np.linspace(0, 0.4, MAX_RANK))
for k in range(MAX_RANK):
    ax3.plot(time_axis, rank_weights[k], lw=1.5, color=colors[k], label=f"rank {k + 1}")
for t in outlier_ts:
    ax3.axvline(t, color="red", alpha=0.18, lw=1)
ax3.set_ylabel("Hypothesis weight")
ax3.set_xlabel("Time step")
ax3.set_ylim(0, 1.05)
ax3.legend(fontsize=8, ncol=MAX_RANK)
ax3.grid(True, alpha=0.4)

# ── 4 · Association history heatmap ──────────────────────────────────────────
# Each row is one surviving hypothesis (sorted highest weight → top).
# Blue  = 'A'  (associated: KF updated with this measurement)
# Red   = 'C'  (clutter: KF coasted, measurement ignored)
# Dashed red columns mark the true outlier steps.
# The top rows (highest weight) should show 'C' at outlier steps.
ax4 = axes[3]
im = ax4.imshow(
    history_matrix,
    aspect="auto",
    cmap="RdYlBu",
    vmin=0,
    vmax=1,
    interpolation="nearest",
    extent=[-0.5, steps - 0.5, n_hyps - 0.5, -0.5],
)

# Mark outlier columns
for t in outlier_ts:
    ax4.axvline(t, color="red", lw=1.5, alpha=0.5, linestyle="--")

# Annotate rows with final weight
for i, w in enumerate(final_weights):
    ax4.text(
        steps - 0.3,
        i,
        f" w={w:.3f}",
        va="center",
        ha="left",
        fontsize=7,
        color="black",
    )

ax4.set_yticks(range(n_hyps))
ax4.set_yticklabels([f"hyp {i + 1}" for i in range(n_hyps)], fontsize=7)
ax4.set_xlabel("Time step")
ax4.set_ylabel("Hypothesis\n(rank 1 = highest weight)")
ax4.set_title(
    "Association history of surviving hypotheses  "
    "[ blue = associate 'A'  |  red = clutter 'C'  |  dashed = true outlier ]",
    fontsize=8,
)

cbar = fig.colorbar(im, ax=ax4, orientation="vertical", fraction=0.015, pad=0.02)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Clutter", "Associate"])

plt.tight_layout()
plt.show()
