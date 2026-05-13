"""
error_state_ekf.py — Error-State Extended Kalman Filter (ES-EKF)
=================================================================

DIRECT EKF vs ERROR-STATE EKF
──────────────────────────────

In a DIRECT EKF the filter tracks x = [x, y, θ] directly.
The motion model f(x, u) is nonlinear (unicycle), so the EKF must
linearise it at every step:  F = ∂f/∂x  evaluated at the current
estimate.  Linearisation errors accumulate, especially for large
rotations.

In an ERROR-STATE EKF the estimation is split into two parts:

  NOMINAL STATE  x̄
  ─────────────────
  Propagated by integrating the (noisy) sensor inputs with the EXACT
  nonlinear model.  No filter — just dead-reckoning.  The nominal state
  is never "touched" by covariance mathematics; it just follows the
  odometry.

  ERROR STATE  δx = x_true − x̄
  ─────────────────────────────
  A small perturbation around the nominal.  Because δx is kept near zero
  (corrections are injected every time a GPS fix arrives), its dynamics
  are approximately LINEAR even when the original system is highly
  nonlinear.  This means a STANDARD Kalman filter from kalman.py is
  sufficient — no EKF Jacobians needed for the filter itself.

ALGORITHM LOOP
──────────────

At every odometry step:
  1. Propagate nominal:  x̄_new  = f(x̄, u)            [exact, nonlinear]
  2. Linearise error:    F_δ    = ∂f/∂(δx) at x̄       [3×3 matrix]
  3. Build noise cov:    Q_δ    = B(x̄) · Q_u · B(x̄)ᵀ  [from input noise]
  4. KF predict:         kf.predict(F_δ, Q_δ)

When GPS arrives:
  5. Innovation:         δz = z_GPS − H · x̄            [measurement residual
                                                         in the ERROR space]
  6. KF correct:         kf.correct(δz, H_δ, R_GPS)    [updates error estimate]
  7. Inject & reset:     x̄ ← x̄ + kf.mu               [apply correction]
                         kf.mu ← 0                     [reset error mean]
                         (covariance stays as posterior)

WHY IS THIS BETTER?
───────────────────
  • The nominal propagation is exact — no linearisation error in the
    prediction step.
  • The KF only linearises the ERROR dynamics, which are small angles and
    small displacements → the linear approximation is very accurate.
  • Crucial for systems with orientation:  θ does not live in a vector
    space (wrap-around), but δθ (a small angle error) does — the KF
    can safely add and subtract it.

DEMO
────
  True trajectory  : unicycle on a circular arc, v = 1 m/s, ω = 0.2 rad/s
  Odometry noise   : Gaussian on v and ω, integrated at 10 Hz
  GPS              : noisy 2-D position, 1 Hz (every 10 odometry steps)
  Comparison       : dead-reckoning only  vs  ES-EKF
"""

import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(__file__))
from kalman import KalmanFilter  # noqa: E402

# The KalmanFilter from kalman.py is used for the ERROR STATE only.
# Interface:
#   KalmanFilter(mu0, cov0)   — mu0: initial error mean (zeros),
#                                cov0: initial error covariance
#   kf.predict(A, Q)          — predict error state forward
#   kf.correct(Z, H, R)       — correct with measurement innovation


# ─────────────────────────────────────────────────────────────────────────────
# Unicycle helpers
# ─────────────────────────────────────────────────────────────────────────────

def propagate_nominal(x_nom, v, omega, dt):
    """
    Propagate nominal state one step with the exact unicycle model.

        x_new = x + v·cos(θ)·dt
        y_new = y + v·sin(θ)·dt
        θ_new = θ + ω·dt

    This is the exact nonlinear integration — no approximation here.
    """
    x, y, theta = x_nom
    return np.array([
        x     + v * np.cos(theta) * dt,
        y     + v * np.sin(theta) * dt,
        theta + omega * dt,
    ])


def linearize_F(x_nom, v, dt):
    """
    Jacobian of the error-state dynamics:  F_δ = ∂(δx_new)/∂(δx)

    For a small error δx = [δx, δy, δθ]:
        δx_new ≈ δx + (−v·sin(θ̄)·δθ)·dt
        δy_new ≈ δy + ( v·cos(θ̄)·δθ)·dt
        δθ_new ≈ δθ

    This is a first-order Taylor expansion of the unicycle model around
    the nominal state.  It is accurate when δθ is small.
    """
    theta = x_nom[2]
    return np.array([
        [1,  0,  -v * np.sin(theta) * dt],
        [0,  1,   v * np.cos(theta) * dt],
        [0,  0,   1                      ],
    ])


def odometry_noise_cov(x_nom, v, sigma_v, sigma_omega, dt):
    """
    Process noise covariance Q_δ for the error state.

    The odometry noise (σ_v on velocity, σ_ω on angular rate) propagates
    into the error state through the input-noise Jacobian:

        B = [[-sin(θ̄)·dt,  0  ],
             [ cos(θ̄)·dt,  0  ],
             [    0,       dt  ]]

    Q_δ = B · diag(σ_v², σ_ω²) · Bᵀ
    """
    theta = x_nom[2]
    B = np.array([
        [-np.sin(theta) * dt,  0.0 ],
        [ np.cos(theta) * dt,  0.0 ],
        [ 0.0,                  dt  ],
    ])
    Q_input = np.diag([sigma_v**2, sigma_omega**2])
    return B @ Q_input @ B.T


def covariance_ellipse(cov2x2, n_sigma=2, n_points=100):
    """
    Return (x, y) points of the n_sigma covariance ellipse for a 2×2
    covariance matrix.  Used to visualise position uncertainty.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov2x2)
    eigenvalues = np.maximum(eigenvalues, 0)
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    ellipse_pts = eigenvectors @ np.diag(n_sigma * np.sqrt(eigenvalues)) @ circle
    return ellipse_pts[0], ellipse_pts[1]


# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(7)

dt          = 0.1         # odometry integration step  [s]
gps_period  = 10          # GPS arrives every this many odometry steps
steps       = 600         # total odometry steps  (60 s of data)

# True motion inputs
v_true     = 1.0          # true forward speed  [m/s]
omega_true = 0.2          # true angular rate   [rad/s]  → circle r = v/ω = 5 m

# Odometry noise (standard deviations)
sigma_v     = 0.05        # velocity noise  [m/s]
sigma_omega = 0.02        # angular rate noise  [rad/s]

# GPS measurement noise
sigma_gps = 0.3           # GPS position noise  [m]
R_gps = np.eye(2) * sigma_gps**2

# Observation matrix for GPS: measures [x, y] from error state [δx, δy, δθ]
H_gps = np.array([
    [1, 0, 0],
    [0, 1, 0],
])

# ─────────────────────────────────────────────────────────────────────────────
# Initialise
# ─────────────────────────────────────────────────────────────────────────────

x0 = np.array([0.0, 0.0, 0.0])   # starting pose

# Nominal state starts at the true initial pose
x_nom = x0.copy()

# Dead-reckoning state (odometry only, no filter)
x_dr = x0.copy()

# KF tracks the error state δx = [δx, δy, δθ]
# Initial error mean = 0 (we assume the nominal starts at true pose)
# Initial error covariance: moderate uncertainty on position, small on heading
kf = KalmanFilter(
    mu0  = np.zeros(3),
    cov0 = np.diag([0.1**2, 0.1**2, (0.01)**2]),
)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

x_true = x0.copy()

true_traj    = [x0.copy()]
nom_traj     = [x0.copy()]   # ES-EKF estimate (= nominal + injected corrections)
dr_traj      = [x0.copy()]   # dead-reckoning
gps_meas     = []            # GPS observations
ellipses     = []            # (x_center, y_center, cov_xy) for plotting

for t in range(1, steps + 1):

    # ── True state update ─────────────────────────────────────────────────────
    x_true = propagate_nominal(x_true, v_true, omega_true, dt)

    # ── Noisy odometry readings ───────────────────────────────────────────────
    v_meas     = v_true     + np.random.normal(0, sigma_v)
    omega_meas = omega_true + np.random.normal(0, sigma_omega)

    # ── Nominal state propagation (exact nonlinear model, noisy inputs) ───────
    # The nominal state just integrates what the odometry sensor reports.
    # No covariance, no linearisation — just dead-reckoning with the best
    # available motion estimate.
    x_nom = propagate_nominal(x_nom, v_meas, omega_meas, dt)

    # ── Dead-reckoning state (same as nominal — shown for comparison) ─────────
    x_dr = propagate_nominal(x_dr, v_meas, omega_meas, dt)

    # ── Error-state prediction (KF predict step) ──────────────────────────────
    # Linearise the error dynamics around the current nominal state.
    F_delta = linearize_F(x_nom, v_meas, dt)

    # Build process noise from odometry measurement uncertainty.
    Q_delta = odometry_noise_cov(x_nom, v_meas, sigma_v, sigma_omega, dt)

    kf.predict(F_delta, Q_delta)

    # ── GPS update (KF correct + inject + reset) ──────────────────────────────
    if t % gps_period == 0:
        # Noisy GPS observation
        z_gps = x_true[:2] + np.random.multivariate_normal([0, 0], R_gps)
        gps_meas.append(z_gps.copy())

        # Innovation lives in the ERROR space:
        #   δz = z_GPS − h(x̄)
        # h(x̄) = [x̄_x, x̄_y]  (GPS measures position, which is also in the
        # nominal state — so the predicted measurement is just the nominal
        # position)
        dz = z_gps - x_nom[:2]

        # KF correct: estimates the error δx
        kf.correct(dz, H_gps, R_gps)

        # Inject correction into nominal state
        # x̄_new ← x̄ + δx̂
        x_nom = x_nom + kf.mu
        x_nom[2] = (x_nom[2] + np.pi) % (2 * np.pi) - np.pi  # wrap heading

        # Reset error mean to zero (posterior covariance is kept)
        kf.mu = np.zeros(3)

    # ── Record ────────────────────────────────────────────────────────────────
    true_traj.append(x_true.copy())
    nom_traj.append(x_nom.copy())
    dr_traj.append(x_dr.copy())

    if t % gps_period == 0:
        ellipses.append((x_nom[0], x_nom[1], kf.cov[:2, :2].copy()))


true_traj = np.array(true_traj)
nom_traj  = np.array(nom_traj)
dr_traj   = np.array(dr_traj)
gps_meas  = np.array(gps_meas)

# Position errors
err_dr    = np.linalg.norm(true_traj[:, :2] - dr_traj[:, :2], axis=1)
err_eskf  = np.linalg.norm(true_traj[:, :2] - nom_traj[:, :2], axis=1)

time_axis = np.arange(steps + 1) * dt

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Error-State EKF  —  2-D unicycle  (noisy odometry + GPS corrections)",
    fontsize=13, fontweight="bold",
)

# ── Left: trajectory ──────────────────────────────────────────────────────────
ax = axes[0]

ax.plot(true_traj[:, 0], true_traj[:, 1], "k-",  lw=2,   label="True trajectory")
ax.plot(dr_traj[:,  0],  dr_traj[:,  1],  "r--", lw=1.5, label="Dead-reckoning (odometry only)")
ax.plot(nom_traj[:, 0],  nom_traj[:, 1],  "b-",  lw=1.5, label="ES-EKF estimate")
ax.scatter(gps_meas[:, 0], gps_meas[:, 1],
           c="green", marker="+", s=60, zorder=5, label="GPS measurements")

# 2-sigma uncertainty ellipses around the ES-EKF estimate
for i, (cx, cy, cov_xy) in enumerate(ellipses):
    ex, ey = covariance_ellipse(cov_xy, n_sigma=2)
    ax.plot(cx + ex, cy + ey, color="steelblue", lw=0.6, alpha=0.5,
            label="2σ ellipse" if i == 0 else "")

ax.set_xlabel("x  [m]")
ax.set_ylabel("y  [m]")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.35)
ax.set_title("Trajectory")

# ── Right: position error over time ───────────────────────────────────────────
ax2 = axes[1]
ax2.plot(time_axis, err_dr,   "r--", lw=1.5,
         label=f"Dead-reckoning  (final={err_dr[-1]:.2f} m)")
ax2.plot(time_axis, err_eskf, "b-",  lw=1.5,
         label=f"ES-EKF          (final={err_eskf[-1]:.2f} m)")
for gps_t in time_axis[gps_period::gps_period]:
    ax2.axvline(gps_t, color="green", alpha=0.12, lw=1)
ax2.set_xlabel("Time  [s]")
ax2.set_ylabel("Position error  [m]")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.35)
ax2.set_title("Position error  (green lines = GPS corrections)")

plt.tight_layout()
plt.show()
