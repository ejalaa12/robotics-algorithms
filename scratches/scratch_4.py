import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# =========================
# Parameters
# =========================
dt = 0.1
N = 100

v = 1.0  # forward velocity
omega = 0.3  # angular velocity

# Process noise (tune this!)
Q = np.diag([0.01, 0.01, 0.005])

# Initial state [x, y, theta]
x = np.array([0.0, 0.0, 0.0])

# Initial covariance
P = np.diag([0.1, 0.1, 0.05])

# Storage
x_history = []
P_history = []

P_xx, P_yy, P_xy = [], [], []

eig1_list = []
eig2_list = []


# =========================
# Dubins model
# =========================
def motion_model(x, v, omega, dt):
    theta = x[2]

    x_new = np.zeros_like(x)
    x_new[0] = x[0] + v * np.cos(theta) * dt
    x_new[1] = x[1] + v * np.sin(theta) * dt
    x_new[2] = x[2] + omega * dt

    return x_new


def compute_F(x, v, dt):
    theta = x[2]

    F = np.eye(3)
    F[0, 2] = -v * np.sin(theta) * dt
    F[1, 2] = v * np.cos(theta) * dt

    return F


# =========================
# Simulation loop
# =========================
for k in range(N):
    # Predict state
    x = motion_model(x, v, omega, dt)

    # Compute Jacobian
    F = compute_F(x, v, dt)

    # Predict covariance
    P = F @ P @ F.T + Q

    # Store
    x_history.append(x.copy())
    P_history.append(P.copy())

    P_xx.append(P[0, 0])
    P_yy.append(P[1, 1])
    P_xy.append(P[0, 1])

    Pxy = P[:2, :2]
    eigvals = np.linalg.eigvalsh(Pxy)  # sorted eigenvalues

    eig1_list.append(eigvals[0])
    eig2_list.append(eigvals[1])

x_history = np.array(x_history)

# =========================
# Plot covariance evolution
# =========================
plt.figure()
plt.plot(P_xx, label="Var(x)")
plt.plot(P_yy, label="Var(y)")
plt.plot(P_xy, label="Cov(x,y)")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Covariance")
plt.title("Covariance evolution (prediction only)")
plt.grid()

# =========================
# Plot covariance evolution (using eigenvalues)
# =========================
plt.figure()
plt.plot(eig1_list, label="Eigenvalue 1 (minor axis)")
plt.plot(eig2_list, label="Eigenvalue 2 (major axis)")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Eigenvalues")
plt.title("Covariance Eigenvalues Evolution")
plt.grid()


# =========================
# Ellipse plotting function
# =========================
def plot_cov_ellipse(P, x, ax, n_std=2.0):
    Pxy = P[:2, :2]

    eigvals, eigvecs = np.linalg.eigh(Pxy)

    angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = Ellipse(
        xy=(x[0], x[1]),
        width=width,
        height=height,
        angle=angle,
        edgecolor="red",
        fc="None",
        lw=1,
    )
    ax.add_patch(ellipse)


# =========================
# Plot trajectory + ellipses
# =========================
fig, ax = plt.subplots()

ax.plot(x_history[:, 0], x_history[:, 1], label="Trajectory")

for k in range(0, N, 5):
    plot_cov_ellipse(P_history[k], x_history[k], ax)

ax.set_aspect("equal")
ax.set_title("Trajectory with Covariance Ellipses")
ax.legend()
ax.grid()

plt.show()
