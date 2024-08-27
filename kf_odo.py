from typing import List, Union
import numpy as np
from filters.kalman.kalman import KalmanFilter
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

dt = 0.1
duration = 10.0
times = np.arange(0, duration, dt)


# System
true_state = np.array([0.0, 0.0]).reshape(-1, 1)
mes_noise = 0.02

# Model with constant velocity: X = (x, v)
class ConstantVelocityKF(KalmanFilter):
    def __init__(self, X0: Union[List, np.ndarray], G0: Union[List, np.ndarray]):
        super().__init__(X0, G0)

    def compute_state_transition_matrix(self) -> np.ndarray:
        return np.array([[1, dt], [0, 1]])

    def compute_control_input_model_matrix(self) -> Union[np.ndarray, None]:
        return None

    def compute_process_noise(self) -> np.ndarray:
        return dt * np.diag([0.1, 0.01])  # this is the

    def correct2(self, z):
        # - and velocity observable
        C = np.array([[0, 1]])
        R = np.diag([mes_noise])
        return super().correct(z, C, R)

kf = ConstantVelocityKF([0, 0], [0.1, 1])

all_X = [kf.X.copy()]
all_G = [kf.G.copy()]
all_true_X = [true_state.copy()]


# Other kalman with speed as prediction model
class OdoModelKF(KalmanFilter):
    def __init__(self, X0: Union[List, np.ndarray], G0: Union[List, np.ndarray]):
        super().__init__(X0, G0)

    def predict(self, u: np.ndarray):
        A = np.eye(self.X.shape[0])
        B = 
        return super().predict(A, Q, B, u)

kf2 = KalmanFilter([0], [0.1])


# speed command
def speed_command(t):
    return 1.0


for t in times[:-1]:
    # speed command
    u = speed_command(t)
    # update system
    true_state[0] += true_state[1] * dt
    true_state[1] = u

    # do measurement of speed
    measurement = true_state[1] + np.random.normal(0, mes_noise)

    # update estimator
    kf.predict()
    kf.correct2(measurement)

    # save new states
    all_true_X.append(true_state.copy())
    all_X.append(kf.X.copy())
    all_G.append(kf.G.copy())

# %% numpify paths
all_X = np.array(all_X)
all_G = np.array(all_G)
all_true_X = np.array(all_true_X)

print(all_X.shape)
print(all_G.shape)
print(all_true_X.shape)

# %% plot
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(times, all_true_X[:, 0], label="$x$")
ax1.errorbar(times, all_X[:, 0], np.sqrt(all_G[:, 0, 0]) * 3, label=r"$\hat{x}$")
ax1.legend()

ax2.plot(times, all_true_X[:, 1], label="$v$")
ax2.errorbar(times, all_X[:, 1], np.sqrt(all_G[:, 1, 1]) * 3, label=r"$\hat{v}$")
ax2.legend()

plt.show()
