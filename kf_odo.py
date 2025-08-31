from typing import List, Union
import numpy as np
from robotics_algorithms.filters.kalman.kalman import KalmanFilter, LinearObservation, GaussianDistribution
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

dt = 0.1
duration = 10.0
times = np.arange(0, duration, dt)


# System
true_state = np.array([0.0, 0.0]).reshape(-1, 1)
mes_noise = 0.02


class ObservationModelV(LinearObservation):

    def get_observation_matrix(self) -> Union[np.ndarray, None]:
        return np.array([[0, 1]])


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


class OdoModelKF(KalmanFilter):
    def __init__(self, X0: Union[List, np.ndarray], G0: Union[List, np.ndarray]):
        super().__init__(X0, G0)
        assert self.distribution.mean.size == 1

    def compute_state_transition_matrix(self) -> np.ndarray:
        return np.eye(1)

    def compute_control_input_model_matrix(self) -> Union[np.ndarray, None]:
        return np.eye(1) * dt

    def compute_process_noise(self) -> np.ndarray:
        return dt * np.diag([0.01])  # this is the

kf = ConstantVelocityKF([0, 0], [0.1, 1])
kf2 = OdoModelKF([0], [0.1])

all_kf_dist = [kf.distribution.copy()]
all_kf2_dist = [kf2.distribution.copy()]
all_true_X = [true_state.copy()]

# speed command
def speed_command(t=None):
    if t < duration / 3:
        return 1.0
    if t < 2 * duration / 3:
        return 0.0

    return 1.0


for t in times[:-1]:
    # speed command
    u = speed_command(t)
    # update true system
    true_state[0] += true_state[1] * dt
    true_state[1] = u

    # do measurement of speed
    measurement = true_state[1] + np.random.normal(0, mes_noise)
    measurement = GaussianDistribution(measurement, mes_noise**2)

    # update constant velocity estimator 
    kf.predict()
    kf.correct(ObservationModelV(measurement))
    # update odo model estimator (no corrections, only prediction)
    kf2.predict(np.array([u]))


    # save new states
    all_true_X.append(true_state.copy())
    all_kf_dist.append(kf.distribution.copy())
    all_kf2_dist.append(kf2.distribution.copy())

# %% numpify paths
all_X = [dist.mean for dist in all_kf_dist]
all_X = np.array([dist.mean for dist in all_kf_dist]).reshape(-1, kf.distribution.mean.size)
all_G = np.array([dist.covariance for dist in all_kf_dist])
all_X2 = np.array([dist.mean for dist in all_kf2_dist]).reshape(-1, kf2.distribution.mean.size)
all_G2 = np.array([dist.covariance for dist in all_kf2_dist])
all_true_X = np.array(all_true_X)

print(all_X.shape)
print(all_X2.shape)
print(all_G.shape)
print(all_true_X.shape)

# %% plot
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(times, all_true_X[:, 0], label="$x$")
ax1.plot(times, all_X[:, 0], label=r"$\hat{x}$")
ax1.plot(times, all_X2[:, 0], label=r"$\hat{x_2}$")
# ax1.errorbar(times, all_X[:, 0], np.sqrt(all_G[:, 0, 0]) * 3, label=r"$\hat{x}$")
# ax1.errorbar(times, all_X2[:, 1], np.sqrt(all_G[:, 0, 0]) * 3, label=r"$\hat{x}$")
ax1.legend()

ax2.plot(times, all_true_X[:, 1], label="$v$")
ax2.errorbar(times, all_X[:, 1], np.sqrt(all_G[:, 1, 1]) * 3, label=r"$\hat{v}$")
ax2.legend()

plt.show()
