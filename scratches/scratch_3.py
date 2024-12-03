import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from robotics_algorithms.utils.math import wrap_to_pi


# %%
def generate_rotation_commands(num_rotations=3, forward_time=5, rotation_time=2, peak_w=1.0):
    """
    Generate a sequence of angular rotation speed (w) commands for a vehicle to perform multiple rotations to the left.

    Parameters:
    - num_rotations: Number of rotations to the left.
    - forward_time: Time spent going forward (in seconds) before each rotation.
    - rotation_time: Time taken to reach peak w and then decrease to 0 (in seconds) for each rotation.
    - peak_w: The maximum angular speed during rotation.

    Returns:
    - A list of angular velocities (w) over time.
    """
    commands = []
    time_points = []

    # Define the rotation profile for smooth transition
    rotation_time_points = np.linspace(0, rotation_time, num=100)
    rotation_profile = peak_w * np.sin(np.pi * rotation_time_points / rotation_time)

    # Loop to add forward and rotation commands
    dt = 1
    for _ in range(num_rotations):
        t = time_points[-1] if len(time_points) > 0 else 0

        # Forward command (w = 0 for `forward_time` seconds)
        commands.extend([0] * forward_time)
        time_points.extend(np.arange(t, t + forward_time, dt))

        # Rotation command (following the sinusoidal profile)
        commands.extend(rotation_profile)
        time_points.extend(t + forward_time + rotation_time_points)

    # Final forward command (w = 0)
    commands.extend([0] * forward_time)
    time_points.extend(np.arange(time_points[-1], time_points[-1] + forward_time, dt))

    return time_points, commands


def generate_rotation_commands_interp(time_points, num_rotations=3, forward_time=5, rotation_time=2, peak_w=1.0):
    """
    Same as above method, but will interpolate the values to the given time_points
    """
    t, c = generate_rotation_commands(num_rotations, forward_time, rotation_time, peak_w)
    return np.interp(time_points, t, c)


# %%


def can_log():
    """
    A check to enable or not debug logging
    """
    global i
    conditions = [
        # i % 10 == 0,
        i >= 2000,
        False,
    ]
    return all(conditions)


class BicycleModel:
    def __init__(self, x0=0, y0=0, v0=0, theta0=0, w0=0, length=1, name=''):
        self.length = length
        self.X = np.array([x0, y0, v0, theta0, w0]).reshape(-1, 1)
        self.P = np.eye(self.dim)
        self.name = name  # optional, just for logging

    @property
    def dim(self):
        return self.X.shape[0]

    @property
    def x(self):
        return self.X.item(0)

    @property
    def y(self):
        return self.X.item(1)

    @property
    def v(self):
        return self.X.item(2)

    @property
    def theta(self):
        return self.X.item(3)

    @property
    def w(self):
        return self.X.item(4)

    def back(self):
        """return speed of back wheel"""
        return self.v

    def front(self):
        """return speed of front wheel"""
        return np.sign(self.v) * np.sqrt(self.v ** 2 + (self.length * self.w) ** 2)

    def delta(self):
        """return steering angle of front wheel"""
        return np.arctan2(self.length * self.w, self.v)

    def control(self, v, w):
        self.X[2] = v
        self.X[4] = w

    def update(self, dt=0.1):
        xdot = np.array([self.v * np.cos(self.theta), self.v * np.sin(self.theta), 0, self.w, 0]).reshape(-1, 1)

        if can_log() and self.name == 'ekf':
            print(f'\t\tpre-update: {self.theta=} + {xdot[3].item() * dt=}, {self.w=}')

        self.X = self.X + xdot * dt

        if can_log() and self.name == 'ekf':
            print('\t\tpost-update:', f'{self.theta=} ')

        self.X[3] = wrap_to_pi(self.theta)


class BicycleModelEkf(BicycleModel):

    def kalman_predict(self, q, dt=0.1):
        if isinstance(q, list):
            q = np.diag(q)
        self.update(dt)
        jA = np.array([
            [0, 0, np.cos(self.theta), -self.v * np.sin(self.theta), 0],
            [0, 0, np.sin(self.theta), self.v * np.cos(self.theta), 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ])
        jA = np.eye(self.dim) + dt * jA
        self.P = jA @ self.P @ jA.T + q

    def kalman_correct(self, mes, mesP, hx, h):
        y = mes - hx
        s = h @ self.P @ h.T + mesP
        if s.size == 1:
            k = self.P @ h.T * np.linalg.inv(s)
        else:
            k = self.P @ h.T @ np.linalg.inv(s)

        i = np.eye(self.dim)

        if y.size == 1:
            self.X = self.X + k * y
        else:
            self.X = self.X + k @ y
        self.P = (i - k @ h) @ self.P

    def correct_front(self, mes, cov):
        # if isinstance(mes, float):
        #     mes = np.array(mes)
        # mes = mes.reshape(-1, 1)

        hx = np.sign(self.v) * np.sqrt(self.v ** 2 + (self.length * self.w) ** 2)
        if np.isclose(np.abs(self.v), 0, atol=1e-5):
            h = np.array([0, 0, 1, 0, 0])
        else:
            h = np.array([0, 0, self.v / hx, 0, self.length ** 2 * self.w / hx])
        h = h.reshape(1, -1)
        self.kalman_correct(mes, cov, hx, h)

    def correct_back(self, mes, cov):
        # if isinstance(mes, float):
        #     mes = np.array(mes)
        # mes = mes.reshape(-1, 1)

        hx = np.array([self.v]).reshape(-1, 1)
        h = np.array([0, 0, 1, 0, 0]).reshape(1, -1)
        self.kalman_correct(mes, cov, hx, h)

    def correct_wz(self, mes, cov):
        pre_w, pre_theta = self.w, self.theta
        hx = np.array([self.w]).reshape(-1, 1)
        h = np.array([0, 0, 0, 0, 1]).reshape(1, -1)
        self.kalman_correct(mes, cov, hx, h)

        if can_log() and self.name == 'ekf':
            print(f'\t\tcorrecting wz with ({mes=}):')
            print(f'\t\t\t{pre_theta=} --> {self.theta=}')
            print(f'\t\t\t{pre_w=} --> {self.w=}')

    def correct_delta(self, mes, cov):
        hx = np.array([self.delta()]).reshape(-1, 1)
        d = self.v ** 2 + (self.length * self.w) ** 2
        if d > 100:
            print('oh oh')
        if np.isclose(np.abs(self.v), 0, atol=1e-5):
            # if the speed is zero, we don't correct
            return
        h = np.array([0, 0, - self.length * self.w / d, 0, self.length * self.v / d])
        self.kalman_correct(mes, cov, hx, h)


# %%
sim_bicycle = BicycleModel(w0=0.02, name='sim')
# an initial non-zero estimate of the rotation speed is needed otherwise the front wheel correct will never correct w
kf_bicycle = BicycleModelEkf(w0=0.00, name='ekf')

duration = 100
dt = 0.01
time_points = np.arange(0, duration, dt)

states = []
wheels = []
kf_states = []
kf_covs = []
commands = np.zeros((len(time_points), 2))
commands[:, 0] = 1.0  # 1m/s
commands[:, 1] = generate_rotation_commands_interp(time_points, 3, 20, 7, 0.2)
for i, time in enumerate(time_points):
    if can_log():
        print(f'{i: >4} {"-" * 20}')
    v, w = commands[i]
    # control real bicycle
    sim_bicycle.control(v, w)
    sim_bicycle.update(dt)
    # update kalman with only wheels mes
    kf_bicycle.kalman_predict([0.01] * 5, dt)
    kf_bicycle.correct_back(sim_bicycle.back(), np.diag([0.01]))
    kf_bicycle.correct_front(sim_bicycle.front(), np.diag([0.01]))
    # kf_bicycle.correct_wz(sim_bicycle.w, np.diag([0.001]))
    kf_bicycle.correct_delta(sim_bicycle.delta(), np.diag([0.001]))
    # save state
    states.append(sim_bicycle.X.flatten())
    wheels.append([sim_bicycle.back(), sim_bicycle.front()])
    kf_states.append(kf_bicycle.X.flatten())
    kf_covs.append(kf_bicycle.P)

states = np.array(states)
wheels = np.array(wheels)
kf_states = np.array(kf_states)
kf_covs = np.array(kf_covs)

# %%
fig = plt.figure(layout='constrained')
gs = GridSpec(3, 4, figure=fig)
ax_xy = fig.add_subplot(gs[:2, :2])
ax_cmd = fig.add_subplot(gs[2, :2])
ax_v = fig.add_subplot(gs[0, 2:])
ax_w = fig.add_subplot(gs[1, 2:])
ax_theta = fig.add_subplot(gs[2, 2:])

ax_xy.set_title('XY')
ax_xy.plot(states[:, 0], states[:, 1], label='truth')
ax_xy.plot(kf_states[:, 0], kf_states[:, 1], label='ekf')
ax_xy.set_aspect('equal', adjustable='box')
# ax_xy.axis('square')


ax_v.set_title('speed')
ax_v.plot(time_points, commands[:, 0], '--', label='commands')
ax_v.plot(time_points, states[:, 2], label='truth')
ax_v.plot(time_points, kf_states[:, 2], label='ekf')
ax_v.legend()

ax_theta.set_title('theta')
ax_theta.plot(time_points, states[:, 3], label='truth')
ax_theta.plot(time_points, kf_states[:, 3], label='ekf')
ax_theta.legend()

ax_w.set_title('w')
ax_w.plot(time_points, commands[:, 1], '--', label='commands')
ax_w.plot(time_points, states[:, 4], label='truth')
ax_w.plot(time_points, kf_states[:, 4], label='ekf')
ax_w.legend()

ax_cmd.set_title('commands')
ax_cmd.plot(time_points, wheels[:, 0], label='back')
ax_cmd.plot(time_points, wheels[:, 1], label='front')
ax_cmd.legend()

plt.show()
