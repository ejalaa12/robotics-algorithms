from typing import Union, Callable

import numpy as np

np.set_printoptions(linewidth=300)


class MonteCarlo:
    """
    A Python implementation of the Monte-Carlo Localization (MCL)

    Histogram does the same but discretizes the space, so less accurate
    """

    def __init__(
        self,
        width: Union[list, np.ndarray],
        initial_estimate: Union[list, np.ndarray],
        n: int = 200,
    ):
        """
        Initialize the particle filter with `n` particles with a uniform distribution in a box
        of width `width` centered around the `initial_estimate` point.

        :param width: the width of the box
        :param initial_estimate: the initial estimate of the filter
        :param n: the number of particles
        """
        if isinstance(width, list):
            width = np.array(width)
        if isinstance(initial_estimate, list):
            initial_estimate = np.array(initial_estimate)

        self.particles = np.random.uniform(
            initial_estimate - width / 2,
            initial_estimate + width / 2,
            (n, initial_estimate.size),
        )
        self.weights = np.zeros((n, 1)) + 1 / n

    def update_motion(self, f: Callable[[np.ndarray], np.ndarray], u: np.ndarray):
        """
        Update the particles with the f function and input u

        The function f must take in a value of the shape of one sample and return the motion_updated new value

        :param f: the motion update function
        :param u: the input of the update function
        """
        self.particles = np.apply_along_axis(f, 1, self.particles, u)

    def update_sensor(self, h: Callable[[np.ndarray], np.ndarray], z: np.ndarray):
        """
        Corrects the weights of the particles with the observation function g with data z

        The function h takes in an array [X, w], and the observation [z] and returns a new weight [w]

        :param h: the observation function
        :param z: the observation of the sensor
        """
        self.weights = np.apply_along_axis(
            h, 1, np.hstack((self.particles, self.weights)), z
        ).reshape(-1, 1)

    def resample(self):
        """
        Resamples the particles by choosing
        :return:
        """
        n = self.particles.shape[0]
        indices = np.random.choice(
            n, p=self.weights.flatten() / np.sum(self.weights), size=n
        )
        self.particles = self.particles[indices]
        self.weights = np.zeros((n, 1)) + 1 / n


# %% Test monte carlo


class Robot:
    def __init__(self, x, y):
        self.X = np.array([x, y])

    def update(self, v, dt):
        self.X = self.X + v * dt

    @property
    def x(self):
        return np.random.normal(self.X[0], 0.2)

    @property
    def y(self):
        return np.random.normal(self.X[1], 0.2)


# %%
import matplotlib.pyplot as plt

m = MonteCarlo([2, 4], [0, 0], n=2000)
r = Robot(0.7, -1.3)

plt.figure()


def draw():
    plt.cla()
    plt.scatter(m.particles[:, 0], m.particles[:, 1], s=m.weights * 100)
    plt.scatter(r.X[0], r.X[1], color="r")


draw()


# %% Assume we observe x, the sensor likelihood function is as follows:


def hx(x, z):
    gauss = lambda x, mu, sigma: np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return gauss(x[0], z, 0.2)


def hy(x, z):
    gauss = lambda x, mu, sigma: np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return gauss(x[1], z, 0.2)


def f(x, u):
    return np.random.normal(x + u, 0.1)


# %% animation
from scipy.spatial.transform import Rotation as Rot
from matplotlib.animation import FuncAnimation

m = MonteCarlo([8, 10], [0, 0], n=2000)
r = Robot(-1.7, -1.3)
fig = plt.figure()
sc = plt.scatter(m.particles[:, 0], m.particles[:, 1], s=m.weights * 1000)
sc2 = plt.scatter(r.X[0], r.X[1], color="r", marker=".")


def update(frame):
    if frame == 0:
        return sc, sc2
    k = 1000
    m.update_sensor(hx, r.x)
    m.resample()
    m.update_sensor(hy, r.y)
    m.resample()
    r.update(np.array([-np.sin(frame / 10), -np.cos(frame / 10)]), 0.1)
    m.update_motion(f, [-np.sin(frame / 10) * 0.1, -np.cos(frame / 10) * 0.1])
    # if frame % 3 == 0:
    #     if frame % 6 == 0:
    #         print("update sensor x")
    #         m.update_sensor(hx, r.x)
    #         m.resample()
    #     else:
    #         print("update sensor y")
    #         m.update_sensor(hy, r.y)
    #         m.resample()
    # k = 10
    # if frame % 3 == 1:
    #     print("resample")
    #     m.resample()
    # if frame % 3 == 2:
    #     print("update motion")
    #     r.update([0.1, 0.4], 1)
    #     m.update_motion(f, [0.1, 0.4])
    sc.set_offsets(m.particles)
    sc.set_sizes(m.weights.flatten() * k)
    sc2.set_offsets(r.X)
    sc2.set_color("r")
    return sc, sc2


def plot_covariance(data):
    cov = np.cov(data)
    (l1, l2), (x1, x2) = np.linalg.eig(cov)
    a1, a2 = np.sqrt(l1), np.sqrt(l2)
    angle = np.atan2(l2, l1)
    rot = Rot.from_euler("z", angle).as_matrix()[:2, :2]


# %%
ani = FuncAnimation(fig, update, interval=100, blit=True)
plt.show()
