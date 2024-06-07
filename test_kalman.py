# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# %% Class 2-Point-1D-System

class TwoPointSystem2D:
    def __init__(self, center, heading, distance):
        self.distance = distance
        self.X = np.array([center - distance / 2, center + distance / 2, heading])

    def update(self, v, dt):
        self.X = self.X + v * dt

    def observe_left(self, sigma=0.1):
        return np.random.normal(self.X[0], sigma)

    def observe_right(self, sigma=0.1):
        return np.random.normal(self.X[1], sigma)

    def observe_heading(self, sigma=np.radians(3)):
        return np.random.normal(self.X[2], sigma)

    @property
    def center(self):
        return self.X[0] + self.distance / 2

    @property
    def heading(self):
        return self.X[2]


# %% SIMULATION

from filters.kalman import ExtendedKalmanFilter

s = TwoPointSystem2D(0, np.radians(10), 4)
kf = ExtendedKalmanFilter(np.array([4, 0.5]), np.eye(2) * 5)


# %%
def hl(x):
    return x - 2


H = np.array([1, 0])

A = np.eye(2)
Q = 0.01 * np.eye(2)
B = np.eye(2)
# %%
rl = 0.1 * 0.1 * np.eye(1)
dt = 0.1
v = 0.2
for i in range(100):
    s.update(0.2, dt)
    u = dt * np.array([0.2, 0.01])
    kf.predict(A, Q, B, u)
    x, y = s.observe_left(), s.observe_right()
    kf.correct(x, hl, H, np.eye(1) * 0.1)

print(f"EKF estimated that the system is at {kf.X[0]}, system is at {s.center}")


# %% Implement animation

# todo: il faut gerer le tf server pour pas oublier la lbl frame
# todo: un serveur pour tcp/udp pour read n'importe quelle sortie gpgga

fig, ax = plt.subplots()

def update(frame):
    pass


# %% Show animation
ani = FuncAnimation(fig, update, interval=100,
                    blit=True)
plt.show()
