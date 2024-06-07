# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:59:53 2021

@author: ejalaa
"""

import numpy as np
from typing import Union, Callable


class KalmanFilter:
    def __init__(self, mu0: np.ndarray, cov0: np.ndarray):
        self.mu = mu0.copy()
        self.cov = cov0.copy()

    def predict(
        self,
        A: np.ndarray,
        Q: Union[None, np.ndarray] = None,
        B: Union[None, np.ndarray] = None,
        u: Union[None, np.ndarray] = None,
    ):
        if B is None or u is None:
            self.mu = A @ self.mu
        else:
            self.mu = A @ self.mu + B @ u

        if Q is None:
            Q = np.zeros(A.shape)
        self.cov = A @ self.cov @ A.T + Q

    def correct(self, Z: np.ndarray, H: np.ndarray, R: np.ndarray):
        y = Z - H @ self.mu
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)

        I = np.eye(self.cov.shape[0])

        self.mu = self.mu + K @ y
        self.cov = (I - K @ H) @ self.cov


class ExtendedKalmanFilter(KalmanFilter):
    def correct(
        self,
        Z: np.ndarray,
        h: Callable,
        H: np.ndarray,
        R: Union[None, np.ndarray] = None,
    ):
        if R is None:
            R = np.zeros((Z.size, Z.size))
        y = Z - h(self.X)
        S = H @ self.G @ H.T + R
        K = self.G @ H.T @ np.linalg.inv(S)

        I = np.eye(self.G.shape[0])

        self.X = self.X + K @ y
        self.G = (I - K @ H) @ self.G


# %%


def test_kalman():
    kf = KalmanFilter(np.random.normal([1, 2], 10), np.eye(2))
    for _ in range(100):
        kf.predict(np.eye(2))
        obs = np.random.normal([1, 2], 0.3)
        kf.correct(obs, np.eye(2), 0.3**2 * np.eye(2))
    return np.linalg.norm(kf.mu - [1, 2])


# %%


def test_kalman1d():
    kf = KalmanFilter(np.random.normal([1], 10), np.eye(1))
    for _ in range(100):
        kf.predict(np.eye(1))
        obs = np.random.normal([1], 0.3)
        kf.correct(obs, np.eye(1), 0.3**2 * np.eye(1))
    return np.linalg.norm(kf.mu - [1])


# %%


def test_kalman_odom_imu():
    # x, y, vx, vy, theta, w
    # kf = KalmanFilter(np.random.normal([0, 0, 0, 0, 0, 0], 5), np.eye(6))
    # for _ in range(100):
    #     kf.predict([])
    pass


# %%

if __name__ == "__main__":
    r = test_kalman()
    assert r <= 0.1

    r = test_kalman1d()
    assert r <= 0.1
