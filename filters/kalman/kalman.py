# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:59:53 2021

@author: ejalaa
"""

import numpy as np
from typing import Union, Callable, List

def kalman_predict(
        X_prev: np.ndarray,
        G_prev: np.ndarray,
    A: np.ndarray,
    Q: Union[None, np.ndarray] = None,
    B: Union[None, np.ndarray] = None,
    u: Union[None, np.ndarray] = None,
):
    """
    Prediction step of the KalmanFilter.
    This uses the transition/evolution model to propagate the state and covariance
    at the next step

    :param X_prev: previous State
    :param G_prev: previous associated covariance
    :param A: evolution matrix
    :param Q: evolution matrix
    :param B: evolution matrix
    :param u: evolution matrix
    """
    # Check inputs




    if B is None or u is None:
        X = A @ X_prev
    else:
        X = A @ X_prev + B @ u
    if Q is None:
        Q = np.zeros(A.shape)
    G = A @ G_prev @ A.T + Q

    return X, G

class KalmanFilter:
    """
    A Python implementation of the KalmanFilter (KF)

    """

    def __init__(self, X0: Union[List, np.ndarray], G0: Union[List, np.ndarray]):
        """
        Initialize the kalman filter mean (X) and covariance (G)

        :param X0: the initial estimate
        :param G0: the covariance associated to the initial estimate
        """
        if isinstance(X0, list):
            X0 = np.array(X0)
        if isinstance(G0, list):
            if len(G0) == len(X0) ** 2:
                G0 = np.array(G0)
            elif len(G0) == len(X0):
                G0 = np.diag(G0)
            else:
                raise ValueError(
                    f"Covariance size must be either the same size as X0 (diagonal),
                    or equivalent square matrix"
                )

        self.X = X0.copy()
        self.G = G0.copy()

    def predict(self):
        pass

    def _correct(self, Z: np.ndarray, H: np.ndarray, R: Union[None, np.ndarray] = None):
        if R is None:
            R = np.zeros((Z.size, Z.size))
        y = Z - H @ self.X
        S = H @ self.G @ H.T + R
        K = self.G @ H.T @ np.linalg.inv(S)

        I = np.eye(self.G.shape[0])

        self.X = self.X + K @ y
        self.G = (I - K @ H) @ self.G
    

class ExtendedKalmanFilter(KalmanFilter):

    def predict(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        ja: Callable,
        Q: Union[None, np.ndarray] = None,
        jB: Union[None, Callable] = None,
    ):
        pass

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
    for i in range(100):
        kf.predict(np.eye(2))
        obs = np.random.normal([1, 2], 0.3)
        kf.correct(obs, np.eye(2), 0.3**2 * np.eye(2))
    return np.linalg.norm(kf.X - [1, 2])


# %%


def test_kalman1d():
    kf = KalmanFilter(np.random.normal([1], 10), np.eye(1))
    for i in range(100):
        kf.predict(np.eye(1))
        obs = np.random.normal([1], 0.3)
        kf.correct(obs, np.eye(1), 0.3**2 * np.eye(1))
    return np.linalg.norm(kf.X - [1])


# %%


def test_kalman_odom_imu():
    # x, y, vx, vy, theta, w
    kf = KalmanFilter(np.random.normal([0, 0, 0, 0, 0, 0], 5), np.eye(6))
    for i in range(100):
        kf.predict([])

    # %%


if __name__ == "__main__":
    r = test_kalman()
    assert r <= 0.1

    r = test_kalman1d()
    assert r <= 0.1
