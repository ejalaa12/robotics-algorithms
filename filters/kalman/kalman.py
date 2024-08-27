# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:59:53 2021

@author: ejalaa
"""

import numpy as np
from typing import Union, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass


def validate_predict_size(A, x, B, u):
    n = x.size
    m = u.size

    if A.shape != (n, n):
        raise ValueError(
            f"A shape must be ({n}, {n}) like x, got {A.shape=} vs {x.shape=}"
        )
    if B.shape != (n, m):
        raise ValueError(
            f"B shape must be ({n}, {m}) like u, got {B.shape=} vs {u.shape=}"
        )


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
    # compute input sizes (for size checking)
    n = X_prev.size
    m = 0
    if u:
        m = u.size

    # fill in B and u if None
    if B is None or u is None:
        B = np.zeros((n, m))
        u = np.zeros((m, 1))

    # fill in Q if None
    if Q is None:
        Q = np.zeros(A.shape)

    # check sizes compatibility
    validate_predict_size(A, X_prev, B, u)

    X = A @ X_prev + B @ u
    G = A @ G_prev @ A.T + Q

    return X, G


def kalman_correct(X, G, H, Z, R):
    if R is None:
        R = np.zeros((Z.size, Z.size))
    y = Z - H @ X
    S = H @ G @ H.T + R
    K = G @ H.T @ np.linalg.inv(S)

    I = np.eye(G.shape[0])

    new_X = X + K @ y
    new_G = (I - K @ H) @ G

    return new_X, new_G


class State(np.ndarray):
    """
    todo: here are some notes:
    - https://numpy.org/doc/stable/user/basics.subclassing.html
    - https://realpython.com/python-magic-methods/
    """
    def __init__(self, array, names):
        assert array.size == len(names)


@dataclass
class GaussianDistribution:
    mean: np.ndarray
    covariance: np.ndarray


class Measurement:
    pass


@dataclass
class Observation(ABC):
    distribution: GaussianDistribution


class LinearObservation(Observation):
    @abstractmethod
    def get_observation_matrix(self) -> Union[np.ndarray, None]:
        pass


class LinearizedObservation(Observation):

    @abstractmethod
    def observe(self, template_X):
        pass


class Filter(ABC):
    @abstractmethod
    def add_measurement(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def correct(self):
        pass


class KalmanFilter(ABC):
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
                    f"Covariance size must be either the same size as X0 (diagonal), or equivalent square matrix"
                )

        self.distribution = GaussianDistribution(X0, G0)

    @abstractmethod
    def compute_state_transition_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    def compute_control_input_model_matrix(self) -> Union[np.ndarray, None]:
        pass

    @abstractmethod
    def compute_process_noise(self) -> Union[np.ndarray, None]:
        pass

    def predict(self, u: Union[np.ndarray, None] = None):
        A = self.compute_state_transition_matrix()
        B = self.compute_control_input_model_matrix()
        Q = self.compute_process_noise()
        x, g = self.distribution.mean, self.distribution.covariance
        self.distribution.mean, self.distribution.covariance = kalman_predict(
            self.distribution.mean, self.distribution.covariance, A, Q, B, u
        )

    def correct(self, observation: LinearObservation):
        obs = observation.distribution.mean
        H = observation.get_observation_matrix()
        R = observation.distribution.covariance
        self.distribution.mean, self.distribution.covariance = kalman_correct(self.distribution.mean, self.distribution.covariance, H, obs, R)


# class ExtendedKalmanFilter(KalmanFilter):
#     """
#     TODO(ejalaa12): replace with LinearizedObservation
#     """

#     def predict2(
#         self,
#         f: Callable[[np.ndarray], np.ndarray],
#         ja: Callable,
#         Q: Union[None, np.ndarray] = None,
#         jB: Union[None, Callable] = None,
#     ):
#         pass

#     def correct(
#         self,
#         Z: np.ndarray,
#         h: Callable,
#         H: np.ndarray,
#         R: Union[None, np.ndarray] = None,
#     ):
#         if R is None:
#             R = np.zeros((Z.size, Z.size))
#         y = Z - h(self.X)
#         S = H @ self.G @ H.T + R
#         K = self.G @ H.T @ np.linalg.inv(S)

#         I = np.eye(self.G.shape[0])

#         self.X = self.X + K @ y
#         self.G = (I - K @ H) @ self.G


# %%


# def test_kalman():
#     kf = KalmanFilter(np.random.normal([1, 2], 10), np.eye(2))
#     for i in range(100):
#         kf.predict(np.eye(2))
#         obs = np.random.normal([1, 2], 0.3)
#         kf.correct(obs, np.eye(2), 0.3**2 * np.eye(2))
#     return np.linalg.norm(kf.X - [1, 2])


# %%


# def test_kalman1d():
#     kf = KalmanFilter(np.random.normal([1], 10), np.eye(1))
#     for i in range(100):
#         kf.predict(np.eye(1))
#         obs = np.random.normal([1], 0.3)
#         kf.correct(obs, np.eye(1), 0.3**2 * np.eye(1))
#     return np.linalg.norm(kf.X - [1])


# %%


def test_kalman_odom_imu():
    # x, y, vx, vy, theta, w
    kf = KalmanFilter(np.random.normal([0, 0, 0, 0, 0, 0], 5), np.eye(6))
    # for i in range(100):
    #     kf.predict([])

    # %%


if __name__ == "__main__":
    pass
    # r = test_kalman()
    # assert r <= 0.1

    # r = test_kalman1d()
    # assert r <= 0.1
