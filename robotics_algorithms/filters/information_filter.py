# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 7 2024
@author: ejalaa
"""

import numpy as np
from typing import Union, Callable


class InformationFilter:
    def __init__(self, mu0: np.ndarray, cov0: np.ndarray):
        self.mu = mu0.copy()
        self.cov = cov0.copy()
        # Information form
        self.inf_mat = np.linalg.inv(cov0)
        self.inf_vec = self.inf_mat @ mu0

    def predict(
        self,
        A: np.ndarray,
        Q: Union[None, np.ndarray] = None,
        B: Union[None, np.ndarray] = None,
        u: Union[None, np.ndarray] = None,
    ):
        if Q is None:
            Q = np.zeros(A.shape)

        self.cov = np.linalg.inv(self.inf_mat)
        self.inf_mat = np.linalg.inv(A @ self.cov @ A.T + Q)

        input_update = np.zeros_like(self.inf_vec)
        if B is None or u is None:
            input_update = np.zeros_like(self.inf_vec)
        else:
            input_update = B.T @ u
        self.inf_vec = self.inf_mat @ (A.T @ self.cov @ self.inf_vec + input_update)

    def correct(self, Z: np.ndarray, H: np.ndarray, R: np.ndarray):
        r1 = np.linalg.inv(R)

        self.inf_mat = H.T @ r1 @ H + self.inf_mat
        self.inf_vec = H.T @ r1 @ Z + self.inf_vec
