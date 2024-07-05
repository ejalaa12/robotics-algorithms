"""
contains a list of condition checking to validate the shape of numpy
matrices that are passed around
"""
import numpy as np

def is_square_matrix(m: np.ndarray) -> bool:
    return m.ndim == 2 and m.shape[0] == m.shape[1]

def state_and_covariance_shapes_compatible(state: np.ndarray, cov: np.ndarray) -> bool:
    return is_square_matrix(cov) and state.size == cov.shape[0]


