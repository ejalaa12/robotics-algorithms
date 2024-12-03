import numpy as np
from typing import Union
from robotics_algorithms.filters.kalman.validators import state_and_covariance_shapes_compatible
import logging


def _kalman_predict(
    x_prev: np.ndarray,
    g_prev: np.ndarray,
    a: np.ndarray,
    q: Union[None, np.ndarray] = None,
    b: Union[None, np.ndarray] = None,
    u: Union[None, np.ndarray] = None,
):
    pass

def kalman_predict(
    x_prev: np.ndarray,
    g_prev: np.ndarray,
    a: np.ndarray,
    q: Union[None, np.ndarray] = None,
    b: Union[None, np.ndarray] = None,
    u: Union[None, np.ndarray] = None,
):
    """
    Prediction step of the KalmanFilter.
    This uses the transition/evolution model to propagate the state and covariance
    at the next step

    :param x_prev: previous State
    :param g_prev: previous associated covariance
    :param a: evolution matrix
    :param q: evolution matrix
    :param b: evolution matrix
    :param u: evolution matrix
    """
    # Check inputs
    if not state_and_covariance_shapes_compatible(x_prev, g_prev):
        raise ValueError(f"x and g should have the same number of rows: {x_prev.size} vs {g_prev[0].size}")
    if a.shape != g_prev.shape:
        raise ValueError(f"Incompatible size between matrix A ({a.shape}) and g ({g_prev.shape})")
    if b is not None:
        if b.ndim != 2:
            if b.ndim == 1 and b.size != x_prev.size:
                raise ValueError(f"Incompatible size between B ({b.shape}) and x ({x_prev.shape}), they should have the same number of rows")
            raise ValueError(f"B should have up to 2 dimensions (got {b.ndim} with shape: {b.shape})")

        if u is None:
            logging.warning(f"B matrix is given but not u")
        else:
            #wip
            pass


    if b is None or u is None:
        x = a @ x_prev
    else:
        x = a @ x_prev + b @ u
    if q is None:
        q = np.zeros(a.shape)
    g = a @ g_prev @ a.T + q

    return x, g


def kalman_correct(state, state_cov, observation_model, observation, observation_cov):
    x = state
    g = state_cov
    h = observation_model
    z = observation
    r = observation_cov

    if r is None:
        r = np.zeros((z.size, z.size))
    y = z - h @ x
    s = h @ g @ h.T + r
    k = g @ h.T @ np.linalg.inv(s)

    i = np.eye(g.shape[0])

    new_x = x + k @ y
    new_g = (i - k @ h) @ g

    return new_x, new_g, y, s, k
