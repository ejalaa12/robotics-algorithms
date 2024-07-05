import numpy as np
from typing import Union, Callable, List
from filters.kalman.validators import state_and_covariance_shapes_compatible
import logging


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
    if not state_and_covariance_shapes_compatible(X_prev, G_prev):
        raise ValueError(f"X and G should have the same number of rows: {X_prev.size} vs {G_prev[0].size}")
    if A.shape != G_prev.shape:
        raise ValueError(f"Incompatible size between matrix A ({A.shape}) and G ({G_prev.shape})")
    if B is not None:
        if B.ndim != 2:
            if B.ndim == 1 and B.size != X_prev.size:
                raise ValueError(f"Incompatible size between B ({B.shape}) and X ({X_prev.shape}), they should have the same number of rows")
            raise ValueError(f"B should have up to 2 dimensions (got {B.ndim} with shape: {B.shape})")

        if u is None:
            logging.warning(f"B matrix is given but not u")
        else:
            #wip
            pass


    if B is None or u is None:
        X = A @ X_prev
    else:
        X = A @ X_prev + B @ u
    if Q is None:
        Q = np.zeros(A.shape)
    G = A @ G_prev @ A.T + Q

    return X, G
