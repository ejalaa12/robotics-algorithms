import numpy as np


def adj(w):
    """
    Given a 1x3 vector
    returns the skew-antisymetric matrix associated,
    which is called the adjoint
    """
    x, y, z = w
    # fmt: off
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    # fmt: on


def adji(w):
    """
    Opposite of method above
    """
    return np.array([-w[1, 2], w[0, 2], -w[0, 1]])


def rodrigues(w_hat):
    """
    w_hat is an element of so<3> (lie algebra)
    """
    w = adji(w_hat)
    theta = np.linalg.norm(w)
    return (
        np.eye(3)
        + np.sin(theta) * (w_hat / theta)
        + (1 - np.cos(theta)) * np.linalg.matrix_power(w_hat, 2) / theta**2
    )


def lie_log(r):
    theta = np.arccos((np.trace(r) - 1) / 2)
    return theta * (r - r.T) / (2 * np.sin(theta))


def lie_exp(w_hat):
    return rodrigues(w_hat)
