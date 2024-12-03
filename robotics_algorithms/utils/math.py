import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
