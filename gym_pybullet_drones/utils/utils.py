"""General use functions.
"""
import time
import argparse
import numpy as np
from scipy.optimize import nnls

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")


def unit_vector(vector):
    return np.array(vector) / (1+np.linalg.norm(vector))

def randrange(a, b):
    """Random number between a and b."""
    return a + np.random.random() * (b - a)


def px_to_grid(px_pos):
    """Convert pixel position to grid position."""
    return np.array([px_pos[0] / params.COL, px_pos[1] / params.ROW])


def grid_to_px(grid_pos):
    """Convert grid position to pixel position."""
    return np.array([grid_pos[0] * params.COL, grid_pos[1] * params.ROW])


def norm(vector):
    """Compute the norm of a vector."""
    return math.sqrt(vector[0]**2 + vector[1]**2)


def norm2(vector):
    """Compute the square norm of a vector."""
    return vector[0] * vector[0] + vector[1] * vector[1]


def dist2(a, b):
    """Return the square distance between two vectors.

    Parameters
    ----------
    a : np.array
    b : np.array
    """
    return norm2(a - b)


def dist(a, b):
    """Return the distance between two vectors.

    Parameters
    ----------
    a : np.array
    b : np.array
    """
    return norm(a - b)


def normalize(vector, pre_computed=None):
    """Return the normalized version of a vector.

    Parameters
    ----------
    vector : np.array
    pre_computed : float, optional
        The pre-computed norm for optimization. If not given, the norm
        will be computed.
    """
    n = pre_computed if pre_computed is not None else norm(vector)
    if n < 1e-13:
        return np.zeros(2)
    else:
        return np.array(vector) / n


def truncate(vector, max_length):
    """Truncate the length of a vector to a maximum value."""
    n = norm(vector)
    if n > max_length:
        return normalize(vector, pre_computed=n) * max_length
    else:
        return vector


def unit_vector(vector):
    return np.array(vector) / (1+np.linalg.norm(vector))