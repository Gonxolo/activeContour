import numpy as np
from math import ceil


def polygon_perimeter(x: list, y: list, xyFactor: list = [1.0, 1.0]) -> np.ndarray:
    
    """Computes the perimeter for a given polygon (list of 2D points).
        The polygon is not required to be closed.

    Parameters
    ----------
    x : list
        The input polygon x-coordinate array.
    y : list
        The input polygon y-coordinate array.
    xyFactor : list, optional
        Scaling factor in x and y (array of 2 elements).

    Returns
    -------
    np.ndarray
        The perimeter of the given polygon.
    """

    if len(x) == 0: 
        return -1.0
    
    if len(x) == 1: 
        return 0.0

    if len(y) == 0: 
        return -1.0
    
    if len(y) == 1: 
        return 0.0

    if len(xyFactor) == 0:
        xyFactor = np.array([1.0, 1.0])
    
    if len(xyFactor) == 1:
        xyFactor = np.array([xyFactor, xyFactor])
    
    return np.sum(np.sqrt(np.square((np.roll(x, -1) - x) * xyFactor[0]) + 
                            np.square((np.roll(y, -1) - y) * xyFactor[1])))

def calcNorm_L1ForVector(vec):
    """Computes the L1-norm for a vector of n components.

    Parameters
    ----------
    vec : np.ndarray
        vector

    Returns
    -------
    float
        Value of the L1-norm for a vector of n components.
    """
    return np.sum(np.abs(vec))

def calcNorm_L2ForVector(vec):
    """Computes the L2-norm for a vector of n components.

    Parameters
    ----------
    vec : np.ndarray
        vector

    Returns
    -------
    float
        Value of the L2-norm for a vector of n components.
    """
    return np.sqrt(np.sum(np.square(vec)))

def calcNorm_LInfiniteForVector(vec):
    """Computes the L-infinity-norm for a vector of n components.

    Parameters
    ----------
    vec : np.ndarray
        vector

    Returns
    -------
    float
        Value of the L-infinity-norm for a vector of n components.
    """
    return np.amax(np.abs(vec))

def polygon_line_sample(x, y, n_pts = 16, f_close_output = None,
                        flag_vector = None, interp_flag_vector = None, 
                        f_force_zero_flag_vector = None) -> np.ndarray:

    x_in = np.array(x)
    y_in = np.array(y)

    if len(x_in.shape) > 1:
        x_in = np.squeeze(x_in)

    if len(y_in.shape) > 1:
        y_in = np.squeeze(y_in)
    
    n_segments = len(x_in) - int(not bool(f_close_output))

    if n_segments < 1: 
        return

    x_out = np.array([])
    y_out = np.array([])

    n_pts_seg = ceil(n_pts/n_segments)

    iter_n = len(x_in) 

    for i in range(iter_n - int(not bool(f_close_output))):
        x_out = np.concatenate((x_out, np.linspace(x_in[i], x_in[(i+1)%iter_n], num=n_pts_seg, endpoint=False)))
        y_out = np.concatenate((y_out, np.linspace(y_in[i], y_in[(i+1)%iter_n], num=n_pts_seg, endpoint=False)))

    if ((x_in[-1] != x_out[-1]) or (y_in[-1] != y_out[-1])) and not f_close_output:
        x_out = np.concatenate((x_out, [x_in[-1]]))
        y_out = np.concatenate((y_out, [y_in[-1]]))

    return x_out, y_out

def get_curv_d(x: np.ndarray, y: np.ndarray, f_close=True) -> np.ndarray:

    if len(x) <= 3 or len(y) <= 3:
        return -1

    gx = np.roll(x, -1) - np.array(x)
    gy = np.roll(y, -1) - np.array(y)
    gxx = np.roll(x, 1) - 2 * np.array(x) + np.roll(x, -1)
    gyy = np.roll(y, 1) - 2 * np.array(y) + np.roll(y, -1)
    curv = ((gx * gyy - gy * gxx) / (np.power((np.square(gx) + np.square(gy)), 1.5))) # * 2 / np.sqrt(2)

    curv[curv == -np.inf] = 0
    curv[curv == np.inf] = 0
    curv[np.isnan(curv)] = 0

    if not f_close:
        curv[0] = 0
        curv[-1] = 0

    return curv
