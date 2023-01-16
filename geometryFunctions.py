import numpy as np


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

def polygon_line_sample(x, y, n_points_per_pix = None, f_close_output = None,
                        flag_vector = None, interp_flag_vector = None, 
                        f_force_zero_flag_vector = None) -> np.ndarray:

    x_in = np.array(x)
    y_in = np.array(y)

    if len(x_in.shape) > 1:
        x_in = np.squeeze(x_in)

    if len(y_in.shape) > 1:
        y_in = np.squeeze(y_in)
    
    n_segments = len(x_in)

    if n_segments < 1: 
        return

    seg_limit = 2
    if bool(f_close_output):
        seg_limit = 1
        x_in = np.concatenate(x_in, [x_in[0]])
        y_in = np.concatenate(y_in, [y_in[0]])
    
    x_out = [x_in[0]]
    y_out = [y_in[0]]

    if not bool(n_points_per_pix):
        n_points_per_pix = 1

    for i in range(n_segments - seg_limit + 1):
        delta_x = x_in[i+1] - x_in[i]
        delta_y = y_in[i+1] - y_in[i]
        seg_len = np.sqrt(np.square(delta_x) + np.square(delta_y))
        n_pts_seg = np.ceil(n_points_per_pix * seg_len) + 1
        delta_x_seg = delta_x / (n_pts_seg-1)
        delta_y_seg = delta_y / (n_pts_seg-1)
        for j in range(1, int(n_pts_seg)):
            x_out = np.concatenate((x_out, [x_in[i] + j*delta_x_seg]))
            y_out = np.concatenate((y_out, [y_in[i] + j*delta_y_seg]))


    return x_out, y_out
