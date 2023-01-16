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