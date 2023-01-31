import numpy as np

def hausdorffDistanceFor2Dpoints(x1, y1, x2, y2):
    """Computes the Hausdorff Distance for 2D points.

    Parameters
    ----------
    x1 : np.ndarray
        x coordinate of point 1
    y1 : np.ndarray
        y coordinate of point 1
    x2 : np.ndarray
        x coordinate of point 2
    y2 : np.ndarray
        y coordinate of point 2

    Returns
    -------
    float
        Value of the Hausdorff Distance for 2D points
    """
    n_verts1 = len(x1)
    n_verts2 = len(x2)
    dist12 = np.zeros(n_verts1)
    dist21 = np.zeros(n_verts2)

    for i in range(n_verts1):
        dist12[i] = np.min(np.sqrt(np.square(x2 - x1[i]) + np.square(y2 - y1[i])))
    
    for j in range(n_verts2):
        dist21[j] = np.min(np.sqrt(np.square(x1 - x2[i]) + np.square(y1 - y2[i])))
    
    if np.max(dist12) > np.max(dist21):
        return np.amax(dist12)
    else:
        return np.amax(dist21)