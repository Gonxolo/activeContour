#funciones en filter activeContour implementadas
import numpy as np
from scipy.interpolate import CubicSpline

def obj_valid() -> bool:
    #checkea si objeto es válido
    return

def getpObjectBorderPolygonList() -> list:
    #voxelObject = self->makePixelObjectInVoxel() (llama a método makePixelObjectInVoxel())
    #labelRegionData = s_2DLabelObjectAndHoles() (llama a método s_2DLabelObjectAndHoles())
    #s_rand4()
    #chainCodeVector = self -> makeObjectBorderChainCode()
    #polygon = self -> makePolygonFromChainCodeVector()
    #polygon = self ->inflatePolygonFromChainCodeVector()
    return

def convexHullPolygon(xCoords, yCoords) -> list:
    return

x = (np.arange(200)/199)*2*np.pi
y = np.sin(x)
cs = CubicSpline(x,y)
x2 = (np.arange(30)/30)*np.pi
