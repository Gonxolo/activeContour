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

def polygonParameter(xPolyCoords, yPolyCoords) -> float:
    #en geometryFunctions, utilizada en adjustContour
    return

x = (np.arange(21)/20)*2*np.pi
y = np.sin(x)
cs = CubicSpline(x,y)
x2 = (np.arange(11)/11)*np.pi

print(cs(x2))

l = np.array([1,2,3])
s = np.array([0])
sl = np.concatenate((np.array([4]),l), axis = None)
#s = np.concatenate(0,l)
print(sl)

