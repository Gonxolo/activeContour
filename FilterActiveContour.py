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

x = (np.arange(21)/20)*2*np.pi
y = np.sin(x)
cs = CubicSpline(x,y)
x2 = (np.arange(11)/11)*np.pi

print(cs(x2))

l = np.array([1,2,3])
s = np.array([0])
sl = np.concatenate((np.array([4]),l), axis = None)
#s = np.concatenate(0,l)
print("sl",sl)

a = np.zeros(4)
a[0]=1
a[2]=1
b = np.zeros(4)
b[3]=3
c=[[1*a],[2*b]]
c1= np.array([1*a,2*b])
print("a",a)
print("b",b)
print("c",c)
print("c1",c1)


p=np.arange(5)

print("p", p) #[0,1,2,3,4] en IDL lo mismo en python
print("p1", p[1:4]) #[1,2,3] en python y [1,2,3,4] en IDL
print("p2", p[4]) #4 en python y 4 en IDL
print("p3", p[1:]) #[1,2,3,4] en python y en IDL

x_out = np.array([1,2,3])
print("l",np.concatenate((x_out, np.array([x_out[0]]))))

x1=[0,1,2,3,4]
y1=[2,3,5,6,7]
dx = np.square(np.roll(x1,-1)-x1)
dy = np.square(np.roll(y1,-1)-y1)
res=np.power(dx + dy, 0.5)
per=np.sum(res)
print("res",res)
print("per",per)
