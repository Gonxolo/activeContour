import numpy as np
import matplotlib.pyplot as plt
def square(n,f):
    x1 = np.arange(n)/n*f
    x2 = np.ones(n+1)*(n)/n*f
    x3 = x1[::-1]
    x4 = np.zeros(n)/n*f

    y1 = np.zeros(n)/n*f
    y2 = np.arange(n+1)/n*f
    y3 = np.ones(n)*n/n*f
    y41 = np.arange(n)*f
    y4 = y41[::-1]/n
    x12 = np.concatenate((x1, x2))
    x34 = np.concatenate((x3, x4))
    x = np.concatenate((x12, x34))

    y12 = np.concatenate((y1, y2))
    y34 = np.concatenate((y3, y4))
    y = np.concatenate((y12, y34))

    return np.array([x, y])

p=square(5,3)
print("square",square(10,5))
alpha = 0.05
beta = 0.3
"""

0,3  1,3  2,3  3,3
0,2            3,2
0,1            3,1
0,0  1,0  2,0  3,0
"""
plt.scatter(p[0],p[1])
#plt.title('\u03B1=%f' %alpha+'\u03B2=%f' %beta)
plt.title('x= '+str(alpha)+', y = '+str(beta))
plt.plot(p[0],p[1])
plt.show()

