import numpy as np
from math import sqrt

def colour_fun1(x):
    xn = x/255
    return np.array([xn, xn, xn, xn])

def colour_fun2(x):
    xn = x/255
    return np.array([1.0, 0.0, 0.0, xn])

def colour_fun3(x):
    xn = x/255
    return np.array([1.0, 1.0, 1.0, xn**2])

def ease_out(x):
    xn = x/255
    return np.array([1.0, 1.0, 1.0, sqrt(1 - (1-xn)**2)])

def colour_C60(x):
    c1 = np.array([1.0, 1.0, 0.0, smoothstep(x, x_min=200, x_max=255)])
    c2 = np.array([0.0, 255/255, 154/255, smoothstep(x, x_min=30, x_max=200, y_min=0, y_max=0.5)])
    return blend_colours(c1, c2)

def smoothstep(x, x_min=0, x_max=1, y_min=0, y_max=1):
    #https://en.wikipedia.org/wiki/Smoothstep
    if x_min >= x:
        y = y_min
    elif x_max <= x:
        y = y_max
    else:
        xp = (x-x_min) / (x_max-x_min)
        y = y_min + (-2*xp**3 + 3*xp**2)*(y_max - y_min)
    return y

def blend_colours(c1, c2):
    a = c1[3] / (c1[3] + c2[3])
    return c1*a + c2*(1-a)

'''def blend_colours(c1, c2):
    alpha = 1 - (1 - c1[3]) * (1 - c2[3])
    c = c1[:3]*c1[3]/alpha + c2[:3]*c2[3]*(1-c1[3])/alpha
    return np.array([c[0], c[1], c[2], alpha])

c1 = np.array([1.0, 0.0, 0.0, 0.5])
c2 = np.array([0.0, 1.0, 1.0, 0.4])
print(blend_colours(c1, c2))
print(blend_colours(c2, c1))'''