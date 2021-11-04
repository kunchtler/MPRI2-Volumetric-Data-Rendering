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
    return np.array([1.0, 1.0, 1.0, sqrt(xn)])

def ease_out(x):
    xn = x/255
    return np.array([1.0, 1.0, 1.0, sqrt(1 - (1-xn)**2)])