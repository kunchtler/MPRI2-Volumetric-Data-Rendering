import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

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
    c1 = np.array([1.0, 1.0, 0.0, smoothstep(x, x_min=190, x_max=255)])
    alpha = smoothmountain(x, 100, 120, 0.0, 0.04, 190, 200, 0.04, 0.0)
    c2 = np.array([0.0, 255/255, 154/255, alpha])
    return blend_colours(c1, c2)

def colour_Foot(x, bone_colour=(249/255, 255/255, 171/255)):
    alpha = smoothmountain(x, 123, 130, 0.0, 1.0, 200, 230, 1.0, 0.0)
    c1 = np.array([*bone_colour, alpha])
    #return c1
    alpha = smoothmountain(x, 60, 65, 0.0, 0.02, 105, 110, 0.02, 0.0)
    c2 = np.array([112/255, 58/255, 16/255, alpha])
    return blend_colours(c1, c2)

def smoothstep(x, x_min, x_max, y_min, y_max):
    #https://en.wikipedia.org/wiki/Smoothstep
    if x_min >= x:
        y = y_min
    elif x_max <= x:
        y = y_max
    else:
        xp = (x-x_min) / (x_max-x_min)
        y = y_min + (-2*xp**3 + 3*xp**2)*(y_max - y_min)
    return y

def smoothmountain(x, x_min1, x_max1, y_min1, y_max1, x_min2, x_max2, y_min2, y_max2):
    return min(smoothstep(x, x_min1, x_max1, y_min1, y_max1), smoothstep(x, x_min2, x_max2, y_min2, y_max2))

def blend_colours(c1, c2):
    if c1[3] + c2[3] == 0:
        return np.array([0, 0, 0, 0])
    a = c1[3] / (c1[3] + c2[3])
    return c1*a + c2*(1-a)

def plot_color_Foot():
    x = np.arange(256)
    y1 = [smoothmountain(elem, 123, 130, 0.0, 1.0, 200, 230, 1.0, 0.0) for elem in x]
    y2 = [smoothmountain(elem, 60, 65, 0.0, 0.02, 105, 110, 0.02, 0.0) for elem in x]
    plt.plot(x, y1, color=(249/255, 255/255, 171/255))
    plt.plot(x, y2, color=(112/255, 58/255, 16/255))
    plt.xlabel(r"$f$")
    plt.ylabel(r"Transparence")
    plt.title(r"Transparence par valeur de $f$ par couleur")
    ax = plt.gca()
    ax.set_facecolor((0.5, 0.5, 0.5))
    plt.show()

'''def blend_colours(c1, c2):
    alpha = 1 - (1 - c1[3]) * (1 - c2[3])
    c = c1[:3]*c1[3]/alpha + c2[:3]*c2[3]*(1-c1[3])/alpha
    return np.array([c[0], c[1], c[2], alpha])

c1 = np.array([1.0, 0.0, 0.0, 0.5])
c2 = np.array([0.0, 1.0, 1.0, 0.4])
print(blend_colours(c1, c2))
print(blend_colours(c2, c1))'''