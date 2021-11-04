import numpy as np
from math import cos, sin, sqrt
import matplotlib.pyplot as plt

def frac(x):
    return x - int(x)

class NullVector(Exception):
    pass

def normalize(vector):
    norm = sqrt(vector @ vector)
    if norm == 0:
        raise NullVector
    return vector / norm

def rotation_mat(axis, theta):
    if axis == 'x':
        return np.array([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
    elif axis == 'y':
        return np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    elif axis == 'z':
        return np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])

def vector_symmetric(u, v):
    '''Symmétrique du vecteur u par rapport au vecteur v'''
    return 2 * (u @ v) / (v @ v) * v - u

def show_image(image):
    #imshow considère que l'axe x est vertical et l'axe y horizontal.
    #Il faut donc transposer l'écran, sans toucher aux vecteur des couleurs
    #De plus, il place l'origine en haut à gauche, alors que le notre est bas à gauche.
    #D'ou l'argument origin='lower'.
    plt.imshow(np.transpose(image, axes=(1, 0, 2)), origin='lower', interpolation='nearest')
    plt.show()