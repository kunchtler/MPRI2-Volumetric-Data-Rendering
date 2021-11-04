import numpy as np

#Toutes ces fonctions sont définies dans le cube [0, 1]**3
def initialize_data(data_nb_voxels, init_value):
    '''Crée des données dont la longueur dans chaque direction est donnée par data_nb_voxels,
    et dont la bounding box et de taille bb_size.
    data indique la valeur de l'élément (i, j, k) (initialement init_value).
    coord indique la position de l'élément (i, j, k) dans le cube [0, 1]**3.'''
    data = np.full(data_nb_voxels, init_value, dtype=np.uint8)
    coord = np.array([[[(np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) / data_nb_voxels
        for i in range(data_nb_voxels[0])]
        for j in range(data_nb_voxels[1])]
        for k in range(data_nb_voxels[2])])
    return data, coord

def add_sphere(center, radius, value, data, coord):
    mask = (np.linalg.norm(coord - center, axis = 3) <= radius)
    data[mask] = value

def add_rect(corner1, corner2, value, data, coord):
    mask = corner1 <= coord <= corner2
    data[mask] = value

def sample_data1(data_nb_voxels, colour):
    '''Just a sphere'''
    data, coord = initialize_data(data_nb_voxels, 0)
    add_sphere(np.array([0.5, 0.5, 0.5]), 0.4, colour, data, coord)
    return data