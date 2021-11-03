import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from math import cos, sin, pi, sqrt

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

#Penser à dessiner les axes en 3D pour aider à mieux comprendre la position de la caméra.
#Verif que camera pas dans bounding box ?

image_resolution = np.array([100, 100])
canvas_center = rotation_mat('z', pi/8) @ np.array([2, 0, 0])
canvas_size = np.array([1, 1])
pixel_size = canvas_size / image_resolution
canvas_look_at = np.array([0, 0, 0])
canvas_normal = normalize(canvas_look_at - canvas_center)
eye_to_canvas_distance = 1
eye = canvas_center - canvas_normal * eye_to_canvas_distance
#tangentes qui donne la direction selon x et selon y de l'image
#La tangente en x est parralèle au plan xy.
canvas_tangentx = normalize(np.array([canvas_normal[1], -canvas_normal[0], 0]))
canvas_tangenty = normalize(np.cross(canvas_tangentx, canvas_normal))
canvas_tangents = np.array([canvas_tangentx, canvas_tangenty])
canvas_origin = canvas_center - canvas_tangents[0] * canvas_size[0] / 2 - canvas_tangents[1] * canvas_size[1]/2
'''canvas_corners = [canvas_origin,
    canvas_origin + canvas_tangents[0] * canvas_size[0],
    canvas_origin + canvas_tangents[1] * canvas_size[1],
    canvas_origin + canvas_tangents[0] * canvas_size[0] + canvas_tangents[1] * canvas_size[1]]'''

#Data variables
#On suppose la bounding box centrée en (0, 0, 0) et orientée selon les axes x, y, z.
#bb_origin = np.array([0, 0, 0])
bb_center = np.array([0, 0, 0])
data_resolution = np.array([10, 10, 10])
bb_size = np.array([1, 1, 1])
voxel_size = bb_size / data_resolution
bb_tangents = np.eye(3)
#bb_center = bb_origin + 1/2 * bb_tangents @ bb_size

'''
   1-------3       ^  z      
  /|      /|       |         
 / |     / |       |--->  y  
5--+----7  |      /          
|  |    |  |     v  x       
|  |    |  |
|  |    |  |     Ordre des coins de bb_corners
|  0----+--2
| /     | /
|/      |/
4-------6
'''
bb_corners = np.array([bb_center -
    (-1)**x * bb_tangents[0] * bb_size[0] / 2 -
    (-1)**y * bb_tangents[1] * bb_size[1] / 2 -
    (-1)**z * bb_tangents[2] * bb_size[2] / 2
    for x in range(2) for y in range(2) for z in range(2)])
_faces_corners_idx = [
    [0, 2, 1, 3],
    [4, 6, 5, 7],
    [0, 4, 1, 5],
    [2, 6, 3, 7],
    [0, 2, 4, 6],
    [1, 3, 5, 7]]
bb_faces_corners = [[bb_corners[i] for i in corners_idx] for corners_idx in _faces_corners_idx]
bb_faces_normals = [np.array([1, 0, 0]), np.array([1, 0, 0]),
    np.array([0, 1, 0]), np.array([0, 1, 0]),
    np.array([0, 0, 1]), np.array([0, 0, 1])]
bb_faces_rect_limits = []
bb_faces_rect_basis = []
for face_corners in bb_faces_corners:
    bottomleft, bottomright, topleft, topright = face_corners
    vecx = bottomright - bottomleft
    vecy = topleft - bottomleft
    rect_basis = np.array([normalize(vecx), normalize(vecy)])
    lower_point = rect_basis @ bottomleft
    upper_point = rect_basis @ topright
    bb_faces_rect_limits.append((lower_point, upper_point))
    bb_faces_rect_basis.append(rect_basis)

#Other variables
sky_colour = np.array([0.0, 0.0, 0.0, 1.0])

def global_pixel_coord(pixel):
    #We consider the coordinate of a pixel to be the middle of its square
    offset = pixel_size * (pixel + np.array([1/2, 1/2]))
    return canvas_origin + canvas_tangents[0] * offset[0] + canvas_tangents[1] * offset[1]

def ray_marching():
    #cf Emma
    pass

def ray_plane_intersect(ray_point, ray_dir, face_point, face_normal):
    return ray_point + (((face_point - ray_point) @ face_normal) / (ray_dir @ face_normal) * ray_dir)

def point_in_rect(point, rect_basis, lower_point, upper_point):
    #We assume the point is on the plane of the rectangle.
    #We want to compute if it is inside.
    local_point = rect_basis @ point
    return lower_point[0] <= local_point[0] <= upper_point[0] and lower_point[1] <= local_point[1] <= upper_point[1]

def ray_box_intersect(ray_point, ray_dir):
    '''Etant donnée un rayon, on renvoie les points d'intersection (potentiellement dupliqués)
    du rayon avec chaque face de la bounding box.'''
    #On cherche tous les points d'intersection entre le rayon et la boite.
    #Ce sont exactement les points d'intersection du rayon avec le plan engendré par
    #chacune des faces, qui sont contenus dans cette-même face.
    intersections = []
    for face_points, face_normal, rect_basis, rect_limits in zip(bb_faces_corners, bb_faces_normals, bb_faces_rect_basis, bb_faces_rect_limits):
        inter = ray_plane_intersect(ray_point, ray_dir, face_points[0], face_normal)
        if point_in_rect(inter, rect_basis, *rect_limits):
            intersections.append(inter)
    return intersections

def compute_image():
    iter_pixels = [(i, j) for i in range(image_resolution[0]) for j in range(image_resolution[1])]
    #shuffle(iter_pixels)
    image = np.zeros((image_resolution[0], image_resolution[1], 4))
    for pixel in iter_pixels:
        #Trouver le rayon venant du point
        ray_canvas_inter = global_pixel_coord(pixel)
        ray_dir = normalize(ray_canvas_inter - eye)
        #Chercher le point d'entrée et de sortie du rayon avec la bounding box.
        intersections = ray_box_intersect(eye, ray_dir)
        #Cas 1 : on n'a pas de point d'intersection.
        if len(intersections) == 0:
            image[pixel] = sky_colour
        #Cas 2 : on intersecte la bounding box.
        else:
            '''#Le point par lequel on y rentre est celui le plus proche de l'oeil.
            #Celui par lequel on sort est le plus éloigné.
            entry_point = min(intersections, key = lambda point : np.linalg.norm(point - eye))
            exit_point  = max(intersections, key = lambda point : np.linalg.norm(point - eye))
            #cf Emma
            colour = ray_marching()
            image[pixel] = colour'''
            image[pixel] = np.array([0.0, 1.0, 0.0, 1.0])
        
    return image
    

def show_image(image):
    #imshow considère que l'axe x est vertical et l'axe y horizontal.
    #Il faut donc transposer l'écran, sans toucher aux vecteur des couleurs
    #De plus, il place l'origine en haut à gauche, alors que le notre est bas à gauche.
    #D'ou l'argument origin='lower'.
    plt.imshow(np.transpose(image, axes=(1, 0, 2)), origin='lower')
    plt.show()

'''def main():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        image = compute_image()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    #stats.print_stats()
    stats.dump_stats(filename='profiling5.prof')
    show_image(image)


if __name__ == '__main__':
    main()'''