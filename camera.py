import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from math import cos, sin, pi, sqrt
from useful_functions import frac, NullVector, normalize, rotation_mat, show_image, vector_symmetric


image_resolution = np.array([100, 100])
#canvas_center = rotation_mat('z', pi/8) @ np.array([2, 0, 0])
canvas_center = rotation_mat('z', pi/8) @ np.array([0.75, 0, 0])
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
import dataset_generation as gen
data = gen.sample_data1(np.full((3,), 64), 255)

#On suppose la bounding box centrée en (0, 0, 0) et orientée selon les axes x, y, z.
#bb_origin = np.array([0, 0, 0])
bb_center = np.array([0, 0, 0])
data_resolution = np.array(data.shape)
#Attention, ici les données sont non pas au centre mais dans les coins d'un voxel.
nb_voxels = data_resolution - 1
bb_size = np.array([1, 1, 1])
voxel_size = bb_size / nb_voxels
bb_tangents = np.eye(3)
bb_voxel_sized_basis = nb_voxels / bb_size * bb_tangents
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
bb_origin = bb_corners[0]
#Pre-computing some useful things about faces needed for later.
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

slight_offset = 0.000000001

#Other variables
def local_bb_coord(point):
    return bb_voxel_sized_basis @ (point - bb_origin)

I_amb = np.array([0.5, 0.5, 0.5])
I_diff = np.array([0.5, 0.5, 0.5])
I_spec = np.ones((3,))
lum_pos = np.array([0, 5, 5])
lum_pos_bb_coord = local_bb_coord(lum_pos)
shininess = 30
#sky_colour = np.array([121, 248, 248, 255])/np.array([255, 255, 255, 255])
sky_colour = np.array([0, 0, 0, 1])
step_size = 1 / 2

import colour_fun as cf
color_function = cf.colour_fun2

def global_pixel_coord(pixel):
    #We consider the coordinate of a pixel to be the middle of its square
    offset = pixel_size * (pixel + np.array([1/2, 1/2]))
    return canvas_origin + canvas_tangents[0] * offset[0] + canvas_tangents[1] * offset[1]

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

def interpolate(pt):
    (a,b,c) = pt
    x = int(a)
    y = int(b)
    z = int(c)
    #On prend le coin le plus proche du centre de la bb afin que data[x+1, y, z] soit bien défini.
    #(ie x+1 < data_resolution[0])
    c000=data[x,y,z]
    c100=data[x+1,y,z]
    c001=data[x,y,z+1]
    c101=data[x+1,y,z+1]
    c010=data[x,y+1,z]
    c110=data[x+1,y+1,z]
    c011=data[x,y+1,z+1]
    c111=data[x+1,y+1,z+1]
    xd = frac(a)
    yd = frac(b)
    zd = frac(c)
    c00 = c000*(1-xd) + c100*xd
    c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd
    c11 = c011*(1-xd) + c111*xd
    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd
    c = c0*(1-zd) + c1*zd
    return c

def normal_to_isosurface(u):
    (x,y,z) = u
    ux_m1 = (max(0 + slight_offset, x-1), y, z)
    uy_m1 = (x, max(0 + slight_offset, y-1), z)
    uz_m1 = (x, y, max(0 + slight_offset, z-1))
    ux_p1 = (min(nb_voxels[0] - slight_offset, x+1), y, z)
    uy_p1 = (x, min(nb_voxels[1] - slight_offset, y+1), z)
    uz_p1 = (x, y, min(nb_voxels[2] - slight_offset, z+1))
    g_x = (interpolate(ux_m1) - interpolate(ux_p1)) / (ux_p1[0] - ux_m1[0])
    g_y = (interpolate(uy_m1) - interpolate(uy_p1)) / (uy_p1[1] - uy_m1[1])
    g_z = (interpolate(uz_m1) - interpolate(uz_p1)) / (uz_p1[2] - uz_m1[2])
    #Il est possible que la normale calculée soit nulle.
    #Par manque de chance, ou car l'isosurface est en fait un volume.
    #Si c'est le cas, on renvoie le vecteur nul pour faire tomber le facteur l'utilisant
    #dans le calcul des ombres.
    try:
        return normalize(np.array([g_x, g_y, g_z]))
    except NullVector:
        return np.array([0, 0, 0])



def compute_color(dir_r, pt_d, pt_f):
    #Coordonnées locales dans la bb.
    #On offset très légèrement les points d'entrée et de sortie pour éviter des problèmes
    #quand les données à interpoler sont pile sur une face de la bb.
    #Mais si les points d'entrée et de sortie sont vraiment trop proches,
    #On abandonne. Oui, c'est sale.
    d = sqrt((pt_f - pt_d) @ (pt_f - pt_d))
    if d <= 2*slight_offset:
        return sky_colour
    pt_d += slight_offset*dir_r
    pt_f -= slight_offset*dir_r
    d = sqrt((pt_f - pt_d) @ (pt_f - pt_d))
    n = int(d / step_size) + 1 #Sinon n est de type np.float64
    sample_values = []
    sample_positions = []
    # creation des valeurs de chaque pixel
    for i in range(n):
        sample_pos = pt_f - dir_r*i*step_size
        sample_val = interpolate(sample_pos)
        sample_values.append(sample_val)
        sample_positions.append(sample_pos)
    # transformation de la couleur pour chaque pixel
    tab_des_alpha = []
    tab_des_couleurs = []
    for i in range(n):
        base_colour = color_function(sample_values[i])
        sample_pos = sample_positions[i]
        N = normal_to_isosurface(sample_pos)
        L = normalize(lum_pos_bb_coord - sample_pos)
        V = - dir_r
        #Si le vecteur normal est nul, on met T = 0 pour faire tomber la composante speculaire.
        T = normalize(vector_symmetric(L, N)) if N @ N != 0 else np.array([0, 0, 0])
        shaded_colour = base_colour[:3] * (I_amb + I_diff * max(0, N @ L)) + I_spec * max(0, V @ T)**shininess
        #Il se peut que la couleur avec ombre ne soit plus dans [0,1]**3, donc on la clip.
        shaded_colour.clip(0, 1, out=shaded_colour)
        #shaded_colour = base_colour[:3]
        tab_des_couleurs.append(shaded_colour)
        tab_des_alpha.append(base_colour[3])
    # Computation de la couleur finale
    color = sky_colour[:3]
    alpha = sky_colour[3]
    for i in range(0,n):
        color = tab_des_alpha[i]*tab_des_couleurs[i]+(1-tab_des_alpha[i])*alpha*color
        alpha = tab_des_alpha[i]+(1-tab_des_alpha[i])*alpha

    return np.array([color[0], color[1], color[2], alpha])


def compute_image(debug=False):
    #iter_pixels = [(79, 79)]
    iter_pixels = [(i, j) for i in range(image_resolution[0]) for j in range(image_resolution[1])]
    #iter_pixels = [(60, 60)]
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
            #Le point par lequel on y rentre est celui le plus proche de l'oeil.
            #Celui par lequel on sort est le plus éloigné.
            entry_point = min(intersections, key = lambda point : np.linalg.norm(point - eye))
            exit_point  = max(intersections, key = lambda point : np.linalg.norm(point - eye))
            #cf Emma
            if debug:
                image[pixel] = np.array([0.0, 1.0, 0.0, 1.0])
            else:
                colour = compute_color(normalize(bb_tangents @ ray_dir), local_bb_coord(entry_point), local_bb_coord(exit_point))
                image[pixel] = colour
            #image[pixel] = np.array([0.0, 1.0, 0.0, 1.0])
        
    return image

'''if __name__ == '__main__':
    image = compute_image()
    show_image(image)'''

def main():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        image = compute_image()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    #stats.print_stats()
    stats.dump_stats(filename='../profiling/test.prof')
    show_image(image)


if __name__ == '__main__':
    main()