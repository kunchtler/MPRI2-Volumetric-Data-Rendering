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
import dataset_generation as gen
data = gen.sample_data1(np.full((3,), 10), 255)
#data = np.full((10, 10, 10), 255)

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

slight_offset = 0.00000001

#Other variables
sky_colour = np.array([0.0, 0.0, 0.0, 1.0])
step_size = np.min(voxel_size) / 3
I_amb = np.ones((3,))
I_diff = np.ones((3,))
I_spec = np.ones((3,))
lum_pos = np.array([0, 10, 5])
shininess = 15
color_function = lambda x : np.array([x, x, x, 1.0])

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
    (a,b,c)=pt
    x=int(a)
    y=int(b)
    z=int(c)
    #(x, y, z) est dans le voxel de coins (int(x), int(y), int(z)) et (int(x)+1, int(y)+1, int(z)+1).
    #Il y a un problème quand x = int(x) = data_resolution[0], ce qui est possible !
    #Dans ce cas, on interpole sur la face / l'arête / le sommet
    c000=data[x,y,z]
    c100=data[x+1,y,z]
    c001=data[x,y,z+1]
    c101=data[x+1,y,z+1]
    c010=data[x,y+1,z]
    c110=data[x+1,y+1,z]
    c011=data[x,y+1,z+1]
    c111=data[x+1,y+1,z+1]
    L=[c000,c100,c010,c110,c001,c101,c011,c111]
    L1=np.array([1,x,y,z,(x)*(y),(x)*(z),(y)*(z),(x)*(y)*(z)])
    L2=np.array([1,x+1,y,z,(x+1)*(y),(x+1)*(z),(y)*(z),(x+1)*(y)*(z)])
    L3=np.array([1,x,y+1,z,(x)*(y+1),(x)*(z),(y+1)*(z),(x)*(y+1)*(z)])
    L4=np.array([1,x+1,y+1,z,(x+1)*(y+1),(x+1)*(z),(y+1)*(z),(x+1)*(y+1)*(z)])
    L5=np.array([1,x,y,z+1,(x)*(y),(x)*(z+1),(y)*(z+1),(x)*(y)*(z+1)])
    L6=np.array([1,x+1,y,z+1,(x+1)*(y),(x+1)*(z+1),(y)*(z+1),(x+1)*(y)*(z+1)])
    L7=np.array([1,x,y+1,z+1,(x)*(y+1),(x)*(z+1),(y+1)*(z+1),(x)*(y+1)*(z+1)])
    L8=np.array([1,x+1,y+1,z+1,(x+1)*(y+1),(x+1)*(z+1),(y+1)*(z+1),(x+1)*(y+1)*(z+1)])
    M=np.array([L1,L2,L3,L4,L5,L6,L7,L8])
    X=np.linalg.solve(M,L)
    return X[0]+X[1]*a+X[2]*b+X[3]*c+X[4]*a*b+X[5]*a*c+X[6]*b*c+X[7]*a*b*c

def normal_to_isosurface(u):
    (x,y,z) = u
    ux_m1 = (max(0, x-1), y, z)
    uy_m1 = (x, max(0, y-1), z)
    uz_m1 = (x, y, max(0, z-1))
    ux_p1 = (min(nb_voxels[0], x+1), y, z)
    uy_p1 = (x, min(nb_voxels[1], y+1), z)
    uz_p1 = (x, y, min(nb_voxels[2], z+1))
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

def vector_symmetric(u, v):
    '''Symmétrique du vecteur u par rapport au vecteur v'''
    return 2 * (u @ v) / (v @ v) * v - u

def local_bb_coord(point):
    return bb_voxel_sized_basis @ (point - bb_origin)

def compute_color(dir_r, pt_d, pt_f):
    d = sqrt((pt_f - pt_d) @ (pt_f - pt_d))
    n = int(d / step_size) #Sinon n est de type np.float64
    sample_values = []
    sample_positions = []
    # creation des valeurs de chaque pixel
    for i in range(n):
        sample_pos = local_bb_coord(pt_f - dir_r*i*step_size) #Le faire dans l'autre sens ?
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
        L = normalize(lum_pos - sample_pos)
        V = -dir_r
        #Si le vecteur normal est nul, on met T = 0 pour faire tomber la composante speculaire.
        T = normalize(vector_symmetric(L, N)) if N @ N != 0 else np.array([0, 0, 0])
        shaded_colour = base_colour[:3] @ (I_amb + I_diff * max(0, N @ L)) + I_spec * max(0, V @ T)**shininess
        tab_des_couleurs.append(shaded_colour)
        tab_des_alpha.append(base_colour[3])
    # Computation de la couleur finale
    color = sky_colour[:3]
    alpha = sky_colour[3]
    for i in range(0,n):
        color = tab_des_alpha[i]*tab_des_couleurs[i]+(1-tab_des_alpha[i])*alpha*color
        alpha = tab_des_alpha[i]+(1-tab_des_alpha[i])*alpha

    return np.array([color[0], color[1], color[2], alpha])


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
            #Le point par lequel on y rentre est celui le plus proche de l'oeil.
            #Celui par lequel on sort est le plus éloigné.
            entry_point = min(intersections, key = lambda point : np.linalg.norm(point - eye))
            exit_point  = max(intersections, key = lambda point : np.linalg.norm(point - eye))
            #cf Emma
            colour = compute_color(ray_dir, entry_point, exit_point)
            image[pixel] = colour
            #image[pixel] = np.array([0.0, 1.0, 0.0, 1.0])
        
    return image
    

def show_image(image):
    #imshow considère que l'axe x est vertical et l'axe y horizontal.
    #Il faut donc transposer l'écran, sans toucher aux vecteur des couleurs
    #De plus, il place l'origine en haut à gauche, alors que le notre est bas à gauche.
    #D'ou l'argument origin='lower'.
    plt.imshow(np.transpose(image, axes=(1, 0, 2)), origin='lower')
    plt.show()

if __name__ == '__main__':
    image = compute_image()
    show_image()

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