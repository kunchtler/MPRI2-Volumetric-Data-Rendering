import numpy as np
import matplotlib.pyplot as plt

resolution = (100, 100)

class NullVector(Exception):
    pass

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise NullVector
    return vector / norm

def find_convex_hull(points):
    """Retourne l'enveloppe convexe 2D d'une ensemble de points sous la forme d'un np array.
    Ces points sont listés dans un ordre trigonométrique.
    Implémente l'algorithme naif de papier cadeau."""
    #De toute façon, on a toujours 8 points donc assez rapide.
    #Réécrire sans numpy ?
    
    if len(points) <= 1:
        return points
    
    #1. Trouver le point le plus à gauche.
    points_dict = {idx : point for idx, point in enumerate(points)}
    first_idx, first_hull_point = min(points_dict.items(), key = lambda x : x[1][0])
    #2 Emballer le nuage de points
    del points_dict[first_idx]
    idx, hull_point = min(points_dict.items(), key = lambda x : np.array([0, 1]) @ (x[1] - first_hull_point))
    points_dict[first_idx] = first_hull_point
    convex_hull = [first_hull_point]
    while idx != first_idx:
        convex_hull.append(hull_point)
        del points_dict[idx]
        idx, hull_point= min(points_dict.items(), key = lambda x :  (convex_hull[-2] - convex_hull[-1]) @ (x[1] - hull_point))
        points_dict[idx] = hull_point

    return np.array(convex_hull)
    

class Camera():

    def __init__(self):
        #Convention pour les coordonnées de la caméra :
        #L'axe x est horizontal, croissant de gauche à droite.
        #L'axe y est vertical, croissant de HAUT EN BAS.
        #Résolution = largeur * hauteur
        self.resolution = np.array([25, 25])
        self.screen_center = np.array([1, 0, 0])
        self.screen_size = np.array([1, 1])
        self.pixel_size = self.screen_size / self.resolution
        self.look_at = np.array([0, 0, 0])
        self.normal = normalize(self.look_at - self.screen_center)
        #self.eye = self.screen_center - self.normal * 2
        self.eye = np.array([3.0, 0.0, 0.5])
        """self.tangentx = normalize(np.array([0, 0, 1]) - self.screen_center)
        self.tangenty = normalize(np.cross(self.normal, self.tangentx))"""
        self.tangentx = np.array([0, 1, 0])
        self.tangenty = np.array([0, 0, 1])
        #[coordonnées relatives à l'écran : (0, 0), (largeur, 0), (0, hauteur), (largeur, hauteur)]
        #tq (largeur/2, hauteur/2) soit (en position globale) en self.screen_center
        self.global_texture_origin = self.screen_center - self.tangentx * self.screen_size[0] / 2 - self.tangenty * self.screen_size[1]/2
        self.screen_corners = [self.global_texture_origin,
            self.global_texture_origin + self.tangentx * self.screen_size[0],
            self.global_texture_origin + self.tangenty * self.screen_size[1],
            self.global_texture_origin + self.tangentx * self.screen_size[0] + self.tangenty * self.screen_size[1]]
        
        #Au besoin, on peut changer le dtype pour coder sur plus de bits
        self.data = np.zeros((self.resolution[0], self.resolution[1], 4), dtype=np.float32)
    
    def get_pixel_coord(self, pixel):
        '''Prend les coordonnées locales pixel = np.array(i, j) d'un pixel de l'écran.
        (0 <= i < self.resolution[0], 0 <= j < self.resolution[1]).
        Retourne les coordonnées globales du centre du pixel.'''
        tmp = self.pixel_size * (pixel + np.array([1/2, 1/2]))
        return self.global_texture_origin + self.tangentx*tmp[0] + self.tangenty*tmp[1]
    
    def get_ray_direction(self, pixel):
        '''Prend les coordonées locales d'un pixel de l'écran.
        Retourne laa direction du ray émis par ce pixel.'''
        return normalize(self.get_pixel_coord(pixel) - self.eye)

    def ray_marching(self, origin, direction):
        step_size = 0.1
        nb_iter = 100
        length_traversed = step_size*nb_iter
        ray_position = origin
        for k in range(nb_iter):
            #if -1 <= ray_position[0] <= 0 and -0.5 <= ray_position[1] <= 0.5 and -0.5 <= ray_position[2] <= 0.5:
            if np.linalg.norm(ray_position - np.array([0.5, 0, 0])) <= 0.5:
                return (0.0, 1.0, 0.0, 1.0)
            ray_position += step_size*direction
        return (1.0, 0.0, 0.0, 1.0)


    def pixels(self):
        for i in range(0, self.resolution[0]):
            for j in range(0, self.resolution[1]):
                yield (i, j)


    def noname(self, bounding_box, debug=False):
        #1. Find Projection of the vertices on the screen
        const = normalize(self.screen_center - self.eye) @ self.normal
        vertices = np.array(bounding_box.vertices)
        direction = np.array([normalize(v - self.eye) for v in vertices])
        projection = self.eye + (const / (direction @ self.normal) * direction.T).T
        #We convert those global positions to local coordinates on the screen.
        #For that, we take the pixel sized vectors of the basis of the plane
        screen_basex = self.tangentx * self.pixel_size[0]
        screen_basey = self.tangenty * self.pixel_size[1]
        on_screen_projection = np.array([projection @ screen_basex, projection @ screen_basey]).T
        #2 Find convex hull of those points in trigonometric order
        convex_hull = find_convex_hull(on_screen_projection)
        #3 Find all pixels left of all edges of the convex hull.
        #Those are exactly the points inside the hull.
        #hull_edges[i] contains the edge from the hull starting at convex_hull[i]
        hull_edges = convex_hull[[(i+1) % len(convex_hull) for i in range(len(convex_hull))], :] - convex_hull
        #The normal to an edge is 2D is given by swapping/multiplying by -1 its components like so:
        normal_to_hull_edges = np.column_stack((-hull_edges[:, 1], hull_edges[:, 0]))
        #Generate all pixels coordinates
        screen_pixels = np.array([[[i, j] for j in range(self.screen_size[1])]for i in range(self.screen_size[0])])
        #in_hull[i, j] tells whether pixel[i, j] is in the convex hull or not.
        in_hull = np.full(screen_pixels.shape[:-1], True)
        for i in range(len(convex_hull)):
            test = (screen_pixels - convex_hull[i]) @ normal_to_hull_edges[i]
            in_hull[test < 0] = False
        if debug: #Shows pixels in hull.  
            plt.imshow(in_hull.T, origin='lower')
            plt.show()


    def compute_image(self):
        for pixel in self.pixels():
            colour = self.ray_marching(self.get_pixel_coord(pixel), self.get_ray_direction(pixel))
            #colour = [1.0, 0.0, 0.0, 1.0]
            self.data[pixel] = colour
        self.data[1, 0] = [1.0, 1.0, 0.0, 1.0]
        #imshow considère que l'axe x est vertical et l'axe y horizontal.
        #Il faut donc transposer l'écran, sans toucher aux vecteur des couleurs
        #De plus, il place l'origine en haut à gauche, alors que le notre est bas à gauche.
        #D'ou l'argument origin='lower'.
        self.data = np.transpose(self.data, axes=(1, 0, 2))
        plt.imshow(self.data, origin='lower')
        plt.show()

camera = Camera()
camera.compute_image()

class VolumetricData():

    def __init__(self):
        self.grid_size = (10, 10, 10)
        self.data = np.zeros(self.grid_size, dtype= np.int8)
        self.center = np.array([-1, 0, 0])
        self.dir = (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    
    def in_bounding_box(self):
        pass
        
#Changer coordonnées screen
#Penser à dessiner les axes en 3D pour aider à mieux comprendre la position de la caméra.
#Implémenter la bounding box pour ne plus avoir besoin d'un nombre d'itérations pour le ray marching.