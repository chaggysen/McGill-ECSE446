# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
from gpytoolbox import read_mesh, per_vertex_normals, per_face_normals # just used to load a mesh, now
##########################################


def normalize(v):
    """
    Returns the normalized vector given vector v.
    Note - This function is only for normalizing 1D vectors instead of batched 2D vectors.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# ray bundles
class Rays(object):

    def __init__(self, Os, Ds):
        """
        Initializes a bundle of rays containing the rays'
        origins and directions. Explicitly handle broadcasting
        for ray origins and directions; they must have the same 
        size for gpytoolbox
        """
        if Os.shape[0] != Ds.shape[0]:
            if Ds.shape[0] == 1:
                self.Os = np.copy(Os)
                self.Ds = np.copy(Os)
                self.Ds[:, :] = Ds[:, :]
            if Os.shape[0] == 1:
                self.Ds = np.copy(Ds)
                self.Os = np.copy(Ds)
                self.Os[:, :] = Os[:, :]
        else:
            self.Os = np.copy(Os)
            self.Ds = np.copy(Ds)

    def __call__(self, t):
        """
        Computes an array of 3D locations given the distances
        to the points.
        """
        return self.Os + self.Ds * t[:, np.newaxis]

    def __str__(self):
        return "Os: " + str(self.Os) + "\n" + "Ds: " + str(self.Ds) + "\n"

    def distance(self, point):
        """
        Compute the distances from the ray origins to a point
        """
        return np.linalg.norm(point[np.newaxis, :] - self.Os, axis=1)


class Geometry(object):
    def __init__(self):
        return

    def intersect(self, rays):
        return


# ===================== our replacement for gpytoolbox routines =====================
def get_bary_coords(intersection, tri):
    denom = area(tri[:, 0], tri[:, 1], tri[:, 2])
    alpha_numerator = area(intersection, tri[:, 1], tri[:, 2])
    beta_numerator = area(intersection, tri[:, 0], tri[:, 2])
    alpha = alpha_numerator / denom
    beta = beta_numerator / denom
    gamma = 1 - alpha - beta
    barys = np.vstack((alpha, beta, gamma)).transpose()
    barys = np.where(np.isnan(barys), 0, barys)
    return barys

def area(t0, t1, t2):
    n = np.cross(t1 - t0, t2 - t0, axis = 1)
    return np.linalg.norm(n, axis = 1) / 2

def ray_mesh_intersect(origin, dir, tri):
    intersection = np.ones_like(dir) * -1
    intersection[:, 2] = np.Inf
    dir = dir[:, None]
    
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0] # (num_triangles, 3)
    s = origin[:, None] - tri[:, 0][None]
    s1 = np.cross(dir, e2)
    s2 = np.cross(s, e1)
    s1_dot_e1 = np.sum(s1 * e1, axis=2)
    results = np.ones((dir.shape[0], tri.shape[0])) * np.Inf

    if (s1_dot_e1 != 0).sum() > 0:
        coefficient = np.reciprocal(s1_dot_e1)
        alpha = coefficient * np.sum(s1 * s, axis=2)
        beta = coefficient * np.sum(s2 * dir, axis=2)
        cond_bool = np.logical_and(
                        np.logical_and(
                            np.logical_and(0 <= alpha,  alpha < 1),
                            np.logical_and(0 <= beta,  beta < 1)
                        ),
                    np.logical_and(0 <= alpha + beta,  alpha + beta < 1)
            ) # (num_rays, num_tri)
        e1_expanded = np.tile(e1[None], (dir.shape[0], 1, 1)) # (num_rays, num_tri, 3)
        dot_temp = np.sum(s1[cond_bool] * e1_expanded[cond_bool], axis = 1) # (num_rays,)
        results_cond1 = results[cond_bool]
        cond_bool2 = dot_temp != 0 

        if cond_bool2.sum() > 0:
              coefficient2 = np.reciprocal(dot_temp)
              e2_expanded = np.tile(e2[None], (dir.shape[0], 1, 1)) # (num_rays, num_tri, 3)
              t = coefficient2 * np.sum( s2[cond_bool][cond_bool2] *
                                         e2_expanded[cond_bool][cond_bool2],
                                         axis = 1)
              results_cond1[cond_bool2] = t
        results[cond_bool] = results_cond1
    results[results <= 0] = np.Inf
    hit_id = np.argmin(results, axis=1)
    min_val = np.min(results, axis=1)
    hit_id[min_val == np.Inf] = -1
    return min_val, hit_id
# ===================== our replacement for gpytoolbox routines =====================

class Mesh(Geometry):
    def __init__(self, filename, brdf_params = np.array([0,0,0,1]), Le = np.array([0,0,0])):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        self.Le = Le
        ### BEGIN CODE
        # [TODO] replace the next line with your code for Phong normal interpolation
        self.face_normals = per_face_normals(self.v, self.f, unit_norm = True)
        ### END CODE
        super().__init__()

    def intersect(self, rays):
        hit_normals = np.array([np.inf, np.inf, np.inf])
        
        hit_distances, triangle_hit_ids = ray_mesh_intersect(rays.Os, rays.Ds, self.v[self.f])
        intersections = rays.Os + hit_distances[:, None] * rays.Ds
        tris = self.v[self.f[triangle_hit_ids]]
        barys = get_bary_coords(intersections, tris)

        ## BEGIN CODE
        # [TODO] replace the next line with your code for Phong normal interpolation
        temp_normals = self.face_normals[triangle_hit_ids]
        ### END CODE

        temp_normals = np.where( (triangle_hit_ids == -1)[:, np.newaxis],
                                 hit_normals,
                                 temp_normals )
        hit_normals = temp_normals

        return hit_distances, hit_normals


class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params = np.array([0,0,0,1]), Le = np.array([0,0,0])):
        """
        Initializes a sphere object with its radius, position and albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        self.Le = Le
        super().__init__()

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays: output the
        intersection distances (set to np.inf if none), and unit hit
        normals (set to [np.inf, np.inf, np.inf] if none.)
        """
        distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        distances[:] = np.inf
        normals = np.zeros(rays.Os.shape, dtype=np.float64)
        normals[:,:] = np.array([np.inf, np.inf, np.inf])

        ### BEGIN CODE
        ### END CODE

        return distances, normals


# Enumerate the different importance sampling strategies we will implement
UNIFORM_SAMPLING, LIGHT_SAMPLING, BRDF_SAMPLING, MIS_SAMPLING = range(4)


class Scene(object):
    def __init__(self, w, h):
        """ Initialize the scene. """
        self.w = w
        self.h = h

        # Camera parameters. Set using set_camera_parameters()
        self.eye = np.empty((3,), dtype=np.float64)
        self.at = np.empty((3,), dtype=np.float64)
        self.up = np.empty((3,), dtype=np.float64)
        self.fov = np.inf

        # Scene objects. Set using add_geometries()
        self.geometries = []

        # Light sources. Set using add_lights()
        self.lights = []

    def set_camera_parameters(self, eye, at, up, fov):
        """ Sets the camera parameters in the scene. """
        self.eye = np.copy(eye)
        self.at = np.copy(at)
        self.up = np.copy(up)
        self.fov = np.float64(fov)

    def add_geometries(self, geometries):
        """ 
        Adds a list of geometries to the scene.
        
        For geometries with non-zero emission,
        additionally add them to the light list.
        """
        for i in range(len(geometries)):
            if (geometries[i].Le != np.array([0, 0, 0])).any():
                self.add_lights([ geometries[i] ])

        self.geometries.extend(geometries)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self, jitter = False):
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """

        ### BEGIN CODE: feel free to remove any/all of our placeholder code, below
        origins = np.zeros((self.w * self.h,3), dtype=np.float64)
        directions = np.zeros((self.w * self.h,3), dtype=np.float64)
        vectorized_eye_rays = Rays(origins, directions)

        return vectorized_eye_rays
        ### END CODE

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        hit_distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        hit_distances[:] = np.inf
        hit_normals = np.zeros(rays.Os.shape, dtype=np.float64)
        hit_normals[:,:] = np.array([np.inf, np.inf, np.inf])
        hit_ids = np.zeros((rays.Os.shape[0],), dtype=np.int)
        hit_ids[:] = -1

        ### BEGIN CODE
        ### END CODE

        return hit_distances, hit_normals, hit_ids

    def render(self, eye_rays, sampling_type=UNIFORM_SAMPLING):
        # vectorized scene intersection
        shadow_ray_o_offset = 1e-8
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),np.array([0,0,0,1])[np.newaxis,:]))[ids]
        L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),np.array([0,0,0])[np.newaxis,:]))[ids]
        objects = np.concatenate((np.array([obj for obj in self.geometries]),np.array([-1])))
        hit_objects = np.concatenate((np.array([obj for obj in self.geometries]),np.array([-1])))[ids]


        # initialize the output "image" (i.e., vector; still needs to be reshaped)
        L = np.zeros(normals.shape, dtype=np.float64)

        # Directly render light sources
        L = np.where(np.logical_and( L_e != np.array([0, 0, 0]), (ids != -1)[:,np.newaxis] ), L_e, L)

        ### BEGIN SOLUTION
        # PLACEHOLDER: our base code renders out debug normals.
        # [TODO] Replace these next three lines with your 
        # solution for your deliverables
        L = np.abs(normals)
        L = L.reshape((self.h, self.w, 3))
        return L
        ### END SOLUTION

    def progressive_render_display( self, jitter = False, total_spp = 20,
                                    sampling_type = UNIFORM_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the 
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(L)

        ### BEGIN CODE (note: we will not grade your progressive rendering code in A3)
        # [TODO] replace the next five lines with 
        # your progressive rendering display loop
        vectorized_eye_rays = self.generate_eye_rays(jitter)
        L = self.render(vectorized_eye_rays, sampling_type)
        image_data.set_data(L)
        plt.pause(0.0001) # add a tiny delay between rendering passes
        ### END CODE

        plt.savefig(f"render-{total_spp}spp.png")
        plt.show(block=True)


if __name__ == "__main__":
    enabled_tests = [True, True, True]

    # Create test scene and test sphere
    scene = Scene(w=int(512), h=int(512)) # TODO: debug at lower resolution
    scene.set_camera_parameters(
        eye=np.array([0, 2, 15], dtype=np.float64),
        at=normalize(np.array([0, -2, 2.5], dtype=np.float64)),
        up=np.array([0, 1, 0], dtype=np.float64),
        fov=int(40)
    )

    # Veach Scene Lights
    scene.add_geometries([ Sphere( 0.0333, np.array([3.75, 0, 0]),
                                   Le = 10 * np.array([901.803, 0, 0]) ),
                           Sphere( 0.1, np.array([1.25, 0, 0]),
                                   Le = 10 * np.array([0, 100, 0]) ),
                           Sphere( 0.3, np.array([-1.25, 0, 0]),
                                   Le = 10 * np.array([0, 0, 11.1111]) ),
                           Sphere( 0.9, np.array([-3.75, 0, 0]),
                                   Le = 10 * np.array([1.23457, 1.23457, 1.23457]) ),
                           Sphere( 0.5, np.array([-10, 10, 4]),
                                   Le = np.array([800, 800, 800]) ) ] ) 
                           
    # Geometry
    scene.add_geometries( [ Mesh( "plate1.obj", 
                                   brdf_params = np.array( [1,1,1,30000] ) ),
                            Mesh( "plate2.obj", 
                                   brdf_params = np.array( [1,1,1,5000] ) ),
                            Mesh( "plate3.obj", 
                                   brdf_params = np.array( [1,1,1,1500] ) ),
                            Mesh( "plate4.obj", 
                                   brdf_params = np.array( [1,1,1,100] ) ),
                            Mesh( "floor.obj", 
                                   brdf_params = np.array( [0.5,0.5,0.5,1] ) ) ])

    #########################################################################
    ### Deliverable 1 TEST: comment/modify as you see fit
    #########################################################################
    if enabled_tests[0]:
        scene.progressive_render_display(total_spp = 1, jitter = True, sampling_type = UNIFORM_SAMPLING)
        scene.progressive_render_display(total_spp = 10, jitter = True, sampling_type = UNIFORM_SAMPLING)
        scene.progressive_render_display(total_spp = 100, jitter = True, sampling_type = UNIFORM_SAMPLING)

    #########################################################################
    ### Deliverable 2 TEST: comment/modify as you see fit
    #########################################################################
    if enabled_tests[1]:
        scene.progressive_render_display(total_spp = 1, jitter = True, sampling_type = LIGHT_SAMPLING)
        scene.progressive_render_display(total_spp = 10, jitter = True, sampling_type = LIGHT_SAMPLING)
        scene.progressive_render_display(total_spp = 100, jitter = True, sampling_type = LIGHT_SAMPLING)

    #########################################################################
    ### Deliverable 3 TEST (Only for ECSE 546 students!): comment/modify as you see fit
    #########################################################################
    if enabled_tests[2]:  
        scene.progressive_render_display(total_spp = 1, jitter = True, sampling_type = MIS_SAMPLING)
        scene.progressive_render_display(total_spp = 10, jitter = True, sampling_type = MIS_SAMPLING)
        scene.progressive_render_display(total_spp = 100, jitter = True, sampling_type = MIS_SAMPLING)
        