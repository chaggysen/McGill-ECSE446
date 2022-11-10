# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
# for ray-mesh intersection queries
from gpytoolbox import ray_mesh_intersect, read_mesh, per_face_normals, per_vertex_normals
##########################################
import math
import random


def normalize(v):
    """
    Returns the normalized vector given vector v.
    Note - This function is only for normalizing 1D vectors instead of batched 2D vectors.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def normalize2D(vs):
    """
    Returns the normalized vector given vector v.
    Note - This function is only for normalizing 1D vectors instead of batched 2D vectors.
    """
    norm = np.linalg.norm(vs, axis=1, keepdims=True)
    return vs/norm


# ray bundles
class Rays(object):

    def __init__(self, Os, Ds):
        """
        Initializes a bundle of rays containing the rays'
        origins and directions. Explicitly handle broadcasting
        for ray origins and directions; gpytoolbox expects the
        number of ray origins to be equal to the number of ray
        directions (our code handles the cases where rays either
        all share the same origin or all share the same direction.)
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
        Computes an array of 3D locations given parametric
        distances along the rays
        """
        return self.Os + self.Ds * t[:, np.newaxis]

    def __str__(self):
        return "Os: " + str(self.Os) + "\n" + "Ds: " + str(self.Ds) + "\n"

    def distance(self, point):
        """
        Compute the distances from the ray origins to a point
        """
        return np.linalg.norm(point[np.newaxis, :] - self.Os, axis=1)


# abstraction for every scene object
class Geometry(object):
    def __init__(self):
        return

    def intersect(self, rays):
        return

# sphere objects for our scene


class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params):
        """
        Initializes a sphere object with its radius, position and diffuse albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.brdf_params = brdf_params
        super().__init__()

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays: output the
        intersection distances (set to np.inf if none), and unit hit
        normals (set to [np.inf, np.inf, np.inf] if none.)
        """

        # initialize hit_distances and hit_normals
        hit_distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        hit_distances[:] = np.inf
        hit_normals = np.zeros(rays.Os.shape, dtype=np.float64)
        hit_normals[:, :] = np.array([np.inf, np.inf, np.inf])

        # BEGIN SOLUTION
        sphere_center, sphere_radius = self.c, self.r
        ray_directions, ray_origins = rays.Ds, rays.Os

        # calculate As, Bs, Cs and discrimiments
        As = np.sum(ray_directions*ray_directions, axis=1)
        Bs = 2.0 * np.sum(ray_directions*(ray_origins - sphere_center), axis=1)
        Cs = np.sum((ray_origins - sphere_center)*(ray_origins -
                                                   sphere_center), axis=1) - (sphere_radius**2)
        discriminents = Bs**2 - (4*As*Cs)

        # UPDATE DISTANCES
        # check discriminent == 0
        distances = (-Bs/(2*As))
        hit_distances = np.where(discriminents == 0, distances, hit_distances)
        # check discriminent > 0
        t1s = (-Bs + np.sqrt(discriminents)) / (2*As)
        t2s = (-Bs - np.sqrt(discriminents)) / (2*As)
        min_distances = np.minimum(t1s, t2s)
        hit_distances = np.where(
            discriminents > 0, min_distances, hit_distances)
        # check distance > epsilon_sphere
        hit_distances = np.where(
            hit_distances > self.EPSILON_SPHERE, hit_distances, np.inf)

        # UPDATE NORMALS
        hit_distances_transpose = np.array([hit_distances]).T
        point_hits = ray_origins + ray_directions*hit_distances_transpose
        hit_normals[hit_distances != np.inf] = normalize2D(
            point_hits[hit_distances != np.inf] - sphere_center)

        return hit_distances, hit_normals

        # END SOLUTION


# triangle mesh objects for our scene
class Mesh(Geometry):
    def __init__(self, filename, brdf_params):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        # BEGIN SOLUTION
        self.face_normals = per_face_normals(self.v, self.f, unit_norm=True)
        self.per_vectex_normals = per_vertex_normals(
            self.v, self.f)
        # END SOLUTION
        super().__init__()

    def intersect(self, rays):

        # initialize hit_distances and hit_normals
        hit_normals = np.tile(
            np.array([np.inf, np.inf, np.inf]), (len(rays.Os), 1))
        hit_distances, triangle_hit_ids, barys = ray_mesh_intersect(
            rays.Os, rays.Ds, self.v, self.f, use_embree=True)

        # find indexes to update
        indexes_to_update = np.array(np.where(triangle_hit_ids != -1)[0])

        # calculate aphas, betas and gammas
        alphas = np.array([barys[indexes_to_update][:, 0]]).T
        betas = np.array([barys[indexes_to_update][:, 1]]).T
        gammas = np.array([barys[indexes_to_update][:, 2]]).T

        # update hit_normals on indexes_to_update
        hit_normals[indexes_to_update] = alphas*self.per_vectex_normals[self.f[triangle_hit_ids[indexes_to_update]][:, 0]] + betas * \
            self.per_vectex_normals[self.f[triangle_hit_ids[indexes_to_update]][:, 1]] + \
            gammas * \
            self.per_vectex_normals[self.f[triangle_hit_ids[indexes_to_update]][:, 2]]

        return hit_distances, hit_normals


# Enumerate the different importance sampling strategies we will implement
UNIFORM_SAMPLING, COSINE_SAMPLING = range(2)


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
        """ Adds a list of geometries to the scene. """
        self.geometries.extend(geometries)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self, jitter=False):
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """

        image_ratio = self.w / self.h
        scale = math.tan(math.radians(self.fov * 0.5))
        zc = normalize(self.at - self.eye)
        ray_origin = self.eye
        x_axis = np.cross(self.up, zc)
        y_axis = np.cross(zc, x_axis)
        cameraToWorld = np.array([
            [x_axis[0], y_axis[0], zc[0], self.eye[0]],
            [x_axis[1], y_axis[1], zc[1], self.eye[1]],
            [x_axis[2], y_axis[2], zc[2], self.eye[2]],
            [0, 0, 0, 1]
        ])
        # offset is assigned based on value of jitter
        offset1 = random.uniform(0, 1) if jitter else 0.5
        offset2 = random.uniform(0, 1) if jitter else 0.5
        x = (2 * (np.arange(self.w) + offset1) /
             self.w - 1) * image_ratio * scale
        y = (1 - 2 * (np.arange(self.h) + offset2) / self.h) * scale
        XX, YY = np.meshgrid(x, y)
        XX_flat, YY_flat, size = XX.flatten(), YY.flatten(), XX.size
        ray_direction = np.dot(cameraToWorld, np.array(
            [XX_flat, YY_flat, np.ones(size), np.zeros(size)]))
        ray_direction = ray_direction[:3].T
        ray_direction = list(map(normalize, ray_direction))  # change
        return Rays(np.tile(ray_origin, (len(ray_direction), 1)), np.array(ray_direction))

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """

        # BEGIN SOLUTION

        hit_distances = np.tile(np.inf, len(rays.Os))
        hit_normals = np.tile(
            np.array([np.inf, np.inf, np.inf]), (len(rays.Os), 1))
        hit_ids = np.tile(-1, len(rays.Os))

        for geo_idx in range(len(self.geometries)):
            geometry = self.geometries[geo_idx]
            result = geometry.intersect(rays)
            distances, normals = result[0], result[1]
            hit_normals = np.where(
                (distances < hit_distances)[..., None], normals, hit_normals)
            hit_ids = np.where((distances < hit_distances), geo_idx, hit_ids)
            hit_distances = np.where(
                distances < hit_distances, distances, hit_distances)
        return hit_distances, hit_normals, hit_ids
        # END SOLUTION

    def render(self, eye_rays, sampling_type=UNIFORM_SAMPLING, num_samples=1):
        shadow_ray_o_offset = 1e-6

        # vectorized primary visibility test
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # CAREFUL! when ids == -1 (i.e., no hit), you still get valid BRDF parameters!
        brdf_params = np.array(
            [obj.brdf_params for obj in self.geometries])[ids]

        # initialize the output "image" (i.e., vector; still needs to be reshaped)
        L = np.zeros(normals.shape, dtype=np.float64)

        # Generate random Ws
        sigma1s = np.random.rand(len(eye_rays.Os))
        sigma2s = np.random.rand(len(eye_rays.Os))
        wzs = 2*sigma1s - 1
        rs = np.sqrt(-np.power(wzs, 2) + 1)
        thetas = 2*math.pi*sigma2s
        wxs = rs*np.cos(thetas)
        wys = rs*np.sin(thetas)
        Ws = np.stack((wxs, wys, wzs), axis=1)

        # Calculate Mont-Carlo Estimator
        constants = (2*brdf_params)/num_samples
        Xs = hit_points + shadow_ray_o_offset*normals
        # Create shadow_rays
        shadow_rays = Rays(Xs, Ws)
        hit_distances, hit_normals, hit_ids = self.intersect(shadow_rays)
        # Set visibilities
        visibilities = np.zeros(len(eye_rays.Os))
        visibilities = np.where(hit_distances != np.inf, visibilities, 1)

        # add values inside sum
        sum_values = np.zeros(len(eye_rays.Os))
        for _ in range(num_samples):
            product = np.sum(normals*Ws, axis=1)
            product = np.where(product > 0, product, 0)
            sum_values += visibilities * product

        sum_values = np.array([sum_values]).T

        L = sum_values * constants
        L = L.reshape((self.h, self.w, 3))

        return L
        # END SOLUTION

    def progressive_render_display(self, jitter=False, total_spp=20, spppp=1,
                                   sampling_type=UNIFORM_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(L)

        # number of rendering iterations needed to obtain
        # (at least) our total desired spp
        progressive_iters = int(np.ceil(total_spp / spppp))

        # BEGIN SOLUTION
        # your progressive rendering display loop

        for i in range(progressive_iters):
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            L -= L/(i+1)
            L += self.render(vectorized_eye_rays,
                             sampling_type, spppp)/(i+1)
            image_data.set_data(L)
            plt.pause(0.001)
            plt.title(f"current spp: {spppp * i} of {1 * progressive_iters}")

        # END SOLUTION

        plt.savefig(f"render-{progressive_iters * spppp}spp.png")
        plt.show(block=True)


if __name__ == "__main__":
    enabled_tests = [True, True, True, False]
    open("./bunny-446.obj")

    #########################################################################
    # Deliverable 1 TESTS Eye Ray Anti Aliasing and Progressive Rendering
    #########################################################################
    if enabled_tests[0]:
        # Create test scene and test sphere
        # DEBUG: use a lower resolution to debug
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))
        scene.set_camera_parameters(
            eye=np.array([2, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(1000, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9 / np.pi, 0.9 / np.pi, 0.9 / np.pi]))
        bunny_sphere = Sphere(1.5, np.array([0.5, -0.5, 0]),
                              brdf_params=np.array([0.9 / np.pi, 0.9 / np.pi, 0.9 / np.pi]))
        scene.add_geometries([bunny_sphere])
        scene.add_geometries([sphere])

        # no-AA
        scene.progressive_render_display(jitter=False)

        # with AA
        scene.progressive_render_display(jitter=True)

    #########################################################################
    # Deliverable 2 TESTS Mesh Intersection and Phong Normal Interpolation
    #########################################################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        # DEBUG: use a lower resolution to debug
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))
        scene.set_camera_parameters(
            eye=np.array([2, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(1000, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9, 0.9, 0.9]))
        bunny = Mesh("bunny-446.obj",
                     brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([sphere])
        scene.add_geometries([bunny])

        # render a 1spp, no AA jittering image
        scene.progressive_render_display(jitter=False, total_spp=1, spppp=1)

    ###########################################################################
    # Deliverable 3 TESTS Ambient Occlusion with Uniform Importance Sampling
    ###########################################################################
    if enabled_tests[2]:
        # Create test scene and test sphere
        # DEBUG: use a lower resolution to debug
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))
        scene.set_camera_parameters(
            eye=np.array([2, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(1000, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([sphere])

        # DEBUG: start with a simpler scene before replacing your spherical bunny with an actual bunny
        # bunny_sphere = Sphere(1.5, np.array([0.5, -0.5, 0]),
        #                       brdf_params=np.array([0.9, 0.9, 0.9]))
        # scene.add_geometries([bunny_sphere])
        bunny = Mesh("bunny-446.obj",
                     brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([bunny])

        scene.add_lights([
            {
                "type": "uniform",
                "color": np.array([0.9, 0.9, 0.9])
            }
        ])

        scene.progressive_render_display(jitter=True, total_spp=100, spppp=1,
                                         sampling_type=UNIFORM_SAMPLING)

    #########################################################################################
    # ECSE 546 Only: Deliverable 4 TESTS Ambient Occlusion with Cosine Importance Sampling
    #########################################################################################
    if enabled_tests[3]:
        # Create test scene and test sphere
        # DEBUG: use a lower resolution to debug
        scene = Scene(w=int(1024 / 4), h=int(768 / 4))
        scene.set_camera_parameters(
            eye=np.array([2, 0.5, -5], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(10, np.array([0, -1002.5, 0]),
                        brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([sphere])

        # DEBUG: start with a simpler scene before replacing your spherical bunny with an actual bunny
        bunny_sphere = Sphere(1.5, np.array([0.5, -0.5, 0]),
                              brdf_params=np.array([0.9, 0.9, 0.9]))
        scene.add_geometries([bunny_sphere])
        # bunny = Mesh("bunny-446.obj",
        #              brdf_params=np.array([0.9, 0.9, 0.9]))
        # scene.add_geometries([bunny])

        scene.add_lights([
            {
                "type": "uniform",
                "color": np.array([0.9, 0.9, 0.9])
            }
        ])

        scene.progressive_render_display(jitter=True, total_spp=100, spppp=1,
                                         sampling_type=COSINE_SAMPLING)
