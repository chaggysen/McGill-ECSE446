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
        # distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        # distances[:] = np.inf
        # normals = np.zeros(rays.Os.shape, dtype=np.float64)
        # normals[:, :] = np.array([np.inf, np.inf, np.inf])

        # BEGIN SOLUTION

        def quadratic(ray_direction, ray_origin):
            inf_distance = np.inf
            inf_normal = np.array([np.inf, np.inf, np.inf])

            sphere_center, sphere_radius = self.c, self.r
            a = np.dot(normalize(ray_direction), normalize(ray_direction))
            b = 2.0 * np.dot(ray_direction, (ray_origin - sphere_center))
            c = np.dot((ray_origin - sphere_center),
                       (ray_origin - sphere_center)) - (sphere_radius**2)
            discriminent = b**2 - (4*a*c)
            if discriminent < 0:
                result = (inf_distance, inf_normal)
            elif discriminent == 0:
                if (-b/(2*a)) > self.EPSILON_SPHERE:
                    distance = (-b/(2*a))
                    point_hit = ray_origin + ray_direction*distance
                    normal = normalize(point_hit - self.c)
                    result = (distance, normal)
                else:
                    result = (inf_distance, inf_normal)
            else:
                t1 = (-b + math.sqrt(discriminent)) / (2*a)
                t2 = (-b - math.sqrt(discriminent)) / (2*a)
                if min(t1, t2) > self.EPSILON_SPHERE:
                    distance = min(t1, t2)
                    point_hit = ray_origin + ray_direction*distance
                    normal = normalize(point_hit - self.c)
                    result = (distance, normal)
                else:
                    result = (inf_distance, inf_normal)
            return result

        quadratic_vectorized = np.vectorize(
            quadratic, signature='(d),(d)->(),(d)')

        # same number of direction and origins

        result = quadratic_vectorized(rays.Ds, rays.Os)
        return result

        # END SOLUTION


# triangle mesh objects for our scene
class Mesh(Geometry):
    def __init__(self, filename, brdf_params):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        # BEGIN SOLUTION
        # [TODO] replace the next line with your code for Phong normal interpolation
        self.face_normals = per_face_normals(self.v, self.f, unit_norm=True)
        self.per_vectex_normals = per_vertex_normals(
            self.v, self.f)
        # END SOLUTION
        super().__init__()

    def intersect(self, rays):

        hit_normals = np.tile(
            np.array([np.inf, np.inf, np.inf]), (len(rays.Os), 1))
        hit_distances, triangle_hit_ids, barys = ray_mesh_intersect(
            rays.Os, rays.Ds, self.v, self.f, use_embree=True)
        indexes_to_update = np.array(np.where(triangle_hit_ids != -1)[0])

        if len(rays.Os) == 1:
            hit_distances = np.array([hit_distances])

        if len(hit_normals) > 1:
            for index in indexes_to_update:
                hit_normals[index] = barys[index][0]*self.per_vectex_normals[self.f[triangle_hit_ids[index]][0]] + barys[index][1] * \
                    self.per_vectex_normals[self.f[triangle_hit_ids[index]][1]] + \
                    barys[index][2] * \
                    self.per_vectex_normals[self.f[triangle_hit_ids[index]][2]]

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

        # BEGIN SOLUTION: feel free to remove any/all of our placeholder code, below
        # origins = np.zeros((self.w * self.h, 3), dtype=np.float64)
        # directions = np.zeros((self.w * self.h, 3), dtype=np.float64)
        # vectorized_eye_rays = Rays(origins, directions)

        # return vectorized_eye_rays

        if jitter:
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
            # the 0.5 can be anything between 0 and 1
            x = (2 * (np.arange(self.w) + random.uniform(0, 1)) /
                 self.w - 1) * image_ratio * scale
            y = (1 - 2 * (np.arange(self.h) + random.uniform(0, 1)) / self.h) * scale
            XX, YY = np.meshgrid(x, y)
            XX_flat, YY_flat, size = XX.flatten(), YY.flatten(), XX.size
            ray_direction = np.dot(cameraToWorld, np.array(
                [XX_flat, YY_flat, np.ones(size), np.zeros(size)]))
            ray_direction = ray_direction[:3].T
            ray_direction = list(map(normalize, ray_direction))
            return Rays(np.tile(ray_origin, (len(ray_direction), 1)), np.array(ray_direction))

        else:
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
            x = (2 * (np.arange(self.w) + 0.5) /
                 self.w - 1) * image_ratio * scale
            y = (1 - 2 * (np.arange(self.h) + 0.5) / self.h) * scale
            XX, YY = np.meshgrid(x, y)
            XX_flat, YY_flat, size = XX.flatten(), YY.flatten(), XX.size
            ray_direction = np.dot(cameraToWorld, np.array(
                [XX_flat, YY_flat, np.ones(size), np.zeros(size)]))
            ray_direction = ray_direction[:3].T
            ray_direction = list(map(normalize, ray_direction))
            return Rays(np.tile(ray_origin, (len(ray_direction), 1)), np.array(ray_direction))
        # END SOLUTION

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """

        # BEGIN SOLUTION
        # Instead of self.spheres, use self.geometries. Normals are already calculated.

        hit_distances = np.tile(np.inf, len(rays.Os))
        hit_normals = np.tile(
            np.array([np.inf, np.inf, np.inf]), (len(rays.Os), 1))
        hit_ids = np.tile(-1, len(rays.Os))

        for geo_idx in range(len(self.geometries)):
            geometry = self.geometries[geo_idx]
            result = geometry.intersect(rays)
            distances, normals = result[0], result[1]
            indexes_to_update = np.array(
                np.where(distances < hit_distances)[0])
            hit_distances = np.where(
                distances < hit_distances, distances, hit_distances)
            for index in indexes_to_update:
                hit_normals[index] = normals[index]
                hit_ids[index] = geo_idx
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

        # print(len(brdf_params))

        # initialize the output "image" (i.e., vector; still needs to be reshaped)
        L = np.zeros(normals.shape, dtype=np.float64)

        # BEGIN SOLUTION
        Ws = []
        for _ in range(len(eye_rays.Os)):
            sigma1 = np.random.rand()
            sigma2 = np.random.rand()
            wz = 2*sigma1 - 1
            r = math.sqrt(1 - wz ** 2)
            theta = 2*math.pi*sigma2
            wx = r*math.cos(theta)
            wy = r*math.sin(theta)
            Ws.append([wx, wy, wz])

        for idx in range(len(hit_points)):
            constant = (2*brdf_params[idx])/num_samples
            sum_value = 0
            for i in range(num_samples):
                X = hit_points[idx] + shadow_ray_o_offset*normals[idx]
                wi = Ws[idx]
                shadow_ray = Rays(np.array([X]), np.array([wi]))
                visibility = 1
                for geometry in self.geometries:
                    hit_distances, hit_normals = geometry.intersect(shadow_ray)
                    if hit_distances[0] != np.inf:
                        visibility -= 1
                        break
                sum_value += visibility*max(0, np.dot(normals[idx], Ws[idx]))
            L[idx] = constant * sum_value
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
    enabled_tests = [False, False, True, False]
    open("./bunny-446.obj")

    #########################################################################
    # Deliverable 1 TESTS Eye Ray Anti Aliasing and Progressive Rendering
    #########################################################################
    if enabled_tests[0]:
        # Create test scene and test sphere
        # DEBUG: use a lower resolution to debug
        scene = Scene(w=int(200 / 4), h=int(100 / 4))
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
        scene = Scene(w=int(200 / 4), h=int(100 / 4))
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
        scene = Scene(w=int(200 / 4), h=int(100 / 4))
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
