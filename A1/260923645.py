# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################
import math
import os
from tabnanny import check
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
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


# Ray bundles
class Rays(object):

    def __init__(self, Os, Ds):
        """
        Initializes a bundle of rays containing the rays'
        origins and directions.
        """
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


# A sphere object encapsulating the geometry and its diffuse material properties
class Sphere(object):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, rho=np.zeros((3,), dtype=np.float64)):
        """
        Initializes a sphere object with its radius, position and albedo.
        """
        self.r = np.float64(r)
        self.c = np.copy(c)
        self.rho = np.copy(rho)

    def intersect(self, rays):
        """
        Intersect the sphere with a bundle of rays, and compute the
        distance between the hit point on the sphere surface and the
        ray origins. If a ray did not intersect the sphere, set the
        distance to np.inf.
        """
        # BEGIN SOLUTION

        def quadratic(ray_direction, ray_origin):
            result = np.inf
            sphere_center, sphere_radius = self.c, self.r
            a = np.dot(normalize(ray_direction), normalize(ray_direction))
            b = 2.0 * np.dot(ray_direction, (ray_origin - sphere_center))
            c = np.dot((ray_origin - sphere_center),
                       (ray_origin - sphere_center)) - (sphere_radius**2)
            discriminent = b**2 - (4*a*c)
            if discriminent < 0:
                result = np.inf
            elif discriminent == 0:
                result = (-b/(2*a)) if (-b/(2*a)) > self.EPSILON_SPHERE else np.inf
            else:
                t1 = (-b + math.sqrt(discriminent)) / (2*a)
                t2 = (-b - math.sqrt(discriminent)) / (2*a)
                result = (min(t1, t2)) if (min(t1, t2)) > self.EPSILON_SPHERE else np.inf
            return result

        quadratic_vectorized = np.vectorize(
            quadratic, signature='(d),(d)->()')

        # same number of direction and origins
        if len(rays.Ds) == len(rays.Os):
            result = quadratic_vectorized(rays.Ds, rays.Os)
            return result
        
        # one direction only and multiple origins
        elif len(rays.Ds) == 1:
            populated_directions = np.tile(rays.Ds, (len(rays.Os), 1))
            result = quadratic_vectorized(populated_directions, rays.Os)
            return result
        
        # one origins and multiple direction
        elif len(rays.Os) == 1:
            populated_origins = np.tile(rays.Os, (len(rays.Ds), 1))
            result = quadratic_vectorized(populated_origins, rays.Os)
            return result

        # END SOLUTION


# A class to encapsulate everything about a scene: image resolution, scene objects, light properties
class Scene(object):
    REFRACTIVE_INDEX_OUT = 1.0
    REFRACTIVE_INDEX_IN = 1.5

    def __init__(self, w, h):
        """ Initialize the scene. """
        self.w = w
        self.h = h

        # Camera parameters. Set using set_camera_parameters()
        self.eye = np.empty((3,), dtype=np.float64)
        self.at = np.empty((3,), dtype=np.float64)
        self.up = np.empty((3,), dtype=np.float64)
        self.fov = np.inf

        # Scene objects. Set using add_objects()
        self.spheres = []

        # Light sources. Set using add_lights()
        self.lights = []

    def set_camera_parameters(self, eye, at, up, fov):
        """ Sets the camera parameters in the scene. """
        self.eye = np.copy(eye)
        self.at = np.copy(at)
        self.up = np.copy(up)
        self.fov = np.float64(fov)

    def add_spheres(self, spheres):
        """ Adds a list of objects to the scene. """
        self.spheres.extend(spheres)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def generate_eye_rays(self):
        """
        Generate a bundle of eye rays.

        The eye rays originate from the eye location, and shoots through each
        pixel into the scene.
        """
        # BEGIN SOLUTION
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
        x = (2 * (np.arange(self.w) + 0.5) / self.w - 1) * image_ratio * scale
        y = (1 - 2 * (np.arange(self.h) + 0.5) / self.h) * scale
        XX, YY = np.meshgrid(x, y)
        XX_flat, YY_flat, size= XX.flatten(), YY.flatten(), XX.size
        ray_direction = np.dot(cameraToWorld, np.array([XX_flat,YY_flat, np.ones(size), np.zeros(size)]))
        ray_direction = ray_direction[:3].T
        ray_direction = list(map(normalize, ray_direction))
        return Rays(np.tile(ray_origin, (len(ray_direction),1)), ray_direction)
        # END SOLUTION

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        hit_ids = []
        hit_distances = []
        hit_normals = []
        global min_distance
        min_distance = np.inf
        global idx
        idx = -1
        global normal
        normal = np.array([np.inf, np.inf, np.inf])      

        # BEGIN SOLUTION

        def check_for_sphere_hit(index):
            global min_distance
            global idx
            global normal
            min_distance = np.inf
            idx = -1
            normal = np.array([np.inf, np.inf, np.inf])
            direction, origin = rays.Ds[index], rays.Os[index]
            for sphere_idx in range(len(self.spheres)):
                sphere = self.spheres[sphere_idx]
                distance = sphere.intersect(
                    Rays(np.array([origin]), np.array([direction])))[0]
                if distance < min_distance:
                    min_distance = distance
                    idx = sphere_idx
                    point_hit = origin + direction*min_distance
                    normal = normalize(point_hit - sphere.c)
            return np.array([min_distance, np.array(normal), idx])

        indexes = np.arange(len(rays.Ds))
        result = np.array(list(map(check_for_sphere_hit, indexes))).T
        result = result.tolist()
        hit_distances, hit_normals, hit_ids= np.array(result[0]), np.array(result[1]), np.array(result[2])
        return hit_distances, hit_normals, hit_ids
        # END SOLUTION


# Shade a scene given a bundle of eye rays; outputs a color image suitable for matplotlib visualization
def shade(scene, rays):
    shadow_ray_o_offset = 1e-6
    # BEGIN SOLUTION

    hit_distances, hit_normals, hit_ids = scene.intersect(rays)
    new_Os = []

    def get_hit_point_and_new_origins(index):
        hit_point = rays.Os[index] + rays.Ds[index]*hit_distances[index]
        new_origin = hit_point+(shadow_ray_o_offset*hit_normals[index])
        return np.array(new_origin)

    indexes = np.arange(len(rays.Os))
    new_Os = np.array(list(map(get_hit_point_and_new_origins, indexes)))

    Ls = []

    lights, spheres = scene.lights, scene.spheres
    for light in lights:
        L = np.zeros(shape=(len(hit_distances), 3))
        direction, color = light["direction"], light["color"]
        shadow_rays = Rays(np.array(new_Os), np.tile(
            direction, (len(new_Os), 1)))
        shadow_ids = scene.intersect(shadow_rays)[2]

        def update_L(index):  
            normal, sphere_id, shadow_id = hit_normals[index], hit_ids[index], shadow_ids[index]
            if sphere_id != -1:
                rho = spheres[sphere_id].rho
                L[index] += (rho/math.pi)*color[:3] * \
                        np.maximum(0, np.dot(normal, direction))
            if shadow_id != -1:
                L[index] = 0
            return L

        l_indexes = np.arange(len(hit_normals))
        call = list(map(update_L, l_indexes))
        Ls.append(L)
        
    global new_L
    new_L = np.zeros(shape=(len(hit_distances), 3))
    l_iter = np.arange(len(Ls))
    
    def calculate_new_L(idx):
        global new_L
        new_L += Ls[idx]
    call = list(map(calculate_new_L, l_iter))
    L = new_L.reshape((scene.h, scene.w, 3))
    return L


if __name__ == "__main__":
    enabled_tests = [True, True, True, True, True]

    ##########################################
    # Deliverable 1 TESTS Rays Sphere Intersection
    ##########################################
    if enabled_tests[0]:
        # Point tests for ray-sphere intersection
        sphere = Sphere(1, np.array([0, 0, 0]))
        rays = Rays(np.array([
            # Moving ray origin along y-axis with x, z axis fixed
            [0, 2, -2],  # should not intersect
            [0, 1, -2],  # should intersect once (tangent)
            [0, 0, -2],  # should intersect twice
            [0, -1, -2],  # should intersect once (bottom)
            [0, -2, -2],  # should not intersect
            # Move back along the z-axis
            [0, 0, -4],  # should have t 2 greater than that of origin [0, 0, -2]
        ]), np.array([[0, 0, 1]]))

        expected_ts = np.array([np.inf, 2, 1, 2, np.inf, 3], dtype=np.float64)
        hit_distances = sphere.intersect(rays)

        if np.allclose(hit_distances, expected_ts):
            print("Rays-Sphere Intersection point test passed")
        else:
            raise ValueError(f'Expected intersection distances {expected_ts}\n'
                             f'Actual intersection distances {hit_distances}')

    ##########################################
    # Deliverable 2 TESTS Eye Ray Generation
    ##########################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        # TIP: if you haven't yet vectorized your code, try debugging at a lower resolution
        # scene = Scene(w=1024, h=768)
        scene = Scene(w=200, h=100)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        sphere = Sphere(10, np.array([0, 0, 50]))

        vectorized_eye_rays = scene.generate_eye_rays()
        hit_distances = sphere.intersect(vectorized_eye_rays)

        # Visualize hit distances
        # plt.matshow(hit_distances.reshape((768, 1024)))
        plt.matshow(hit_distances.reshape((100, 200)))
        plt.title("Distances")
        plt.colorbar()
        plt.show()

    ##########################################
    # Deliverable 3 TESTS Rays Scene Intersection
    ##########################################
    if enabled_tests[2]:
        # Set up scene
        scene = Scene(w=200, h=100)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        scene.add_spheres([
            # x+ => right; y+ => up; z+ => close to camera
            # Left Sphere in the image
            Sphere(16.5, np.array([-30, -22.5, 140]),
                   rho=np.array([0.999, 0.5, 0.5])),
            # Left Sphere in the image
            Sphere(16.5, np.array([22, -27.5, 140]),
                   rho=np.array([0.5, 0.999, 0.5])),
            # Ground
            Sphere(1650, np.array([23, -1700, 140]),
                   rho=np.array([0.7, 0.7, 0.7])),
        ])

        vectorized_eye_rays = scene.generate_eye_rays()
        hit_distances, hit_normals, hit_ids = scene.intersect(
            vectorized_eye_rays)

        # Visualize distances, normals and IDs
        plt.matshow(hit_distances.reshape((100, 200)))
        plt.title("Distances")
        plt.show()
        plt.matshow(np.abs(hit_normals.reshape((100, 200, 3))))
        plt.title("Normals")
        plt.show()
        plt.matshow(hit_ids.reshape((100, 200)))
        plt.title("IDs")
        plt.show()

    ##########################################
    # Deliverable 4 TESTS Shading
    ##########################################
    if enabled_tests[3]:
        # Set up scene
        scene = Scene(w=200, h=100)
        scene.set_camera_parameters(
            eye=np.array([0, 0, -10], dtype=np.float64),
            at=normalize(np.array([0, 0, 1], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=60
        )
        scene.add_spheres([
            # x+ => right; y+ => up; z+ => close to camera
            # Left Sphere in the image
            Sphere(16.5, np.array([-30, -22.5, 140]),
                   rho=np.array([0.999, 0.5, 0.5])),
            # Right Sphere in the image
            Sphere(16.5, np.array([22, -27.5, 140]),
                   rho=np.array([0.5, 0.999, 0.5])),
            # Ground
            Sphere(1650, np.array([23, -1700, 140]),
                   rho=np.array([0.7, 0.7, 0.7])),
        ])
        scene.add_lights([
            {
                "type": "directional",
                # Top-Left of the scene
                "direction": normalize(np.array([1, 1, 0])),
                "color": np.array([2, 0, 0, 1])  # Red
            },
            {
                "type": "directional",
                # Top-Right of the scene
                "direction": normalize(np.array([-1, 1, 0])),
                "color": np.array([0, 2, 0, 1])  # Green
            },
            {
                "type": "directional",
                # Top of the scene
                "direction": normalize(np.array([0, 1, 0])),
                "color": np.array([2, 2, 2, 1])  # White
            },
        ])

        vectorized_eye_rays = scene.generate_eye_rays()
        L = shade(scene, vectorized_eye_rays)

        plt.matshow(L)
        plt.title("Rendered Image")
        # plt.savefig("numpy-image.png")
        plt.show()

    ##########################################
    # Deliverable 5 TESTS Shading with Point Light
    ##########################################
    # if enabled_tests[4]:
    #     # Set up scene
    #     scene = Scene(w=1024, h=768)
    #     scene.set_camera_parameters(
    #         eye=np.array([0, 0, -10], dtype=np.float64),
    #         at=normalize(np.array([0, 0, 1], dtype=np.float64)),
    #         up=np.array([0, 1, 0], dtype=np.float64),
    #         fov=60
    #     )
    #     scene.add_spheres([
    #         # x+ => right; y+ => up; z+ => close to camera
    #         # Left Sphere in the image
    #         Sphere(16.5, np.array([-30, -22.5, 140]), rho=np.array([0.999, 0.5, 0.5])),
    #         # Right Sphere in the image
    #         Sphere(16.5, np.array([22, -27.5, 140]), rho=np.array([0.5, 0.999, 0.5])),
    #         # Ground
    #         Sphere(1650, np.array([23, -1700, 140]), rho=np.array([0.7, 0.7, 0.7])),
    #     ])
    #     scene.add_lights([
    #         {
    #             "type": "point",
    #             # Top Left
    #             "position": np.array([-50, 30, 140], dtype=np.float64),
    #             "color": np.array([1e5, 0, 0, 1])  # Red
    #         },
    #         {
    #             "type": "point",
    #             # Top Right
    #             "position": np.array([50, 30, 140], dtype=np.float64),
    #             "color": np.array([0, 1e5, 0, 1])  # Green
    #         },
    #         {
    #             "type": "point",
    #             # Between the spheres
    #             "position": np.array([-4, -25, 140], dtype=np.float64),
    #             "color": np.array([1e4, 1e4, 1e4, 1])  # White
    #         },
    #         {
    #             "type": "point",
    #             # Center but closer to brighten up the scene
    #             "position": np.array([0, 30, 90], dtype=np.float64),
    #             "color": np.array([1e5, 1e5, 1e5, 1])  # White
    #         },
    #     ])
    #
    #     vectorized_eye_rays = scene.generate_eye_rays()
    #     L = shade(scene, vectorized_eye_rays)
    #
    #     plt.matshow(L)
    #     plt.title("Rendered Image")
    #     # plt.savefig("numpy-image.png")
    #     plt.show()
