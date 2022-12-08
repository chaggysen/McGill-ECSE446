# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
# just used to load a mesh and compute per-vertex normals
from gpytoolbox import read_mesh, per_vertex_normals, per_face_normals
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
    Note - This function is only for normalizing batched 2D vectors.
    """
    norm = np.linalg.norm(vs, axis=1, keepdims=True)
    return vs/norm


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
    n = np.cross(t1 - t0, t2 - t0, axis=1)
    return np.linalg.norm(n, axis=1) / 2


def ray_mesh_intersect(origin, dir, tri):
    intersection = np.ones_like(dir) * -1
    intersection[:, 2] = np.Inf
    dir = dir[:, None]

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]  # (num_triangles, 3)
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
                np.logical_and(0 <= alpha, alpha < 1),
                np.logical_and(0 <= beta, beta < 1)
            ),
            np.logical_and(0 <= alpha + beta, alpha + beta < 1)
        )  # (num_rays, num_tri)
        # (num_rays, num_tri, 3)
        e1_expanded = np.tile(e1[None], (dir.shape[0], 1, 1))
        dot_temp = np.sum(
            s1[cond_bool] * e1_expanded[cond_bool], axis=1)  # (num_rays,)
        results_cond1 = results[cond_bool]
        cond_bool2 = dot_temp != 0

        if cond_bool2.sum() > 0:
            coefficient2 = np.reciprocal(dot_temp)
            # (num_rays, num_tri, 3)
            e2_expanded = np.tile(e2[None], (dir.shape[0], 1, 1))
            t = coefficient2 * np.sum(s2[cond_bool][cond_bool2] *
                                      e2_expanded[cond_bool][cond_bool2],
                                      axis=1)
            results_cond1[cond_bool2] = t
        results[cond_bool] = results_cond1
    results[results <= 0] = np.Inf
    hit_id = np.argmin(results, axis=1)
    min_val = np.min(results, axis=1)
    hit_id[min_val == np.Inf] = -1
    return min_val, hit_id


# ===================== our replacement for gpytoolbox routines =====================


class Mesh(Geometry):
    def __init__(self, filename, brdf_params=np.array([0, 0, 0, 1]), Le=np.array([0, 0, 0])):
        self.v, self.f = read_mesh(filename)
        self.brdf_params = brdf_params
        self.Le = Le
        # BEGIN CODE
        self.face_normals = per_face_normals(self.v, self.f, unit_norm=True)
        self.per_vectex_normals = per_vertex_normals(
            self.v, self.f)
        # END CODE
        super().__init__()

    def intersect(self, rays):
        hit_normals = np.tile(
            np.array([np.inf, np.inf, np.inf]), (len(rays.Os), 1))

        hit_distances, triangle_hit_ids = ray_mesh_intersect(
            rays.Os, rays.Ds, self.v[self.f])
        intersections = rays.Os + hit_distances[:, None] * rays.Ds
        tris = self.v[self.f[triangle_hit_ids]]
        barys = get_bary_coords(intersections, tris)

        # BEGIN CODE
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


class Sphere(Geometry):
    EPSILON_SPHERE = 1e-4

    def __init__(self, r, c, brdf_params=np.array([0, 0, 0, 1]), Le=np.array([0, 0, 0])):
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


# Enumerate the different importance sampling strategies we will implement
IMPLICIT_UNIFORM_SAMPLING, EXPLICIT_UNIFORM_SAMPLING, IMPLICIT_BRDF_SAMPLING, EXPLICIT_LIGHT_BRDF_SAMPLING = range(
    4)


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
                self.add_lights([geometries[i]])

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

        # BEGIN CODE: feel free to remove any/all of our placeholder code, below
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
        ray_direction = normalize2D(ray_direction)  # change
        return Rays(np.tile(ray_origin, (len(ray_direction), 1)), np.array(ray_direction))
        # END CODE

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        hit_distances = np.zeros((rays.Os.shape[0],), dtype=np.float64)
        hit_distances[:] = np.inf
        hit_normals = np.zeros(rays.Os.shape, dtype=np.float64)
        hit_normals[:, :] = np.array([np.inf, np.inf, np.inf])
        hit_ids = np.zeros((rays.Os.shape[0],), dtype=np.int)
        hit_ids[:] = -1

        # BEGIN CODE
        for geo_idx in range(len(self.geometries)):
            geometry = self.geometries[geo_idx]
            result = geometry.intersect(rays)
            distances, normals = result[0], result[1]
            hit_normals = np.where(
                (distances < hit_distances)[..., None], normals, hit_normals)
            hit_ids = np.where((distances < hit_distances), geo_idx, hit_ids)
            hit_distances = np.where(
                distances < hit_distances, distances, hit_distances)
        # END CODE

        return hit_distances, hit_normals, hit_ids

    def render(self, eye_rays, num_bounces=3, sampling_type=IMPLICIT_BRDF_SAMPLING):
        # vectorized scene intersection
        shadow_ray_o_offset = 1e-8
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),
                                      np.array([0, 0, 0, 1])[np.newaxis, :]))[ids]
        L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),
                              np.array([0, 0, 0])[np.newaxis, :]))[ids]
        objects = np.concatenate((np.array([obj for obj in self.geometries]),
                                  np.array([-1])))
        hit_objects = np.concatenate((np.array([obj for obj in self.geometries]),
                                      np.array([-1])))[ids]

        # initialize the output "image" (i.e., vector; still needs to be reshaped)
        L = np.zeros(normals.shape, dtype=np.float64)

        # Directly render light sources
        L = np.where(np.logical_and(L_e != np.array(
            [0, 0, 0]), (ids != -1)[:, np.newaxis]), L_e, L)

        # BEGIN SOLUTION
        if sampling_type == IMPLICIT_BRDF_SAMPLING:
            prev_Ws = None
            throughputs = np.ones(shape=normals.shape)
            active = np.ones((normals.shape), dtype=bool)
            global_Les = np.zeros(shape=normals.shape)
            for bounce_nb in range(num_bounces):
                alphas = brdf_params[:, 3]
                reflectances = brdf_params[:, 0:3]

                # generate Ws, canonical orientation
                sigma1s = np.random.rand(len(eye_rays.Os))
                sigma2s = np.random.rand(len(eye_rays.Os))
                wzs = np.power(sigma1s, (1/(alphas+1)))
                rs = np.sqrt(-np.power(wzs, 2) + 1)
                thetas = 2*math.pi*sigma2s
                wxs = rs*np.cos(thetas)
                wys = rs*np.sin(thetas)
                Ws = np.stack((wxs, wys, wzs), axis=1)

                # calculate wos and wrs
                wos = -eye_rays.Ds if bounce_nb == 0 else -prev_Ws
                wrs = 2*np.sum(normals*wos, axis=1)[:, np.newaxis]*normals-wos

                # rotate Ws orientation
                sigma1s_2 = np.random.rand(len(eye_rays.Os))
                sigma2s_2 = np.random.rand(len(eye_rays.Os))
                wzs_2 = np.power(sigma1s_2, (1/(alphas+1)))
                rs_2 = np.sqrt(-np.power(wzs_2, 2) + 1)
                thetas_2 = 2*math.pi*sigma2s_2
                wxs_2 = rs_2*np.cos(thetas_2)
                wys_2 = rs_2*np.sin(thetas_2)
                Ws_2 = np.stack((wxs_2, wys_2, wzs_2), axis=1)

                centers = np.zeros(shape=normals.shape)
                centers = np.where((alphas == 1)[..., None], normals, centers)
                centers = np.where((alphas > 1)[..., None], wrs, centers)
                As = normalize2D(centers)
                Bs = normalize2D(np.cross(Ws_2, As))
                Cs = normalize2D(np.cross(As, Bs))

                As_0, Bs_0, Cs_0 = As[:, 0], Bs[:, 0], Cs[:, 0]
                M_0 = np.stack((Cs_0, Bs_0, As_0), axis=1)
                As_1, Bs_1, Cs_1 = As[:, 1], Bs[:, 1], Cs[:, 1]
                M_1 = np.stack((Cs_1, Bs_1, As_1), axis=1)
                As_2, Bs_2, Cs_2 = As[:, 2], Bs[:, 2], Cs[:, 2]
                M_2 = np.stack((Cs_2, Bs_2, As_2), axis=1)

                Ms = np.stack((M_0, M_1, M_2), axis=1)
                Ws = np.sum(Ms*Ws[:, None, :], axis=2)

                # calculate p_brdf_w
                p_brdf_ws = np.zeros(shape=(len(ids),))

                # diffuse
                product_diffuse = np.sum(normals*Ws, axis=1)
                product_diffuse = np.where(
                    product_diffuse > 0, product_diffuse, 0)
                product_diffuse = (1/np.pi)*product_diffuse
                p_brdf_ws = np.where((alphas == 1), product_diffuse, p_brdf_ws)

                # glossy Phong
                product_phong = np.sum(wrs*Ws, axis=1)**alphas
                product_phong = np.where(product_phong > 0, product_phong, 0)
                product_phong = ((alphas+1)/(2*np.pi))*product_phong
                p_brdf_ws = np.where((alphas > 1), product_phong, p_brdf_ws)

                # trace rays in the direction from Ws from hit_points
                Xs = hit_points + shadow_ray_o_offset*normals
                shadow_rays = Rays(Xs, Ws)

                # since lights are geometries, check if intersect with lights
                shadow_distances, shadow_normals, shadow_ids = self.intersect(
                    shadow_rays)

                # create les
                Les = np.concatenate((np.array([obj.Le for obj in self.geometries]), np.array(
                    [0, 0, 0])[np.newaxis, :]))[shadow_ids]

                global_Les = np.where(np.logical_and(Les != np.array(
                    [0, 0, 0]), (active == np.array([True, True, True]))), Les, global_Les)

                # calculate direct illumination
                frs = np.zeros(shape=normals.shape)

                diffuse_reflectances = reflectances/math.pi

                product = (np.sum(wrs*Ws, axis=1))**alphas
                product = np.where(product > 0, product, 0)
                specular_reflectances = (
                    (reflectances * (alphas + 1)[:, np.newaxis]) / 2*math.pi) * product[:, np.newaxis]
                frs = np.where(
                    (alphas == 1)[..., None], diffuse_reflectances, frs)
                frs = np.where((alphas > 1)[..., None],
                               specular_reflectances, frs)

                maxes = np.sum(normals*Ws, axis=1)
                maxes = np.where(maxes > 0, maxes, 0)

                # update throughputs
                new_throughputs = throughputs * \
                    (frs*maxes[:, np.newaxis])/p_brdf_ws[..., None]
                throughputs = np.where(active == np.array(
                    [True, True, True]), new_throughputs, throughputs)

                # update active
                active = np.where(np.logical_or(((shadow_ids == -1)[:, np.newaxis]), (Les != np.array(
                    [0, 0, 0]))), np.array([False, False, False]), active)

                # update prev_Ws, prev_normals, brdf_params and hit_points
                prev_Ws = Ws
                brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.geometries]),
                                              np.array([0, 0, 0, 1])[np.newaxis, :]))[shadow_ids]
                normals = shadow_normals
                hit_points = shadow_rays(shadow_distances)

            L += throughputs*global_Les
            L = np.where(np.logical_and(L_e != np.array(
                [0, 0, 0]), (ids != -1)[:, np.newaxis]), L_e, L)
            L = L.reshape((self.h, self.w, 3))

        return L
        # END SOLUTION

    def progressive_render_display(self, jitter=False, total_spp=20, num_bounces=3,
                                   sampling_type=IMPLICIT_BRDF_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(L)

        # BEGIN CODE (note: we will not grade your progressive rendering code in A4)

        for i in range(total_spp):
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            L -= L/(i+1)
            L += self.render(vectorized_eye_rays,
                             num_bounces, sampling_type)/(i+1)
            image_data.set_data(L)
            plt.pause(0.001)
            plt.title(f"current spp: {i} of {1 * total_spp}")
        # END CODE

        plt.savefig(f"render-{total_spp}spp.png")
        plt.show(block=True)


if __name__ == "__main__":
    enabled_tests = [True, True, True]
    # NOTE: ECSE 546 students can set the second boolean to True
    enable_deliverables = [True, False]

    #########################################################################
    # Test Case 1: Default Cornell Box Scene
    #########################################################################
    if enabled_tests[0]:
        # Create test scene and test sphere
        # TODO: debug at lower resolution
        scene = Scene(w=int(128 / 2), h=int(128 / 2))
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_ceiling.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_greenwall.obj",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.obj",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1]))
        ])

        #########################################################################
        # Deliverable 1: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            scene.progressive_render_display(total_spp=512, jitter=True, num_bounces=2,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=3,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=2048, jitter=True, num_bounces=4,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        # Deliverable 2: ECSE 546 Only - Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=3,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=3,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=100, jitter=True, num_bounces=3,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.progressive_render_display(total_spp=1, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=100, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

    #########################################################################
    # Test Case 2: Scene with decreasing light size (constant power)
    #########################################################################
    if enabled_tests[1]:
        # Create test scene and test sphere
        # TODO: debug at lower resolution
        scene = Scene(w=int(512 / 2), h=int(512 / 2))
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_ceiling.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_greenwall.obj",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.obj",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1]))
        ])

        #########################################################################
        # Deliverable 1: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            scene.geometries[0].r = 60
            scene.geometries[0].Le = 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

            scene.geometries[0].r = 30
            scene.geometries[0].Le = 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

            scene.geometries[0].r = 10
            scene.geometries[0].Le = 9 * 4 * \
                1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        # Deliverable 2: ECSE 546 Only - Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.geometries[0].r = 60
            scene.geometries[0].Le = 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.geometries[0].r = 30
            scene.geometries[0].Le = 4 * 1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

            scene.geometries[0].r = 10
            scene.geometries[0].Le = 9 * 4 * \
                1.25 * np.array([15.6, 15.6, 15.6])
            scene.progressive_render_display(total_spp=10, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)

    #########################################################################
    # Test Case 3: Scene with different BRDFs
    #########################################################################
    if enabled_tests[2]:
        # Create test scene and test sphere
        # TODO: debug at lower resolution
        scene = Scene(w=int(512 / 2), h=int(512 / 2))
        scene.set_camera_parameters(
            eye=np.array([278, 273, -770], dtype=np.float64),
            at=(np.array([278, 273, -769], dtype=np.float64)),
            up=np.array([0, 1, 0], dtype=np.float64),
            fov=int(39)
        )

        scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.obj",
                 brdf_params=np.array([0.86, 0.86, 0.86, 1])),
            Mesh("cbox_ceiling.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 50])),
            Mesh("cbox_greenwall.obj",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.obj",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.obj",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.obj",
                 brdf_params=np.array([0.86, 0.86, 0.86, 1000]))
        ])

        #########################################################################
        # Deliverable 1: Implicit BRDF Sampling
        #########################################################################
        if enable_deliverables[0]:
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=3,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=4,
                                             sampling_type=IMPLICIT_BRDF_SAMPLING)

        #########################################################################
        # Deliverable 2: ECSE 546 Only - Explicit Light BRDF Sampling
        #########################################################################
        if enable_deliverables[1]:
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=2,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=3,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
            scene.progressive_render_display(total_spp=1024, jitter=True, num_bounces=4,
                                             sampling_type=EXPLICIT_LIGHT_BRDF_SAMPLING)
