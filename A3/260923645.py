# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################
import matplotlib.pyplot as plt  # plotting
import numpy as np  # all of numpy
# just used to load a mesh, now
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
                np.logical_and(0 <= alpha,  alpha < 1),
                np.logical_and(0 <= beta,  beta < 1)
            ),
            np.logical_and(0 <= alpha + beta,  alpha + beta < 1)
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
        # print(hit_distances.shape)
        # print(triangle_hit_ids.shape)
        # print(barys.shape)

        # BEGIN CODE

        # find indexes to update
        indexes_to_update = np.array(np.where(triangle_hit_ids != -1)[0])

        # calculate aphas, betas and gammas
        alphas = np.array([barys[indexes_to_update][:, 0]]).T
        betas = np.array([barys[indexes_to_update][:, 1]]).T
        gammas = np.array([barys[indexes_to_update][:, 2]]).T

        # print(alphas.shape)
        # print(betas.shape)
        # print(gammas.shape)
        # print("-----")

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

    def render(self, eye_rays, sampling_type=UNIFORM_SAMPLING):
        # vectorized scene intersection
        shadow_ray_o_offset = 1e-8
        distances, normals, ids = self.intersect(eye_rays)

        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = eye_rays(distances)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array(
            [obj.brdf_params for obj in self.geometries]), np.array([0, 0, 0, 1])[np.newaxis, :]))[ids]

        L_e = np.concatenate((np.array([obj.Le for obj in self.geometries]), np.array(
            [0, 0, 0])[np.newaxis, :]))[ids]  # stores the light emission at every pixel
        objects = np.concatenate(
            (np.array([obj for obj in self.geometries]), np.array([-1])))
        hit_objects = np.concatenate(
            (np.array([obj for obj in self.geometries]), np.array([-1])))[ids]

        # initialize the output "image" (i.e., vector; still needs to be reshaped)
        L = np.zeros(normals.shape, dtype=np.float64)

        # Directly render light sources
        L = np.where(np.logical_and(L_e != np.array(
            [0, 0, 0]), (ids != -1)[:, np.newaxis]), L_e, L)  # generate light directly on the scene

        # BEGIN SOLUTION
        # PLACEHOLDER: our base code renders out debug normals.
        # [TODO] Replace these next three lines with your
        # solution for your deliverables
        if sampling_type == UNIFORM_SAMPLING:
            # Generate random Ws
            sigma1s = np.random.rand(len(eye_rays.Os))
            sigma2s = np.random.rand(len(eye_rays.Os))
            wzs = 2*sigma1s - 1
            rs = np.sqrt(-np.power(wzs, 2) + 1)
            thetas = 2*math.pi*sigma2s
            wxs = rs*np.cos(thetas)
            wys = rs*np.sin(thetas)
            Ws = np.stack((wxs, wys, wzs), axis=1)

            # trace rays in the direction from Ws from hit_points
            Xs = hit_points + shadow_ray_o_offset*normals
            shadow_rays = Rays(Xs, Ws)

            # since lights are geometries, check if intersect with lights
            _, _, lights_ids = self.intersect(shadow_rays)

            # create les
            Les = np.concatenate((np.array([obj.Le for obj in self.geometries]), np.array(
                [0, 0, 0])[np.newaxis, :]))[lights_ids]

            # calculate direct illumination
            frs = np.tile(np.array([0, 0, 0]), (len(ids), 1))

            alphas = brdf_params[:, 3]
            reflectances = brdf_params[:, 0:3]

            diffuse_reflectances = reflectances/math.pi

            wos = -eye_rays.Ds
            wrs = 2*np.sum(normals*wos, axis=1)[:, np.newaxis]*normals-wos
            product = (np.sum(wrs*Ws, axis=1))**alphas
            product = np.where(product > 0, product, 0)
            specular_reflectances = (
                (reflectances * (alphas + 1)[:, np.newaxis]) / 2*math.pi) * product[:, np.newaxis]
            frs = np.where((alphas == 1)[..., None], diffuse_reflectances, frs)
            frs = np.where((alphas > 1)[..., None], specular_reflectances, frs)

            maxes = np.sum(normals*Ws, axis=1)
            maxes = np.where(maxes > 0, maxes, 0)

            # update L
            L += Les*frs*maxes[:, np.newaxis]*4*math.pi
            L = L.reshape((self.h, self.w, 3))

        elif sampling_type == LIGHT_SAMPLING:
            for light in self.lights:

                # calculate delta_max and omega_e
                delta_max = np.arcsin(
                    light.r/np.linalg.norm((light.c - hit_points), axis=1))
                omega_e = 2*math.pi*(1 - np.cos(delta_max))
                p_light_w = 1/omega_e

                # generate Ws
                sigma1s = np.random.rand(len(eye_rays.Os))
                sigma2s = np.random.rand(len(eye_rays.Os))
                wzs = 1 - sigma1s*(1 - np.cos(delta_max))
                rs = np.sqrt(-np.power(wzs, 2) + 1)
                thetas = 2*math.pi*sigma2s
                wxs = rs*np.cos(thetas)
                wys = rs*np.sin(thetas)
                Ws = np.stack((wxs, wys, wzs), axis=1)

                # generate Ws2
                sigma1s_2 = np.random.rand(len(eye_rays.Os))
                sigma2s_2 = np.random.rand(len(eye_rays.Os))
                wzs_2 = 1 - sigma1s_2*(1 - np.cos(delta_max))
                rs_2 = np.sqrt(-np.power(wzs_2, 2) + 1)
                thetas_2 = 2*math.pi*sigma2s_2
                wxs_2 = rs_2*np.cos(thetas_2)
                wys_2 = rs_2*np.sin(thetas_2)
                Ws_2 = np.stack((wxs_2, wys_2, wzs_2), axis=1)

                As = normalize2D(light.c - hit_points)
                Bs = normalize2D(np.cross(Ws_2, As))
                Cs = normalize2D(np.cross(As, Bs))

                As_0, Bs_0, Cs_0 = As[:, 0], Bs[:, 0], Cs[:, 0]
                M_0 = np.stack((Cs_0, Bs_0, As_0), axis=1)
                As_1, Bs_1, Cs_1= As[:, 1], Bs[:, 1], Cs[:, 1]
                M_1 = np.stack((Cs_1, Bs_1, As_1), axis=1)
                As_2, Bs_2, Cs_2 = As[:, 2], Bs[:, 2], Cs[:, 2]
                M_2 = np.stack((Cs_2, Bs_2, As_2), axis=1)

                Ms = np.stack((M_0, M_1, M_2), axis=1)
                Ws = np.sum(Ms*Ws[:, None, :], axis=2)

                # Light Importance MC estimators
                # trace rays in the direction from Ws from hit_points
                Xs = hit_points + shadow_ray_o_offset*normals
                shadow_rays = Rays(Xs, Ws)

                # since lights are geometries, check if intersect with lights
                _, _, lights_ids = self.intersect(shadow_rays)

                # create les
                Les = np.concatenate((np.array([obj.Le for obj in self.geometries]), np.array(
                    [0, 0, 0])[np.newaxis, :]))[lights_ids]

                # calculate direct illumination
                frs = np.tile(np.array([0, 0, 0]), (len(ids), 1))

                alphas = brdf_params[:, 3]
                reflectances = brdf_params[:, 0:3]

                diffuse_reflectances = reflectances/math.pi

                wos = -eye_rays.Ds
                wrs = 2*np.sum(normals*wos, axis=1)[:, np.newaxis]*normals-wos
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

                # update L
                L += (Les*frs*maxes[:, np.newaxis])/p_light_w[..., None]
        
        elif sampling_type == BRDF_SAMPLING:
            alphas = brdf_params[:, 3]

            # generate Ws, canonical orientation

            # calculate wos and wrs

            # rotate Ws orientation

            # calculate p_brdf_w

            # calculate direct illumination

            # update L
            
            L = np.where(np.logical_and(L_e != np.array(
                [0, 0, 0]), (ids != -1)[:, np.newaxis]), L_e, L)
            L = L.reshape((self.h, self.w, 3))         

        return L
        # END SOLUTION

    def progressive_render_display(self, jitter=False, total_spp=20,
                                   sampling_type=UNIFORM_SAMPLING):
        # matplotlib voodoo to support redrawing on the canvas
        plt.figure()
        plt.ion()
        plt.show()

        L = np.zeros((self.h, self.w, 3), dtype=np.float64)

        # more matplotlib voodoo: update the plot using the
        # image handle instead of looped imshow for performance
        image_data = plt.imshow(L)

        # BEGIN SOLUTION
        # your progressive rendering display loop

        for i in range(total_spp):
            vectorized_eye_rays = self.generate_eye_rays(jitter)
            L -= L/(i+1)
            L += self.render(vectorized_eye_rays, sampling_type)/(i+1)
            image_data.set_data(L)
            plt.pause(0.001)
            plt.title(f"current spp: {i} of {1 * total_spp}")
        # END CODE

        plt.savefig(f"render-{total_spp}spp.png")
        plt.show(block=True)


if __name__ == "__main__":
    enabled_tests = [False, True, False]
    open("./plate1.obj")
    open("./plate2.obj")
    open("./plate3.obj")
    open("./plate4.obj")
    open("./floor.obj")

    # Create test scene and test sphere
    scene = Scene(w=int(64), h=int(64))  # TODO: debug at lower resolution
    scene.set_camera_parameters(
        eye=np.array([0, 2, 15], dtype=np.float64),
        at=normalize(np.array([0, -2, 2.5], dtype=np.float64)),
        up=np.array([0, 1, 0], dtype=np.float64),
        fov=int(40)
    )

    # Veach Scene Lights
    scene.add_geometries([Sphere(0.0333, np.array([3.75, 0, 0]),
                                 Le=10 * np.array([901.803, 0, 0])),
                          Sphere(0.1, np.array([1.25, 0, 0]),
                                 Le=10 * np.array([0, 100, 0])),
                          Sphere(0.3, np.array([-1.25, 0, 0]),
                                 Le=10 * np.array([0, 0, 11.1111])),
                          Sphere(0.9, np.array([-3.75, 0, 0]),
                                 Le=10 * np.array([1.23457, 1.23457, 1.23457])),
                          Sphere(0.5, np.array([-10, 10, 4]),
                                 Le=np.array([800, 800, 800]))])

    # Geometry
    scene.add_geometries([Mesh("plate1.obj",
                               brdf_params=np.array([1, 1, 1, 30000])),
                          Mesh("plate2.obj",
                               brdf_params=np.array([1, 1, 1, 5000])),
                          Mesh("plate3.obj",
                               brdf_params=np.array([1, 1, 1, 1500])),
                          Mesh("plate4.obj",
                               brdf_params=np.array([1, 1, 1, 100])),
                          Mesh("floor.obj",
                               brdf_params=np.array([0.5, 0.5, 0.5, 1]))])

    #########################################################################
    # Deliverable 1 TEST: comment/modify as you see fit
    #########################################################################
    if enabled_tests[0]:
        scene.progressive_render_display(
            total_spp=1024, jitter=True, sampling_type=UNIFORM_SAMPLING)
        scene.progressive_render_display(
            total_spp=32768, jitter=True, sampling_type=UNIFORM_SAMPLING)
        scene.progressive_render_display(
            total_spp=131072, jitter=True, sampling_type=UNIFORM_SAMPLING)

    #########################################################################
    # Deliverable 2 TEST: comment/modify as you see fit
    #########################################################################
    if enabled_tests[1]:
        scene.progressive_render_display(
            total_spp=1, jitter=True, sampling_type=LIGHT_SAMPLING)
        scene.progressive_render_display(
            total_spp=10, jitter=True, sampling_type=LIGHT_SAMPLING)
        scene.progressive_render_display(
            total_spp=100, jitter=True, sampling_type=LIGHT_SAMPLING)

    #########################################################################
    # Deliverable 3 TEST (Only for ECSE 546 students!): comment/modify as you see fit
    #########################################################################
    if enabled_tests[2]:
        scene.progressive_render_display(
            total_spp=1, jitter=True, sampling_type=MIS_SAMPLING)
        scene.progressive_render_display(
            total_spp=10, jitter=True, sampling_type=MIS_SAMPLING)
        scene.progressive_render_display(
            total_spp=100, jitter=True, sampling_type=MIS_SAMPLING)
