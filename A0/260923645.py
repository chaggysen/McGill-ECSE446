# [TODO] Rename this file to YOUR-STUDENT-ID.py

##########################################
# DO NOT EDIT THESE IMPORT STATEMENTS!
##########################################

import matplotlib.pyplot as plt   # plotting
import numpy as np                # all of numpy, for now...
##########################################


#####################
# Deliverable 1:
#####################
def render_checkerboard_slow(width, height, stride):
    output = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            if ((int(x/stride) + int(y/stride)) % 2) == 1:
                output[y, x] = 1
    return output


def render_checkerboard_fast(width, height, stride):
    # BEGIN SOLUTION
    yx = np.indices((height, width))
    y_idx, x_idx = yx[0:2, :, :]
    y_idx, x_idx = yx[0, :, :], yx[1, :, :]
    sum_table = (y_idx/stride).astype(int) + (x_idx/stride).astype(int)
    sum_table[sum_table % 2 == 1] = 1
    sum_table[sum_table % 2 != 1] = 0
    return sum_table
    # END SOLUTION


#####################
# Deliverable 2
#####################
def circle(x, y):
    return (x * x) + (y * y) - 1/4


def heart(x, y):
    return (x**2 + y**2 - 1)**3 - \
        (x*x * y*y*y)



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def visibility_2d(scene_2d):
    width, height = scene_2d["output"]["width"], scene_2d["output"]["height"]

    x_min, x_max = scene_2d['view']['xmin'], scene_2d['view']['xmax']
    y_min, y_max = scene_2d['view']['ymin'], scene_2d['view']['ymax']

    # get the indices
    yx = np.indices((height, width))
    y_idx, x_idx = yx[0:2, :, :]
    y_idx, x_idx = yx[0, :, :], yx[1, :, :]

    x_norm = NormalizeData(x_idx) - 0.5
    y_norm = - (NormalizeData(y_idx) - 0.5)

    # # normalize and transform to image coord
    x_img_coord = x_idx - (width//2)
    y_img_coord = -(y_idx - (height//2))

    # # perform action and filter table 
    result_table = scene_2d['shape_2d'](x_norm, y_norm)
    # & (x_min <= x_img_coord) & (x_max >= x_img_coord) & (y_max >= y_img_coord) & (y_min <= y_img_coord)
    result_table = np.where((result_table <= 0), 1, 0)
    # result_table[result_table <= 0] = 1
    # result_table[result_table > 0] = 0

    return result_table






    # BEGIN SOLUTION
    # END SOLUTION


#####################
# Deliverable 3
#####################
def heart_3d(x, y):
    alpha = 9/4
    beta = 9/200

    # handle y = 0 discontinuity with an offset hack
    y = np.where(np.isclose(y, 0), y + 0.005, y)

    z_roots = np.empty((x.shape[0], x.shape[1], 6), dtype="complex")

    z_roots[:, :, 0] = np.where(np.isclose(y, 0), np.sqrt(0j + (1 - x**2)/complex(alpha)), -np.sqrt(6)*np.sqrt(0j + -2**(2/3)*3**(1/3)*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(
        1/3) - 6*x**2/complex(alpha) - 6*y**2/complex(alpha) + 6/complex(alpha) - 2*2**(1/3)*3**(2/3)*complex(beta)*y**3/(complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)))/6)
    z_roots[:, :, 1] = -z_roots[:, :, 0]
    z_roots[:, :, 2] = np.where(np.isclose(y, 0), 1j, -np.sqrt(0j + -6**(2/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(0j + 27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(2/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) - 3*2**(2/3)*3**(1/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(2/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) - 6*3**(1/3)*complex(alpha)**2*x**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) + 6*3**(5/6)*(1j)*complex(alpha)**2*x**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) - 6*3**(1/3)*complex(alpha)**2*y**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(
        alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) + 6*3**(5/6)*(1j)*complex(alpha)**2*y**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) + 6*3**(1/3)*complex(alpha)**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) - 6*3**(5/6)*(1j)*complex(alpha)**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) + 12*2**(1/3)*complex(beta)*y**3/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) - 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3))))
    z_roots[:, :, 3] = -z_roots[:, :, 2]
    z_roots[:, :, 4] = np.where(np.isclose(y, 0), 1j, -np.sqrt(0j + -6**(2/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(0j + 27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(2/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) + 3*2**(2/3)*3**(1/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(2/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) - 6*3**(1/3)*complex(alpha)**2*x**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) - 6*3**(5/6)*(1j)*complex(alpha)**2*x**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) - 6*3**(1/3)*complex(alpha)**2*y**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(
        alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) - 6*3**(5/6)*(1j)*complex(alpha)**2*y**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) + 6*3**(1/3)*complex(alpha)**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) + 6*3**(5/6)*(1j)*complex(alpha)**2*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3)) + 12*2**(1/3)*complex(beta)*y**3/(6*3**(1/3)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3) + 6*3**(5/6)*(1j)*complex(alpha)**3*(np.sqrt(3)*np.sqrt(27*x**4*y**6/complex(alpha)**6 - 54*complex(beta)*x**4*y**6/complex(alpha)**7 - 54*complex(beta)*x**2*y**8/complex(alpha)**7 + 54*complex(beta)*x**2*y**6/complex(alpha)**7 + 27*complex(beta)**2*x**4*y**6/complex(alpha)**8 + 54*complex(beta)**2*x**2*y**8/complex(alpha)**8 - 54*complex(beta)**2*x**2*y**6/complex(alpha)**8 + 27*complex(beta)**2*y**10/complex(alpha)**8 - 54*complex(beta)**2*y**8/complex(alpha)**8 + 27*complex(beta)**2*y**6/complex(alpha)**8 - 4*complex(beta)**3*y**9/complex(alpha)**9) - 9*x**2*y**3/complex(alpha)**3 + 9*complex(beta)*x**2*y**3/complex(alpha)**4 + 9*complex(beta)*y**5/complex(alpha)**4 - 9*complex(beta)*y**3/complex(alpha)**4)**(1/3))))
    z_roots[:, :, 5] = -z_roots[:, :, 4]

    # find the smallest (nearest) *real* root, for each pixel
    z_roots = np.where(np.isclose(z_roots.imag, 0), z_roots.real, np.Inf)
    z = np.min(z_roots, axis=2)

    values = (x**2 + (alpha)*z**2 + y**2 - 1)**3 - \
        x**2 * y**3 - (beta) * z**2 * y**3

    normals = np.empty((x.shape[0], x.shape[1], 3))

    normals[:, :, 0] = -2*x*y**3 + 6*x*(alpha*z**2 + x**2 + y**2 - 1)**2
    normals[:, :, 1] = -3*beta*y**2*z**2 - 3*x**2 * \
        y**2 + 6*y*(alpha*z**2 + x**2 + y**2 - 1)**2
    normals[:, :, 2] = 6*alpha*z * \
        (alpha*z**2 + x**2 + y**2 - 1)**2 - 2*beta*y**3*z

    norms = np.linalg.norm(normals, axis=2)

    # normalization with broadcast
    normals[:, :] = normals[:, :]/norms[:, :, np.newaxis]

    return z, values, normals


def render(scene_3d):
    width, height = scene_3d['output']['width'], scene_3d['output']['height']

    x_min, x_max = scene_3d['view']['xmin'], scene_3d['view']['xmax']
    y_min, y_max = scene_3d['view']['ymin'], scene_3d['view']['ymax']

    # get the indices
    yx = np.indices((height, width))
    y_idx, x_idx = yx[0:2, :, :]
    y_idx, x_idx = yx[0, :, :], yx[1, :, :]

    shape_3d = scene_3d['shape_3d']

    z, values, normals = heart_3d(x_idx, y_idx)
    


# Some example test routines for the deliverables.
# Feel free to write and include your own tests here.
# Code in this main block will not count for credit,
# but the collaboration and plagiarism policies still hold.
# You can change anything in the mainline -- it will not be graded
if __name__ == "__main__":

    # at some point, your code may (purposefully) propagate np.Inf values,
    # and you may want to disable RuntimeWarnings for them; we will NOT penalize RuntimeWarnings,
    # _so long as your implementation produces the desired output_
    np.seterr(invalid='ignore')

    # convenience variable to enable/disable tests for different deliverables
    enabled_tests = [True, True, True]

    ##########################################
    # Deliverable 1 TESTS
    ##########################################
    if enabled_tests[0]:
        # Test code to visualize the output of the functions
        plt.imshow(render_checkerboard_slow(256, 256, 3))
        plt.show()  # this is a *blocking* function call: code will not continue to execute until you close the plotting window!

        plt.imshow(render_checkerboard_fast(256, 256, 3))
        plt.show()

        plt.imshow(render_checkerboard_slow(256, 256, 3) -
                   render_checkerboard_fast(256, 256, 3))
        plt.show()

        # import _anything you like_ but ONLY in the mainline for testing, not in your solution code above
        import time  # useful for performance measurement; see time.perf_counter()

        # four orders of magnitude, starting from 100
        log_test_lengths = np.arange(3) + 2
        fast_perfs = []
        slow_perfs = []
        for length in log_test_lengths:
            start = time.perf_counter()
            render_checkerboard_fast(10**length, 10**length, 2)
            end = time.perf_counter()
            fast_perfs.append(end - start)

            start = time.perf_counter()
            render_checkerboard_slow(10**length, 10**length, 2)
            end = time.perf_counter()
            slow_perfs.append(end - start)

        plt.title("Checkerboard Performance Comparison")
        plt.xlabel("Output Image Side Length (in pixels)")
        plt.ylabel("Performance (in seconds)")
        plt.plot(10**log_test_lengths, slow_perfs, '-*')
        plt.plot(10**log_test_lengths, fast_perfs, '-x')
        plt.legend(['Naive', 'Fast'])
        plt.show()

    ############################################################################################################################
    # Deliverable 2 TESTS 
    ############################################################################################################################
    if enabled_tests[1]:
        test_scene_2d = {
            "output": {  # output image dimensions
                "width": 100,
                "height": 100
            },
            "shape_2d": circle,  # 2D shape function to query during visibility plotting
            "view": {  # 2D plotting limits
                "xmin": -1,
                "xmax": 1,
                "ymin": -1,
                "ymax": 1
            }
        }

        plt.imshow(visibility_2d(test_scene_2d))
        plt.show()

        test_scene_2d["shape_2d"] = heart
        test_scene_2d["view"]["xmin"] = -1.25
        test_scene_2d["view"]["xmax"] = 1.25
        test_scene_2d["view"]["ymin"] = 1.5
        test_scene_2d["view"]["ymax"] = -1.25
        plt.imshow(visibility_2d(test_scene_2d))
        plt.show()

    ##########################################
    # Deliverable 3 TEST
    ##########################################
    if enabled_tests[2]:
        test_scene_3d = {
            "output": {  # output image dimensions
                "width": 100,
                "height": 100
            },
            "shape_3d": heart_3d,  # 3D shape function to query during rendering
            "lights":  # (directional) lights to use during shading
            [
                {
                    "direction": np.array([-3, 3, -1])/np.linalg.norm(np.array([-3, 3, -1])),
                    "color": np.array([1, 0.125, 0.125, 1])
                },
                {
                    "direction": np.array([3, 3, -1])/np.linalg.norm(np.array([3, 3, -1])),
                    "color": np.array([0.125, 1.0, 0.125, 1])
                },
                {
                    "direction": np.array([0, -3, -1])/np.linalg.norm(np.array([0, -3, -1])),
                    "color": np.array([0.125, 0.125, 1.0, 1])
                }
            ],
            # 2D plotting limits (z limits are -infinity to infinity, i.e., consider all roots without clipping in z)
            "view":
            {
                "xmin": -1.25,
                "xmax": 1.25,
                "ymin": 1.5,
                "ymax": -1.25
            }
        }

        plt.imshow(render(test_scene_3d))
        plt.show()
