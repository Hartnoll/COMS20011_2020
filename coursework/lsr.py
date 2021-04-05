import os
import sys
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(segmentsX, segmentsY, fnc, ones):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    seg_e = []
    for idx, xs in enumerate(segmentsX):
        
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

def least_squares(xs_e, ys):
    return np.linalg.inv(xs_e.T.dot(xs_e)).dot(xs_e.T).dot(ys)

def linear(ones, xs):
    return np.column_stack((xs, ones))

def quad(ones, xs):
    return np.column_stack((xs**2, xs, ones))

def poly3(ones, xs):
    return np.column_stack((xs**3, xs**2, xs, ones))

def poly4(ones, xs):
    return np.column_stack((xs**4, xs**3, xs**2, xs, ones))
    
def poly5(ones, xs):
    return np.column_stack((xs**5, xs**4, xs**3, xs**2, xs, ones))

def poly6(ones, xs):
    return np.column_stack((xs**6, xs**5, xs**4, xs**3, xs**2, xs, ones))

def sin1(ones, xs):
    return np.column_stack((math.sin(xs), ones))

def opt_func(fl, xs, ys):
    ones = np.ones(xs.shape)
    err_dict = {}
    for f in fl:
        xs_e = f(ones, xs)
        Wh = least_squares(xs_e, ys)
        err = 0
        for idx, x in enumerate(xs_e):
            expect = x@Wh
            err += (ys[idx] - expect)**2 
        err_dict[f] = err
    return min(err_dict, key=err_dict.get), min(err_dict.values())


fl = [linear, quad, poly3, poly4, poly5, poly6]

file = sys.argv[1]
xs, ys = load_points_from_file(file)
segmentsX = [xs[x:x+20] for x in range(0, len(xs), 20)]
segmentsY = [ys[x:x+20] for x in range(0, len(ys), 20)]
err_sum = 0
fnc = []
for idx, Xs in enumerate(segmentsX):
    func, err = opt_func(fl, Xs, segmentsY[idx])
    fnc.append(func)
    err_sum += err
print(err_sum)

if sys.argv.__contains__("--plot"):
    ones = np.ones(xs.shape)
    view_data_segments(segmentsX, segmentsY, fnc, ones)