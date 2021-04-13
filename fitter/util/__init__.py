'''Various utility functions for use in fitter likelihoods and scripts'''

from .data import *
from .units import *

import numpy as np


# --------------------------------------------------------------------------
# Generic Helper Functions
# --------------------------------------------------------------------------


def gaussian(x, sigma, mu):
    '''Gaussian PDF, evaluated over `x` with mean `mu` and width `sigma`'''
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * (((x - mu) / sigma) ** 2))
    return norm * exponent


def RV_transform(domain, f_X, h, h_prime):
    '''Transformation of a random variable over a function :math:`g=h^{-1}`'''
    f_Y = f_X(h(domain)) * h_prime(domain)
    return np.nan_to_num(f_Y)


# --------------------------------------------------------------------------
# Integration functions
# --------------------------------------------------------------------------

def MC_sample(poly, M):
    '''Random sampling of `M` points from a 2D `poly` shapely Polygon'''

    import random
    from shapely.prepared import prep

    def uniform_sample(pi, Mi):

        if pi.is_empty:
            return []

        prepped_pi = prep(pi)

        minx, miny, maxx, maxy = pi.bounds

        points = []

        while len(points) < Mi:
            test_pnt = geom.Point(random.uniform(minx, maxx),
                                  random.uniform(miny, maxy))

            if prepped_pi.contains(test_pnt):
                points.append(test_pnt)

        return points

    if poly.geom_type == 'Polygon':
        points = uniform_sample(poly, M)

    elif poly.geom_type == 'MultiPolygon':
        points = []
        tot_area = poly.area

        for poly_i in poly:
            points += uniform_sample(poly_i, M * poly_i.area / tot_area)

    return geom.MultiPoint(points)
