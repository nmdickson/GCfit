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
