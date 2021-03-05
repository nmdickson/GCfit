'''
util

various utility and helper functions
'''

from .data import *
from .units import *

import numpy


# --------------------------------------------------------------------------
# Generic Helper Functions
# --------------------------------------------------------------------------


# Simple gaussian implementation
def gaussian(x, sigma, mu):
    norm = 1 / (sigma * numpy.sqrt(2 * numpy.pi))
    exponent = numpy.exp(-0.5 * (((x - mu) / sigma) ** 2))
    return norm * exponent


def RV_transform(domain, f_X, h, h_prime):
    '''transformation of a random variable over a function g=h^-1'''
    f_Y = f_X(h(domain)) * h_prime(domain)
    return numpy.nan_to_num(f_Y)
