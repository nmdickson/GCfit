import scipy.stats
import numpy as np
import astropy.units as u


__all__ = ['angular_width', 'angular_speed', 'interpQuantity']


# TODO should probably be using `Equivalency` class?


def angular_width(D):
    '''AstroPy units conversion equivalency for angular to linear widths.
    See: https://docs.astropy.org/en/stable/units/equivalencies.html
    '''

    D = D.to(u.pc)

    def pc2rad(r):
        return 2 * np.arctan(r / (2 * D.value))

    def rad2pc(θ):
        return np.tan(θ / 2) * (2 * D.value)

    return [(u.pc, u.rad, pc2rad, rad2pc)]


def angular_speed(D):
    '''AstroPy units conversion equivalency for angular to tangential speeds.
    See: https://docs.astropy.org/en/stable/units/equivalencies.html
    '''

    D = D.to(u.pc)

    def kms2asyr(vt):
        return vt / (4.74 * D.value)

    def asyr2kms(μ):
        return 4.74 * D.value * μ

    return [((u.km / u.s), (u.arcsec / u.yr), kms2asyr, asyr2kms)]


# TODO maybe univariatespline instead, that gets used often, or maybe both?
class interpQuantity(scipy.interpolate.interp1d):

    def __init__(self, x, y, bounds_error=False, *args, **kwargs):
        self._xunit = x.unit
        self._yunit = y.unit

        super().__init__(x, y, bounds_error=bounds_error, *args, **kwargs)

    def __call__(self, x):
        return super().__call__(x.to_value(self._xunit)) << self._yunit