import warnings

import scipy.stats
import numpy as np
import astropy.units as u


__all__ = ['angular_width', 'angular_speed', 'QuantitySpline']


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


def angular_area(D):
    '''AstroPy units conversion equivalency for angular to linear areas.
    See: https://docs.astropy.org/en/stable/units/equivalencies.html

    Given X rad/pc, with X given by your distance conversion,
    then 1 rad^2 = 1 rad x 1 rad / (X rad/pc)^2 = (1/X^2) pc^2
    '''

    D = D.to(u.pc)

    def pc2rad(r):
        return 2 * np.arctan(r / (2 * D.value))

    def pcsq2radsq(r):
        return (1. / pc2rad(1)**2) * r

    def rad2pc(θ):
        return np.tan(θ / 2) * (2 * D.value)

    def radsq2radsq(θ):
        return (1. / rad2pc(1)**2) * θ

    return [(u.pc**2, u.rad**2, pcsq2radsq, radsq2radsq)]


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


# TODO this really needs unittest, the unitless stuff made it complicated
class QuantitySpline(scipy.interpolate.UnivariateSpline):
    '''Subclass of SciPy's UnivariateSpline, supporting AstroPy `Quantity`s'''

    def __init__(self, x, y, w=None, bbox=[None] * 2, k=3, s=0,
                 ext=1, check_finite=False):

        self._xunit = getattr(x, 'unit', None)
        self._yunit = getattr(y, 'unit', None)

        # Convert boundary box, if needed
        if bbox[0] is not None and hasattr(bbox[0], 'unit'):
            bbox[0] = bbox[0].to_value(self._xunit)

        if bbox[1] is not None and hasattr(bbox[1], 'unit'):
            bbox[1] = bbox[1].to_value(self._xunit)

        super().__init__(x, y, w=w, bbox=bbox, k=k, s=s,
                         ext=ext, check_finite=check_finite)

    @classmethod
    def _from_tck(cls, tck, x_unit=None, y_unit=None, ext=0):

        obj = super()._from_tck(tck, ext=ext)

        obj._xunit = x_unit
        obj._yunit = y_unit

        return obj

    def __call__(self, x, nu=0, ext=None):
        '''
        if no _xunit, but x has a unit, assume it matches
        '''

        if hasattr(x, 'unit'):
            x = x.to_value(self._xunit)

        res = super().__call__(x, nu=nu, ext=ext)

        if self._yunit:
            res <<= self._yunit

        return res

    def integral(self, a, b):
        '''
        if no xunit, but a,b does have units, assumes they match
        '''

        if (hasattr(a, 'unit') and hasattr(b, 'unit')):

            # TODO if only one has units, error is ugly

            a = a.to_value(self._xunit)
            b = b.to_value(self._xunit)

        else:
            if self._xunit:
                mssg = f"a and b must have units matching x ({self._xunit})"
                raise ValueError(mssg)

        res = super().integral(a, b)

        # if either unit is missing, just ignore final units completely
        if self._xunit and self._yunit:
            res <<= (self._xunit * self._yunit)

        return res

    def roots(self):

        res = super().roots()

        if self._xunit:
            res <<= self._xunit

        return res

    def derivative(self, n=1):
        from scipy.interpolate import fitpack

        tck = fitpack.splder(self._eval_args, n)

        # if self.ext is 'const', derivative.ext will be 'zeros'
        ext = 1 if self.ext == 3 else self.ext

        # if either unit is missing, just ignore final units completely
        if self._xunit and self._yunit:
            der_unit = self._yunit * self._xunit**-n
        else:
            warnings.warn("Missing units, resulting spline will have no y-unit")
            der_unit = None

        return QuantitySpline._from_tck(
            tck, x_unit=self._xunit, y_unit=der_unit, ext=ext,
        )
