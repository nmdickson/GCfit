import warnings

import scipy
import numpy as np
import astropy.units as u
from astropy.units.equivalencies import Equivalency


__all__ = ['angular_width', 'QuantitySpline', 'q2pv']


def angular_width(D):
    '''AstroPy units conversion equivalency for angular to linear distances

    See: https://docs.astropy.org/en/stable/units/equivalencies.html
    '''

    D = D.to(u.pc)

    # Angular width
    def pc_to_rad(r):
        '''Parsecs to radians.'''
        return 2 * np.arctan(r / (2 * D.value))

    def rad_to_pc(θ):
        '''Radians to parsecs.'''
        return np.tan(θ / 2) * (2 * D.value)

    # Angular area
    def pcsq_to_radsq(r):
        '''Parsecs squared to radians squared.'''
        return (1. / rad_to_pc(1)**2) * r

    def radsq_to_radsq(θ):
        '''Radians squared to parsec squared.'''
        return (1. / pc_to_rad(1)**2) * θ

    # Inverse Angular area
    def inv_pcsq_to_radsq(r):
        '''Parsecs squared reciprocal to radians squared reciprocal.'''
        return (rad_to_pc(1)**2) * r

    def inv_radsq_to_radsq(θ):
        '''Radians squared reciprocal to parsecs squared reciprocal.'''
        return (pc_to_rad(1)**2) * θ

    # Angular Speed
    def kms_to_asyr(vt):
        '''Kilometres per second to arcseconds per year.'''
        return vt / (4.74 * D.value)

    def asyr_to_kms(μ):
        '''Arcseconds per year to kilometres per second.'''
        return 4.74 * D.value * μ

    return Equivalency([
        (u.pc, u.rad, pc_to_rad, rad_to_pc),
        (u.pc**2, u.rad**2, pcsq_to_radsq, radsq_to_radsq),
        (u.pc**-2, u.rad**-2, inv_pcsq_to_radsq, inv_radsq_to_radsq),
        ((u.km / u.s), (u.arcsec / u.yr), kms_to_asyr, asyr_to_kms)
    ], 'angular_width', {"D": D})


# TODO this really needs unittest, the unitless stuff made it complicated
class QuantitySpline(scipy.interpolate.UnivariateSpline):
    '''Subclass of SciPy's UnivariateSpline, supporting AstroPy ``Quantity``

    1-D smoothing spline fit to a given set of data points, with support for
    units through using ``astropy.Quantity`` arrays as input.

    All functions will call their corresponding scipy native functions, and
    simply add the relevant units to their outputs.

    See Also
    --------
    scipy.interpolate.UnivariateSpline
    '''

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

        # if no _xunit, but x has a unit, assume it matches
        if hasattr(x, 'unit'):
            x = x.to_value(self._xunit)

        # call the original UnivariateSpline __call__
        res = super().__call__(x, nu=nu, ext=ext)

        # if has yunit, apply that to the results
        if self._yunit:
            res <<= self._yunit

        return res

    def integral(self, a, b):

        # if no xunit, but a,b does have units, assumes they match
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

        # if has xunit, apply that to the results
        if self._xunit:
            res <<= self._xunit

        return res

    def derivative(self, n=1):
        from scipy.interpolate import splder

        tck = splder(self._eval_args, n)

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


def q2pv(p, v):
    '''Form two array[3] of positions and velocities into a pv array for erfa'''
    import erfa

    pvunit = u.StructuredUnit((p.unit, v.unit))
    pv = np.rec.fromarrays([p, v], erfa.dt_pv) << pvunit

    return pv
