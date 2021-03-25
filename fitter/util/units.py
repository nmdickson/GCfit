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


class QuantitySpline(scipy.interpolate.UnivariateSpline):
    '''Subclass of SciPy's UnivariateSpline, supporting AstroPy `Quantity`s'''

    def __init__(self, x, y, w=None, bbox=[None] * 2, k=3, s=0,
                 ext=1, check_finite=False):

        self._xunit = x.unit
        self._yunit = y.unit

        # Convert boundary box, if needed
        if bbox[0] is not None:
            bbox[0] = bbox[0].to_value(self._xunit)

        if bbox[1] is not None:
            bbox[1] = bbox[1].to_value(self._xunit)

        super().__init__(x, y, w=w, bbox=bbox, k=k, s=s,
                         ext=ext, check_finite=check_finite)

    @classmethod
    def _from_tck(cls, tck, x_unit, y_unit, ext=0):

        obj = super()._from_tck(tck, ext=ext)

        obj._xunit = x_unit
        obj._yunit = y_unit

        return obj

    def __call__(self, x, nu=0, ext=None):
        x = x.to_value(self._xunit)
        return super().__call__(x, nu=nu, ext=ext) << self._yunit

    def integral(self, a, b):
        a = a.to_value(self._xunit)
        b = b.to_value(self._xunit)

        integ_unit = self._xunit * self._yunit

        return super().integral(a, b) << integ_unit

    def roots(self):
        return super().roots() << self._xunit

    def derivative(self, n=1):
        from scipy.interpolate import fitpack

        tck = fitpack.splder(self._eval_args, n)
        # if self.ext is 'const', derivative.ext will be 'zeros'
        ext = 1 if self.ext == 3 else self.ext

        der_unit = self._yunit * self._xunit**-n

        return QuantitySpline._from_tck(
            tck, x_unit=self._xunit, y_unit=der_unit, ext=ext,
        )
