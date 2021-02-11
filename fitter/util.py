import pathlib
from importlib import resources

import scipy.stats
import numpy as np
import astropy.units as u
from astropy.constants import c

# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------


# Simple gaussian implementation
def gaussian(x, sigma, mu):
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * (((x - mu) / sigma) ** 2))
    return norm * exponent


def RV_transform(domain, f_X, h, h_prime):
    '''transformation of a random variable over a function g=h^-1'''
    f_Y = f_X(h(domain)) * h_prime(domain)
    return np.nan_to_num(f_Y)


def galactic_pot(lat, lon, D):
    '''b, l, d'''
    import gala.potential as pot

    from astropy.coordinates import SkyCoord

    # Mikly Way Potential
    mw = pot.BovyMWPotential2014()

    # TODO chck that these dont already have units (or maybe require them first)
    # Pulsar position in galactocentric coordinates
    b_pulsar, l_pulsar, D_pulsar = lat * u.deg, lon * u.deg, D * u.kpc

    crd = SkyCoord(b=b_pulsar, l=l_pulsar, distance=D_pulsar, frame='galactic')
    XYZ = crd.galactocentric.cartesian.xyz

    # Sun position in galactocentric coordinates
    b_sun = np.zeros_like(lat) * u.deg
    l_sun = np.zeros_like(lon) * u.deg
    D_sun = np.zeros_like(D) * u.kpc

    # TODO the transformations are kinda slow, and are prob uneccessary here
    sun = SkyCoord(b=b_sun, l=l_sun, distance=D_sun, frame='galactic')
    XYZ_sun = sun.galactocentric.cartesian.xyz

    PdotP = mw.acceleration(XYZ).si / c

    # scalar projection of PdotP along the position vector from pulsar to sun
    LOS = XYZ_sun - XYZ
    # PdotP_LOS = np.dot(PdotP, LOS) / np.linalg.norm(LOS)
    PdotP_LOS = np.einsum('i...,i...->...', PdotP, LOS) / np.linalg.norm(LOS)

    return PdotP_LOS


def pulsar_Pdot_KDE(*, pulsar_db='field_msp.dat', corrected=True):
    '''Return a gaussian kde
    psrcat -db_file psrcat.db -c "p0 p1 p1_i GB GL Dist" -l "p0 < 0.1 &&
        p1 > 0 && p1_i > 0 && ! assoc(GC)" -x > field_msp.dat
    '''
    # Get field pulsars data
    with resources.path('fitter', 'resources') as datadir:
        pulsar_db = pathlib.Path(f"{datadir}/{pulsar_db}")
        cols = (0, 3, 6, 7, 8, 9)
        P, Pdot, Pdot_pm, lat, lon, D = np.genfromtxt(pulsar_db, usecols=cols).T

    # Compute and remove the galactic contribution from the PM corrected Pdot
    # TODO dont use value, make everything else be units
    Pdot_int = Pdot_pm - galactic_pot(lat, lon, D).value

    P = np.log10(P)
    Pdot_int = np.log10(Pdot_int)

    # TODO some Pdot_pm < Pdot_gal; this may or may not be physical, need check
    finite = np.isfinite(Pdot_int)

    # Create Gaussian P-Pdot_int KDE
    return scipy.stats.gaussian_kde(np.vstack([P[finite], Pdot_int[finite]]))


def cluster_list():
    with resources.path('fitter', 'resources') as datadir:
        return [f.stem for f in pathlib.Path(datadir).glob('[!TEST]*.hdf5')]
