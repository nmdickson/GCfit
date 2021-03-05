import pathlib
from importlib import resources

import scipy.stats
import numpy as np
import astropy.units as u
from astropy.constants import c


__all__ = ['gaussian', 'RV_transform', 'galactic_pot', 'pulsar_Pdot_KDE']


# --------------------------------------------------------------------------
# Probability Distribution Helper Functions
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


# --------------------------------------------------------------------------
# Pulsar Period Component Functions
# --------------------------------------------------------------------------
# TODO should maybe be in probabilities with other pulsar code


def galactic_pot(lat, lon, D):
    '''b, l, d'''
    import gala.potential as pot

    from astropy.coordinates import SkyCoord

    # Mikly Way Potential
    mw = pot.BovyMWPotential2014()

    # Pulsar position in galactocentric coordinates
    crd = SkyCoord(b=lat, l=lon, distance=D, frame='galactic')
    XYZ = crd.galactocentric.cartesian.xyz

    # Sun position in galactocentric coordinates
    b_sun = np.zeros_like(lat)
    l_sun = np.zeros_like(lon)
    D_sun = np.zeros_like(D)

    # TODO the transformations are kinda slow, and are prob uneccessary here
    sun = SkyCoord(b=b_sun, l=l_sun, distance=D_sun, frame='galactic')
    XYZ_sun = sun.galactocentric.cartesian.xyz

    PdotP = mw.acceleration(XYZ).si / c

    # scalar projection of PdotP along the position vector from pulsar to sun
    LOS = XYZ_sun - XYZ
    # PdotP_LOS = np.dot(PdotP, LOS) / np.linalg.norm(LOS)
    PdotP_LOS = np.einsum('i...,i...->...', PdotP, LOS) / np.linalg.norm(LOS)

    return PdotP_LOS.decompose()


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
    Pdot_int = Pdot_pm - galactic_pot(*(lat, lon) * u.deg, D * u.kpc).value

    P = np.log10(P)
    Pdot_int = np.log10(Pdot_int)

    # TODO some Pdot_pm < Pdot_gal; this may or may not be physical, need check
    finite = np.isfinite(Pdot_int)

    # Create Gaussian P-Pdot_int KDE
    return scipy.stats.gaussian_kde(np.vstack([P[finite], Pdot_int[finite]]))
