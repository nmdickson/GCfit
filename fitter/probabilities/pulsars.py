import scipy.stats
import numpy as np
import astropy.units as u
from astropy.constants import c
from scipy.interpolate import UnivariateSpline

import pathlib
from importlib import resources


__all__ = [
    'cluster_component',
    'galactic_component',
    'shklovskii_component',
    'field_Pdot_KDE',
]


# --------------------------------------------------------------------------
# Pulsar acceleration (Pdot / P) components
# --------------------------------------------------------------------------


def cluster_component(model, R, mass_bin, *, logspaced=False):
    """
    Computes probability distribution for a range of line of sight
    accelerations at projected R : P(az|R)
    Returns the an array containing the probability distribution and the Pdot/P
    domain of said distribution

    `logspaced` uses a logspace for the acceleration domain, rather than linear

    """

    if R >= model.rt:
        raise ValueError(f"Pulsar position outside cluster bound ({model.rt})")

    nz = model.nstep

    # maximum z value at R, based on cluster bound
    zt = np.sqrt(model.rt ** 2 - R ** 2)

    # log-spaced domain of z values, with explicit max value (for f.p. errors)
    z = np.geomspace(model.r[1], zt, nz)
    z[-1] = zt

    # Corresponding radial distances of z
    r = np.sqrt(R ** 2 + z ** 2)

    # Spline for enclosed mass
    Mr = UnivariateSpline(model.r, model.mc, s=0, ext=1)(r) * model.mc.unit

    # LOS acceleration calculation
    az = model.G * Mr * z / r ** 3

    # convert to [m/s^2]
    az = az.to(u.Unit('m/s^2'))

    # Spline for LOS acceleration
    # 4th order, as k=2 derivatives cannot find roots
    az_spl = UnivariateSpline(z, az, k=4, s=0, ext=1)
    az_der = az_spl.derivative()

    # Location of the maximum acceleration along this los
    zmax = az_der.roots() * z.unit

    # Acceleration at zt
    azt = az[-1]

    # Setup spline for the density, depending on mass bin
    if mass_bin == 0 and model.nmbin == 1:
        rho = model.rho
    else:
        rho = model.rhoj[mass_bin]

    # Project the density along z
    rho_spl = UnivariateSpline(model.r, rho, ext=1, s=0)
    rhoz_spl = UnivariateSpline(z, rho_spl(r), ext=1, s=0)

    # Now compute P(a_z|R)
    # There are 2 possibilities depending on R:
    #  (1) the maximum acceleration occurs within the cluster boundary, or
    #  (2) max(a_z) = a_z,t (this happens when R ~ r_t)

    nr, k = nz, 3  # bit of experimenting

    # Option (1): zmax < max(z)
    if len(zmax) > 0:
        # Take first entry for the rare cases with multiple peaks
        zmax = zmax[0]
        # Set up 2 splines for the inverse z(a_z) for z < zmax and z > zmax
        z1 = np.linspace(z[0], zmax, nr)

        # What we want is a continuation of the acceleration space past the zmax point
        # so that we come to the z2 point for which we can calculate a separate probability.
        # The z2 point needs to be calculated separately because the calculation depends on
        # the density, which is diffrent at each z point.
        # Unclear why original implementation doesn't always work, seems perfectly fine.
        # The reason for the reverse is that it needs to be strictly increasing for the spline

        z2 = np.linspace(zmax, z[-1], nr)[::-1]

        z1_spl = UnivariateSpline(az_spl(z1), z1, k=k, s=0, ext=1)

        # Original implementation
        # changing the spline here doesn't fix the z2 interpolation error.
        z2_spl = UnivariateSpline(az_spl(z2), z2, k=k, s=0, ext=1)

    # Option 2: zmax = max(z)
    else:
        zmax = z[-1]
        z1 = np.linspace(z[0], zmax, nr)
        z1_spl = UnivariateSpline(az_spl(z1), z1, k=k, s=0, ext=1)

    # Value of the maximum acceleration, at the chosen root
    azmax = az_spl(zmax) * az.unit

    # define the acceleration space domain, based on amax

    # TODO this (especially w/ logspaced) requires care to ensure its normalized
    bound = azmax + (5e-9 * az.unit)
    num = int(bound.value / 15e-9 * 150)
    if logspaced:
        az_domain = np.r_[0., np.geomspace(5e-11 * az.unit, bound, num - 1)]
    else:
        az_domain = np.linspace(0. * az.unit, bound, num)

    # All invalid acceleratoin space (outside azmax) will have probability = 0

    Paz_dist = np.zeros(az_domain.shape) * u.dimensionless_unscaled

    within_max = np.where(az_domain < azmax)

    # TODO look at the old new_Paz to get the comments for this stuff

    z1 = np.maximum(z1_spl(az_domain[within_max]) * z1.unit, z[0])

    Paz = rhoz_spl(z1) / abs(az_der(z1))

    outside_azt = np.where(az_domain[within_max] > azt)

    # Only use z2 if any outside_azt values exist (ensures z2_spl exists)
    if outside_azt[0].size > 0:

        z2 = z2_spl(az_domain[within_max][outside_azt]) * z2.unit

        within_bounds = np.where(z2 < zt)

        Paz[outside_azt][within_bounds] += (rhoz_spl(z2[within_bounds])
                                            / abs(az_der(z2[within_bounds])))

        Paz[outside_azt][within_bounds] /= rhoz_spl.integral(0., zt.value)

    Paz_dist[within_max] = Paz

    # Mirror the distributions
    Paz_dist = np.concatenate((np.flip(Paz_dist[1:]), Paz_dist))
    az_domain = np.concatenate((np.flip(-az_domain[1:]), az_domain))

    # Change the acceleration domain to a Pdot / P domain
    PdotP_domain = az_domain / c

    return PdotP_domain, Paz_dist


def galactic_component(lat, lon, D):
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


def shklovskii_component(pm, D):
    '''
    pm in angular speed (mas/yr)
    '''
    pm = pm.to("1/s", u.dimensionless_angles())

    D = D.to('m')

    PdotP_pm = pm**2 * D / c

    return (PdotP_pm).decompose()


# --------------------------------------------------------------------------
# Distribution of field pulsar parameters
# --------------------------------------------------------------------------


def field_Pdot_KDE(*, pulsar_db='field_msp.dat', corrected=True):
    '''Return a gaussian kde
    psrcat -db_file psrcat.db -c "p0 p1 p1_i GB GL Dist" -l "p0 < 0.1 &&
        p1 > 0 && p1_i > 0 && ! assoc(GC)" -x > field_msp.dat
    '''
    # Get field pulsars data
    with resources.path('fitter', 'resources') as datadir:
        pulsar_db = pathlib.Path(f"{datadir}/{pulsar_db}")
        cols = (0, 3, 6, 7, 8, 9)
        P, Pdot, Pdot_pm, lat, lon, D = np.genfromtxt(pulsar_db, usecols=cols).T

    lat <<= u.deg
    lon <<= u.deg
    D <<= u.kpc

    # Compute and remove the galactic contribution from the PM corrected Pdot
    Pdot_int = Pdot_pm - galactic_component(*(lat, lon), D).value

    P = np.log10(P)
    Pdot_int = np.log10(Pdot_int)

    # TODO some Pdot_pm < Pdot_gal; this may or may not be physical, need check
    finite = np.isfinite(Pdot_int)

    # Create Gaussian P-Pdot_int KDE
    return scipy.stats.gaussian_kde(np.vstack([P[finite], Pdot_int[finite]]))