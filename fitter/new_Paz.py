import numpy as np

from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt

# TODO vectorize this along pulsar R's as well

def vec_Paz(model, R, mass_bin, *, logged=False):
    """ 
    Computes probability distribution for a range of line of sight
    accelerations at projected R : P(az|R)
    Returns the an array containing the probability distribution.

    `logged` uses a logspace for the acceleration domain, rather than linear
    """

    # Assumes units of az [m/s^2] if model.G == 0.004302, else models units
    # Conversion factor from [pc (km/s)^2/Msun] -> [m/s^2]
    az_fac = 1.0 / 3.0857e10 if (model.G == 0.004302) else 1

    # This flag will be set to false in the case of an interpolation error,
    # we will then ignore that point and not append to the list.
    append = True

    if R > model.rt:
        raise ValueError(f"Pulsar position outside cluster bound ({model.rt})")
        # return

    nz = model.nstep

    # maximum z value at R, based on cluster bound
    zt = np.sqrt(model.rt ** 2 - R ** 2)

    # log-spaced domain of z values, with explicit max value (for f.p. errors)
    z = np.geomspace(model.r[1], zt, nz)
    z[-1] = zt

    # Corresponding radial distances of z
    r = np.sqrt(R ** 2 + z ** 2)

    # Spline for enclosed mass
    spl_Mr = UnivariateSpline(model.r, model.mc, s=0, ext=1)

    # LOS acceleration calculation
    az = model.G * spl_Mr(r) * z / r ** 3

    # convert to [m/s^2]
    az *= az_fac

    # Spline for LOS acceleration
    # 4th order, as k=2 derivatives cannot find roots
    az_spl = UnivariateSpline(z, az, k=4, s=0, ext=1)
    az_der = az_spl.derivative()

    # Location of the maximum acceleration along this los
    zmax = az_der.roots()

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
        zmax = zmax[0]  # Take first entry for the rare cases with multiple peaks
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
    azmax = az_spl(zmax)

    # define the acceleration space domain, based on amax
    # TODO this (especially with logged) requires care to ensure its normalized

    bound = azmax + 5e-9
    if logged:
        az_domain = np.geomspace(5e-11, bound, int(bound / 15e-9 * 150))
    else:
        az_domain = np.linspace(5e-11, bound, int(bound / 15e-9 * 150))

    # All invalid acceleratoin space (outside azmax) will have probability = 0

    Paz_dist = np.zeros_like(az_domain)

    within_max = np.where(az_domain < azmax)

    # TODO look at the old new_Paz to get the comments for this stuff

    z1 = np.maximum(z1_spl(az_domain[within_max]), z[0])

    Paz = rhoz_spl(z1) / abs(az_der(z1))

    outside_azt = np.where(az_domain[within_max] > azt)

    z2 = z2_spl(az_domain[within_max][outside_azt])

    within_bounds = np.where(z2 < zt)

    Paz[outside_azt][within_bounds] += rhoz_spl(z2[within_bounds]) / abs(az_der(z2[within_bounds]))

    Paz[outside_azt][within_bounds] /= rhoz_spl.integral(0., zt)

    Paz_dist[within_max] = Paz

    # Mirror the distributions
    Paz_dist = np.concatenate((np.flip(Paz_dist), Paz_dist))
    az_domain = np.concatenate((np.flip(-az_domain), az_domain))

    return az_domain, Paz_dist
