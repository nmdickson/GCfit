from ..util import QuantitySpline, gaussian

import scipy.stats
import scipy as sp
import numpy as np
import astropy.units as u
from astropy.constants import c

import pathlib
from importlib import resources
import logging


__all__ = [
    'cluster_component',
    'galactic_component',
    'shklovskii_component',
    'field_Pdot_KDE',
]


# --------------------------------------------------------------------------
# Pulsar acceleration (Pdot / P) components
# --------------------------------------------------------------------------


# Helpers for DM stuff, TODO relocate to utils or wherever


# gaussian error propagation for division
def div_error(a, a_err, b, b_err):
    f = a / b
    return abs(f) * np.sqrt((a_err / a) ** 2 + (b_err / b) ** 2)


# Get the LOS position and uncertainty based on the DM
def los_dm(dm, dm_err, DM_mdata):

    # TODO I guess these don't have units attached already
    ng = DM_mdata["ng"] * u.Unit("1/cm3")
    delta_ng = DM_mdata["Δng"] * u.Unit("1/cm3")
    DMc = DM_mdata["DMc"] * u.Unit("pc/cm3")
    delta_DMc = DM_mdata["ΔDMc"] * u.Unit('pc/cm3')

    los = (dm - DMc) / ng
    err = div_error(
        a=(dm - DMc),
        a_err=(dm_err + delta_DMc),
        b=ng,
        b_err=delta_ng,
    )
    return los, err


def cluster_component(model, R, mass_bin, DM=None, DM_mdata=None, *, eps=1e-3):
    """
    Computes probability distribution for a range of line of sight
    accelerations at projected R : P(az|R)
    Returns the an array containing the probability distribution and the Pdot/P
    domain of said distribution

    """

    R = R.to(model.rt.unit)





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
    Mr = QuantitySpline(model.r, model.mc, s=0, ext=1)(r)

    # LOS acceleration calculation
    az = model.G * Mr * z / r ** 3

    # convert to [m/s^2]
    az = az.to(u.Unit('m/s^2'))

    # Spline for LOS acceleration
    # 4th order, as k=2 derivatives cannot find roots
    az_spl = QuantitySpline(z, az, k=4, s=0, ext=1)
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
    rho_spl = QuantitySpline(model.r, rho, ext=1, s=0)
    rhoz_spl = QuantitySpline(z, rho_spl(r), ext=1, s=0)

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

        # What we want is a continuation of the acceleration space past the zmax
        # point so that we come to the z2 point for which we can calculate a
        # separate probability. The z2 point needs to be calculated separately
        # because the calculation depends on the density, which is diffrent at
        # each z point. Unclear why original implementation doesn't always work,
        # seems perfectly fine. The reason for the reverse is that it needs to
        # be strictly increasing for the spline

        z2 = np.linspace(zmax, z[-1], nr)[::-1]

        z1_spl = QuantitySpline(az_spl(z1), z1, k=k, s=0, ext=1)

        # Original implementation
        # changing the spline here doesn't fix the z2 interpolation error.
        z2_spl = QuantitySpline(az_spl(z2), z2, k=k, s=0, ext=1)

    # Option 2: zmax = max(z)
    else:
        zmax = z[-1]
        z1 = np.linspace(z[0], zmax, nr)
        z1_spl = QuantitySpline(az_spl(z1), z1, k=k, s=0, ext=1)

    # Value of the maximum acceleration, at the chosen root
    azmax = az_spl(zmax)

    # Old version here for future reference
    # increment density by 2 order of magn. smaller than azmax
    # Δa = 10**(np.floor(np.log10(azmax.value)) - 2)

    # define the acceleration space domain, based on amax and Δa
    # az_domain = np.arange(0.0, azmax.value + Δa, Δa) << azmax.unit

    # Define the acceleration domain, using 2*nr points


    # TODO: find a good spacing for the DM method
    if DM is None:
        az_domain = np.linspace(0.0, azmax.value, 2 * nr) << azmax.unit
    else:
        az_domain = np.linspace(0.0, azmax.value, 50 * nr) << azmax.unit

    Δa = np.diff(az_domain)[1]


    # TODO look at the old new_Paz to get the comments for this stuff

    z1 = np.maximum(z1_spl(az_domain), z[0])

    # TODO fix all the DM stuff and then clean it up
    if DM is None:
        Paz_dist = (rhoz_spl(z1) /
                    abs(az_der(z1))).value * u.dimensionless_unscaled

        outside_azt = az_domain > azt

        # Only use z2 if any outside_azt values exist (ensures z2_spl exists)
        if np.any(outside_azt):

            z2 = z2_spl(az_domain)

            within_bounds = outside_azt & (z2 < zt)

            Paz_dist[within_bounds] += (rhoz_spl(z2[within_bounds])
                                        / abs(az_der(z2[within_bounds]))).value

        Paz_dist /= rhoz_spl.integral(0. << z.unit, zt).value

    else:

        # need to look at full cluster now that it's no longer symmetric
        az_domain = np.concatenate((np.flip(-az_domain[1:]), az_domain))
        # also need the signs to pick which side of the cluster the pulsar is on
        az_signs = np.sign(az_domain)

        # make sure we get the DM data
        if DM is None:
            raise ValueError(
                "Pulsar DM data required to use DM based likelihood."
            )

        # make sure we get the cluster DM mdata too
        if DM_mdata is None:
            raise ValueError(
                "Cluster DM data is required to use DM based likelihood."
            )

        # unpack DM data
        DM, sigma_DM = DM

        # get los pos, err using cluster DM data
        DM_los, DM_los_err = los_dm(DM, sigma_DM, DM_mdata)

        # TODO: Not 100% sure that all pulsars are within the rh for all
        # clusters so for now I'll put this in and when this is working properly
        # we can see if we have any problems.

        if DM_los.to("pc") > model.rh.to("pc"):
            logging.ERROR("Pulsar LOS position outside of rh.")

        z_domain = np.linspace(-model.rh, model.rh, len(az_domain))

        # set up the LOS spline for the DM based Paz
        DM_gaussian = gaussian(x=z_domain, mu=DM_los, sigma=DM_los_err)
        DM_los_spl = sp.interpolate.UnivariateSpline(
            x=z_domain, y=DM_gaussian, s=0, k=3, ext=1
        )

        # I think we still only want positive values for the other splines
        az_domain = np.abs(az_domain)

        z1 = np.maximum(z1_spl(az_domain), z[0])

        # Here we add the signs back in only for the DM splines
        Paz_dist = (DM_los_spl(z1 * -1 * az_signs) /
                    abs(az_der(z1))).value * u.dimensionless_unscaled

        outside_azt = az_domain > azt

        # Only use z2 if any outside_azt values exist (ensures z2_spl exists)
        if np.any(outside_azt):

            z2 = z2_spl(az_domain)

            within_bounds = outside_azt & (z2 < zt)

            # Here we add the signs back in only for the DM splines
            Paz_dist[within_bounds] += (DM_los_spl(z2[within_bounds] *
                                        -1 * (az_signs)[within_bounds])
                                        / abs(az_der(z2[within_bounds]))).value


    # Ensure Paz is normalized
    # NOTE: this version requires more than 2*nr steps, do more testing

    if DM is None:
        # for density based Paz, distributions are symmetric
        target_norm = 2.0

        norm = 0.0
        for ind, P_b in enumerate(Paz_dist[1:], 1):
            P_a = Paz_dist[ind - 1]

            # Integrate using trapezoid rule cumulatively
            norm += (0.5 * Δa * (P_a + P_b)).value

            # NOTE: Norm target here needs to be 1.0 not 0.5 in order to fully
            # sample the distribution, this means we need to divide by 2
            # somewhere along the way. If converges, cut domain at this index
            if abs(1.0 - norm) < eps:
                break

            # If passes normalization, backup a step to cut domain as close
            # as possible
            elif norm > 1.0:
                ind -= 1
                break

        else:
            # This is the case where our probability distribution doesn't
            # integrate to one, just don't cut anything off, log the
            # normalization and manually normalize it.
            logging.warning("Probability distribution failed to integrate "
                            f"to 1.0, area: {norm:.6f}")

            # Manual normalization
            Paz_dist /= norm

    # For DM we need to hand the asymmetric Pz distributions
    else:
        target_norm = 1.0
        eps = 1e-5
        norm = 0.0
        # get the midpoint of the Paz dist
        mid = len(Paz_dist) // 2
        for ind in range(mid):

            # positive side
            P_a = Paz_dist[mid + ind]
            P_b = Paz_dist[mid + ind + 1]

            # negative side
            P_c = Paz_dist[mid - ind]
            P_d = Paz_dist[mid - ind - 1]

            # Integrate using trapezoid rule cumulatively
            norm += (0.5 * Δa * (P_a + P_b)).value
            norm += (0.5 * Δa * (P_c + P_d)).value

            # NOTE for the DM Paz dist, the entire distribution integrates to
            # 1.0 unlike the density based Paz
            if abs(target_norm - norm) <= eps:
                break

            # If passes normalization, backup a step to cut domain as close
            # as possible
            elif norm > target_norm:
                ind -= 1
                break

        else:
            # This is the case where our probability distribution doesn't
            # integrate to one, just don't cut anything off, log the
            # normalization and manually normalize it.
            # import matplotlib.pyplot as plt
            # area = sp.integrate.simpson(x=az_domain, y=Paz_dist)
            # plt.plot(az_domain, Paz_dist, label=area)
            # plt.legend()
            # plt.show()
            logging.warning("Probability distribution failed to integrate "
                            f"to 1.0, area: {norm:.6f}")


            # Manual normalization
            # TODO see how often this is happening with DM, adjust the a-space?
            Paz_dist /= norm


    if DM is None:
        # Set the rest to zero
        Paz_dist[ind + 1:] = 0

        # Mirror the distributions
        Paz_dist = np.concatenate((np.flip(Paz_dist[1:]), Paz_dist))
        az_domain = np.concatenate((np.flip(-az_domain[1:]), az_domain))

    else:
        # Set the rest to zero
        Paz_dist[:mid - ind] = 0
        # Off by one?
        Paz_dist[mid + ind + 1:] = 0




    # if DM is None:
    #     Paz_spl = UnivariateSpline(x=az_domain, y=Paz_dist, k=3, s=0, ext=1)
    # else:
    #     # add the signs back in to the domain
    #     Paz_spl = UnivariateSpline(
    #         x=az_signs * az_domain, y=Paz_dist, k=3, s=0, ext=1
    #     )
    # area = Paz_spl.integral(-np.inf, np.inf)
    # if DM is None:
    #     area /= 2
    # print(f"step: {ind}/{len(az_domain)//2}, norm: {area}")


    # Normalize the Paz dist, before this step the area should be ~2 because
    # each side of the dist needs to be cutoff at an area of 1.0.
    # (Only for density method)
    if DM is None:
        Paz_dist /= 2



    # Change the acceleration domain to a Pdot / P domain
    PdotP_domain = az_domain / c

    # put the signs back in for the DM method
    if DM is not None:
        PdotP_domain *= az_signs

    return PdotP_domain, Paz_dist


def galactic_component(lat, lon, D):
    '''b, l, d'''
    import gala.potential as pot

    from astropy.coordinates import SkyCoord

    # Milky Way Potential
    mw = pot.BovyMWPotential2014()

    # Pulsar position in galactocentric coordinates
    crd = SkyCoord(b=lat, l=lon, distance=D, frame='galactic')
    XYZ = crd.galactocentric.cartesian.xyz

    # Sun position in galactocentric coordinates
    b_sun = np.zeros_like(lat)
    l_sun = np.zeros_like(lon)
    D_sun = np.zeros_like(D)

    # TODO the transformations are kinda slow, and are prob unnecessary here
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
