from ..util import QuantitySpline, gaussian, div_error, trim_peaks, find_intersections

import scipy.stats
import scipy as sp
import numpy as np
import astropy.units as u
from astropy.constants import c

import pathlib
import logging


__all__ = [
    'cluster_component',
    'galactic_component',
    'shklovskii_component',
    'field_Pdot_KDE',
]


# --------------------------------------------------------------------------
# LOS position of pulsars from DM
# --------------------------------------------------------------------------


def los_dm(dm, dm_err, DM_mdata):
    """
    Compute line-of-sight position and uncertainty based on pulsar DM data.

    Parameters
    ----------
    dm : float
        Dispersion measure of pulsar. Must be in pc/cm^3.
    dm_err : float
        Error associated with DM measurement. Must be in pc/cm^3.
    DM_mdata : dict
        Cluster-specific DM data, includes mean cluster-DM as well as
        cluster gas-density and uncertainty. Cluster-DM must be in pc/cm^3
        and gas density in cm^-3.

    Returns
    -------
    position : tuple
        A tuple of floats corresponding to the mean LOS position and
        its uncertainty.

    Notes
    -----
    Assumes a uniform gas density within the cluster.

    """

    ng = DM_mdata["ng"] << u.Unit("1/cm3")
    delta_ng = DM_mdata["Δng"] << u.Unit("1/cm3")

    DMc = DM_mdata["DMc"] << u.Unit("pc/cm3")
    delta_DMc = DM_mdata["ΔDMc"] << u.Unit('pc/cm3')

    los = (dm - DMc) / ng
    err = div_error(
        a=(dm - DMc),
        a_err=(dm_err + delta_DMc),
        b=ng,
        b_err=delta_ng,
    )
    return los, err

# --------------------------------------------------------------------------
# Pulsar acceleration (Pdot / P) components
# --------------------------------------------------------------------------


def cluster_component(model, R, mass_bin, DM=None, ΔDM=None, DM_mdata=None, *, eps=1e-3):
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

    # Setup spline for the density, depending on mass bin
    if mass_bin == 0 and model.nmbin == 1:
        rho = model.rho
    else:
        rho = model.rhoj[mass_bin]

    # Project the density along z
    rho_spl = QuantitySpline(model.r, rho, ext=1, s=0)
    rhoz_spl = QuantitySpline(z, rho_spl(r), ext=1, s=0)

    # Now compute P(a_z|R)

    nr = nz  # bit of experimenting

    # There are 2 possibilities depending on R:
    # (1) the maximum acceleration occurs within the cluster boundary, or
    # (2) max(a_z) = a_z,t (this happens when R ~ r_t)
    if len(zmax) > 0:
        azmax = az_spl(zmax[0])
    else:
        azmax = az_spl(z[-1])

    # Old version here for future reference
    # increment density by 2 order of magnitude smaller than azmax
    # Δa = 10**(np.floor(np.log10(azmax.value)) - 2)
    # define the acceleration space domain, based on amax and Δa
    # az_domain = np.arange(0.0, azmax.value + Δa, Δa) << azmax.unit

    # Define the acceleration domain, using 2*nr points (for density method)
    # for the DM method, just loading up on points and then trimming the dist
    # seems to be the easiest way to do things
    if DM is None:
        az_domain = np.linspace(0.0, azmax.value, 2 * nr) << azmax.unit
    else:
        az_domain = np.linspace(0.0, azmax.value, 50 * nr) << azmax.unit

    Δa = np.diff(az_domain)[1]

    # TODO look at the old new_Paz to get the comments for this stuff

    if DM is None:

        Paz_dist = np.zeros_like(az_domain.value)
        for i, a in enumerate(az_domain):

            z_values = find_intersections(az, z, a)

            Paz_dist[i] = np.sum(
                rhoz_spl(z_values).value / np.abs(az_der(z_values).value), axis=0
            )

        Paz_dist <<= u.dimensionless_unscaled
        # This new method gives a zero in at the start of the distribution
        # which I expect might do bad things to convolution, so I'm just
        # going to use the second value
        Paz_dist[0] = Paz_dist[1]

        # Normalise the distribution
        Paz_dist /= rhoz_spl.integral(0.0 << z.unit, zt).value

    else:

        # need to look at full cluster now that it's no longer symmetric
        az_domain = np.concatenate((np.flip(-az_domain[1:]), az_domain))

        # also need the signs to pick which side of the cluster the pulsar is on
        # negative az means a positive z position, opposite for positive
        az_signs = np.sign(az_domain)

        # make sure we get the cluster DM mdata too
        if DM_mdata is None:
            mssg = "Cluster DM data is required to use DM based likelihood."
            raise ValueError(mssg)

        # get los pos, err using cluster DM data
        DM_los, DM_los_err = los_dm(DM, ΔDM, DM_mdata)
        
        # Pulsars should alway be within the half-light radius, if not the
        # spline will just return probability of zero anyway so we should be
        # fine. We should log here anyway so that we can make sure this isn't
        # happening often.
        if DM_los.to("pc") > model.rh.to("pc"):
            logging.warning("Pulsar DM-based LOS position outside of rh.")

        z_domain = np.linspace(-model.rh, model.rh, len(az_domain))

        # set up the LOS spline for the DM based Paz
        DM_gaussian = gaussian(x=z_domain, mu=DM_los, sigma=DM_los_err)
        DM_los_spl = sp.interpolate.UnivariateSpline(
            x=z_domain, y=DM_gaussian, s=0, k=3, ext=1
        )

        # We still only want positive values for the other splines
        az_domain = np.abs(az_domain)

        Paz_dist = np.zeros_like(az_domain.value)
        for i, a in enumerate(az_domain):

            z_values = find_intersections(az, z, a)

            Paz_dist[i] = np.sum(
                (DM_los_spl(-az_signs[i] * z_values) / abs(az_der(z_values))).value,
                axis=0,
            )

        Paz_dist <<= u.dimensionless_unscaled

        # This new method gives a zero in at the start of the distribution
        # which I expect might do bad things to convolution, so I'm just
        # going to interpolate it, this is a bit hacky but I don't see a cleaner
        # way to do it, and this gives a nice smooth distribution
        if np.any(Paz_dist == 0.0):
            # get the index
            arg_zero = np.argmin(Paz_dist)
            # dont overflow the array
            if 0 < arg_zero <len (Paz_dist)-1:
                # interpolate between the two neighbouring points
                Paz_dist[arg_zero] = (
                    Paz_dist[arg_zero - 1] + Paz_dist[arg_zero + 1]
                ) / 2.0

    # Ensure Paz is normalized (slightly different for density vs DM methods)

    if DM is None:
        # for density based Paz, distributions are symmetric
        target_norm = 2.0

        norm = 0.0
        for ind, P_b in enumerate(Paz_dist[1:], 1):
            P_a = Paz_dist[ind - 1]

            # Integrate using trapezoid rule cumulatively
            norm += (0.5 * Δa * (P_a + P_b)).value

            # NOTE: Norm target here needs to be 1 not 0.5 in order to fully
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
            # integrate to one.

            # If the area is way less than 1, we should just throw an exception
            if norm < 0.9:
                raise ValueError("Paz failed to integrate to 1.0, too small to"
                                f"continue. Area: {norm:.6f}")

            # If the area is close to 1, we should just log a warning and
            # manually normalize it
            logging.warning(
                "Probability distribution failed to integrate "
                f"to 1.0, area: {norm:.6f}"
            )

            # Manual normalization
            Paz_dist /= norm

    # For DM we need to handle the asymmetric Pz distributions
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

            # If the area is way less than 1, we should just throw an exception
            if norm < 0.9:
                raise ValueError("Paz failed to integrate to 1.0, too small to"
                                f" continue. Area: {norm:.6f}")

            logging.warning(
                "Probability distribution failed to integrate "
                f"to 1.0, area: {norm:.6f}"
            )

            # Manual normalization
            Paz_dist /= norm

    if DM is None:
        # Set the rest to zero
        Paz_dist[ind + 1:] = 0

        # Mirror the distributions
        Paz_dist = np.concatenate((np.flip(Paz_dist[1:]), Paz_dist))
        az_domain = np.concatenate((np.flip(-az_domain[1:]), az_domain))

    else:
        # Set the rest (both sides) to zero
        Paz_dist[:mid - ind] = 0
        Paz_dist[mid + ind + 1:] = 0

    # Normalize the Paz dist, before this step the area should be ~2 because
    # each side of the dist needs to be cutoff at an area of 1.0.
    # (Only for density method)
    if DM is None:
        Paz_dist /= 2

    # Change the acceleration domain to a Pdot / P domain
    # TODO (Peter): Reminder here to double check with Nolan/Vincent that this
    # final distribution should be normalized to 1.0, even after the conversion
    # from az to Pdot/P.
    PdotP_domain = az_domain / c
    P_PdotP_dist = Paz_dist * c.value

    # put the signs back in for the DM method
    if DM is not None:
        PdotP_domain *= az_signs

    # Trim the peaks from numerical instability around azmax
    Paz_dist = trim_peaks(PdotP_domain, P_PdotP_dist)

    return PdotP_domain, P_PdotP_dist


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
    from ..util.data import _open_resources

    # Get field pulsars data

    with _open_resources() as datadir:
        pulsar_db = pathlib.Path(f"{datadir}/{pulsar_db}")
        cols = (0, 3, 6, 7, 8, 9)
        P, Pdot, Pdot_pm, lat, lon, D = np.genfromtxt(pulsar_db, usecols=cols).T

    lat <<= u.deg
    lon <<= u.deg
    D <<= u.kpc

    # Compute and remove the galactic contribution from the PM corrected Pdot
    Pdot_int = Pdot_pm - galactic_component(*(lat, lon), D).value

    P = np.log10(P)
    # TODO would be nice to get rid of the warning that happens here everytime
    Pdot_int = np.log10(Pdot_int)

    # TODO some Pdot_pm < Pdot_gal; this may or may not be physical, need check
    finite = np.isfinite(Pdot_int)

    # Create Gaussian P-Pdot_int KDE
    return scipy.stats.gaussian_kde(np.vstack([P[finite], Pdot_int[finite]]))
