from .pulsars import *
from .. import util
from ..core.data import DEFAULT_INITIALS, Model

import numpy as np
import astropy.units as u
import scipy.integrate as integ
import scipy.interpolate as interp

import logging


__all__ = [
    'likelihood_pulsar_spin',
    'likelihood_pulsar_orbital',
    'likelihood_number_density',
    'likelihood_pm_tot',
    'likelihood_pm_ratio',
    'likelihood_pm_T',
    'likelihood_pm_R',
    'likelihood_LOS',
    'likelihood_mass_func',
    'posterior'
]


# TODO standardize which interpolation funciton we're using, 3 are in play rn


# TODO this messes up the error messages sometimes (when .get(model) fails)
def _angular_units(func):
    '''decorator for supporting all angular unit equivalencies,
    for likelihoods
    assumes 'model' will be first arg, or in kwargs
    '''
    import functools

    @functools.wraps(func)
    def angular_units_decorator(*args, **kwargs):

        model = kwargs.get('model') or args[0]

        eqvs = [util.angular_width(model.d)[0],
                util.angular_speed(model.d)[0]]

        with u.set_enabled_equivalencies(eqvs):
            return func(*args, **kwargs)

    return angular_units_decorator


# --------------------------------------------------------------------------
# Component likelihood functions
# --------------------------------------------------------------------------


@_angular_units
def likelihood_pulsar_spin(model, pulsars, Pdot_kde, cluster_μ, coords, *,
                           mass_bin=None):
    '''Compute the log likelihood of pulsar spin period derivatives

    Computes the log likelihood component of a cluster's pulsar's spin
    period derivatives, evaluating the observed pulsar timing solutions
    against the combined probability distributions of the clusters acceleration
    field, the pulsars intrinsic spin-down, the proper-motion contribution and
    the galactic potential.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    pulsars : fitter.core.data.Dataset
        Pulsars dataset used to compute probability distribution and evaluate
        log likelihood

    Pdot_kde : scipy.stats.gaussian_kde
        Gaussian KDE of the galactic field pulsars Pdot-P distribution, from
        `field_Pdot_KDE`. Should be generated beforehand for speed, but if
        None, will generate at runtime.

    cluster_μ : float
        Total cluster proper motion, in mas/yr

    coords : 2-tuple of float
        Cluster Galactic (Latitude, Longitude), in degrees

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else -1

    Returns
    -------
    float
        Log likelihood value

    See Also
    --------
    likelihood_pulsar_orbital : Binary pulsar orbital period likelihood
    fitter.probabilities.pulsars : Module containing all pulsar prob. components

    '''

    # ----------------------------------------------------------------------
    # Get the pulsar P-Pdot_int kde
    # ----------------------------------------------------------------------

    if Pdot_kde is None:
        Pdot_kde = field_Pdot_KDE()

    Pdot_min, Pdot_max = Pdot_kde.dataset[1].min(), Pdot_kde.dataset[1].max()

    # ----------------------------------------------------------------------
    # Get pulsar mass bins
    # ----------------------------------------------------------------------

    if mass_bin is None:
        if 'm' in pulsars.mdata:
            mass_bin = np.where(model.mj == pulsars.mdata['m'] * u.Msun)[0][0]
        else:
            logging.debug("No mass bin provided for pulsars, using -1")
            mass_bin = -1

    # ----------------------------------------------------------------------
    # Iterate over all pulsars
    # ----------------------------------------------------------------------

    N = pulsars['r'].size
    probs = np.zeros(N)

    for i in range(N):

        # ------------------------------------------------------------------
        # Get this pulsars necessary data
        # ------------------------------------------------------------------

        R = pulsars['r'][i].to(u.pc)

        P = pulsars['P'][i].to('s')

        Pdot_meas = pulsars['Pdot_meas'][i]
        ΔPdot_meas = np.abs(pulsars['ΔPdot_meas'][i])

        # ------------------------------------------------------------------
        # Compute the cluster component distribution, from the model
        # ------------------------------------------------------------------

        PdotP_domain, PdotP_c_prob = cluster_component(model, R, mass_bin,
                                                       logspaced=True)
        Pdot_domain = (P * PdotP_domain).decompose()

        # linear to avoid effects around asymptote
        Pdot_c_spl = interp.UnivariateSpline(
            Pdot_domain, PdotP_c_prob, k=1, s=0, ext=1
        )

        # ------------------------------------------------------------------
        # Compute gaussian measurement error distribution
        # ------------------------------------------------------------------

        # TODO if width << Pint width, maybe don't bother with first conv.

        err = util.gaussian(x=Pdot_domain, sigma=ΔPdot_meas, mu=0)

        err_spl = interp.UnivariateSpline(Pdot_domain, err, k=3, s=0, ext=1)

        # ------------------------------------------------------------------
        # Create a slice of the P-Pdot space, along this pulsars P
        # ------------------------------------------------------------------

        lg_P = np.log10(P / P.unit)

        P_grid, Pdot_int_domain = np.mgrid[lg_P:lg_P:1j, Pdot_min:Pdot_max:200j]

        P_grid, Pdot_int_domain = P_grid.ravel(), Pdot_int_domain.ravel()

        # ------------------------------------------------------------------
        # Compute the Pdot_int distribution from the KDE
        # ------------------------------------------------------------------

        Pdot_int_prob = Pdot_kde(np.vstack([P_grid, Pdot_int_domain]))

        Pdot_int_spl = interp.UnivariateSpline(
            Pdot_int_domain, Pdot_int_prob, k=3, s=0, ext=1
        )

        Pdot_int_prob = util.RV_transform(
            domain=10**Pdot_int_domain, f_X=Pdot_int_spl,
            h=np.log10, h_prime=lambda y: (1 / (np.log(10) * y))
        )

        Pdot_int_spl = interp.UnivariateSpline(
            10**Pdot_int_domain, Pdot_int_prob, k=3, s=0, ext=1
        )

        # ------------------------------------------------------------------
        # Set up the equally-spaced linear convolution domain
        # ------------------------------------------------------------------

        # TODO both 5000 and 1e-18 need to be computed dynamically
        #   5000 to be enough steps to sample the gaussian and int peaks
        #   1e-18 to be far enough for the int distribution to go to zero
        #   Both balanced so as to use way too much memory uneccessarily
        #   Must be symmetric, to avoid bound effects

        # mirrored/starting at zero so very small gaussians become the δ-func
        lin_domain = np.linspace(0., 1e-18, 5_000 // 2)
        lin_domain = np.concatenate((np.flip(-lin_domain[1:]), lin_domain))

        # ------------------------------------------------------------------
        # Convolve the different distributions
        # ------------------------------------------------------------------

        conv1 = np.convolve(err_spl(lin_domain), Pdot_c_spl(lin_domain), 'same')

        conv2 = np.convolve(conv1, Pdot_int_spl(lin_domain), 'same')

        # Normalize
        conv2 /= interp.UnivariateSpline(
            lin_domain, conv2, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        # ------------------------------------------------------------------
        # Compute the Shklovskii (proper motion) effect component
        # ------------------------------------------------------------------

        cluster_μ <<= u.Unit("mas/yr")

        PdotP_pm = shklovskii_component(cluster_μ, model.d)

        # ------------------------------------------------------------------
        # Compute the galactic potential component
        # ------------------------------------------------------------------

        PdotP_gal = galactic_component(*(coords * u.deg), D=model.d)

        # ------------------------------------------------------------------
        # Interpolate the likelihood value from the overall distribution
        # ------------------------------------------------------------------

        prob_dist = interp.interp1d(
            (lin_domain / P) + PdotP_pm + PdotP_gal, conv2,
            assume_sorted=True, bounds_error=False, fill_value=0.0
        )

        probs[i] = prob_dist((Pdot_meas / P).decompose())

    # ----------------------------------------------------------------------
    # Multiply all the probabilities and return the total log probability.
    # ----------------------------------------------------------------------

    # TODO should a probs of zero or less return a final 0 or -inf?

    logprobs = np.log(probs)
    logprobs[np.isnan(logprobs)] = np.NINF

    return np.sum(logprobs)


@_angular_units
def likelihood_pulsar_orbital(model, pulsars, cluster_μ, coords, *,
                              mass_bin=None):
    '''Compute the log likelihood of binary pulsar orbital period derivatives

    Computes the log likelihood component of a cluster's binary pulsar's orbital
    period derivatives, evaluating the observed orbital timing solutions
    against the combined probability distributions of the clusters acceleration
    field, the proper-motion contribution and the galactic potential.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    pulsars : fitter.core.data.Dataset
        Pulsars dataset used to compute probability distribution and evaluate
        log likelihood

    cluster_μ : float
        Total cluster proper motion, in mas/yr

    coords : 2-tuple of float
        Cluster Galactic (Latitude, Longitude), in degrees

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else -1

    Returns
    -------
    float
        Log likelihood value

    See Also
    --------
    likelihood_pulsar_spin : Pulsar spin period likelihood
    fitter.probabilities.pulsars : Module containing all pulsar prob. components

    '''

    # ----------------------------------------------------------------------
    # Get pulsar mass bins
    # ----------------------------------------------------------------------

    if mass_bin is None:
        if 'm' in pulsars.mdata:
            mass_bin = np.where(model.mj == pulsars.mdata['m'] * u.Msun)[0][0]
        else:
            logging.debug("No mass bin provided for pulsars, using -1")
            mass_bin = -1

    # ----------------------------------------------------------------------
    # Iterate over all pulsars
    # ----------------------------------------------------------------------

    N = pulsars['r'].size
    probs = np.zeros(N)

    for i in range(N):

        # ------------------------------------------------------------------
        # Get this pulsars necessary data
        # ------------------------------------------------------------------

        R = pulsars['r'][i].to(u.pc)

        Pb = pulsars['Pb'][i].to('s')

        Pbdot_meas = pulsars['Pbdot_meas'][i]
        ΔPbdot_meas = pulsars['ΔPbdot_meas'][i]

        # ------------------------------------------------------------------
        # Compute the cluster component distribution, from the model
        # ------------------------------------------------------------------

        PdotP_domain, PdotP_c_prob = cluster_component(model, R, mass_bin,
                                                       logspaced=True)
        Pdot_domain = (Pb * PdotP_domain).decompose()

        # ------------------------------------------------------------------
        # Compute gaussian measurement error distribution
        # ------------------------------------------------------------------

        err = util.gaussian(x=Pdot_domain, sigma=ΔPbdot_meas, mu=0)

        # ------------------------------------------------------------------
        # Convolve the different distributions
        # ------------------------------------------------------------------

        conv = np.convolve(err, PdotP_c_prob, 'same')

        # Normalize
        conv /= interp.UnivariateSpline(
            Pdot_domain, conv, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        # ------------------------------------------------------------------
        # Compute the Shklovskii (proper motion) effect component
        # ------------------------------------------------------------------

        cluster_μ <<= u.Unit("mas/yr")

        PdotP_pm = shklovskii_component(cluster_μ, model.d)

        # ------------------------------------------------------------------
        # Compute the galactic potential component
        # ------------------------------------------------------------------

        PdotP_gal = galactic_component(*(coords * u.deg), D=model.d)

        # ------------------------------------------------------------------
        # Interpolate the likelihood value from the overall distribution
        # ------------------------------------------------------------------

        prob_dist = interp.interp1d(
            (PdotP_domain) + PdotP_pm + PdotP_gal, conv,
            assume_sorted=True, bounds_error=False, fill_value=0.0
        )

        probs[i] = prob_dist(Pbdot_meas / Pb)

    # ----------------------------------------------------------------------
    # Multiply all the probabilities and return the total log probability.
    # ----------------------------------------------------------------------

    return np.sum(np.log(probs))


@_angular_units
def likelihood_number_density(model, ndensity, *, mass_bin=None):
    r'''Compute the log likelihood of the cluster number density profile

    Computes the log likelihood component of a cluster's number density profile,
    assuming a Gaussian likelihood. The model profile is scaled to fit the shape
    of the observation data, and a nuisance parameter is introduced to
    add a constant error component and minimize the background effects present
    near the outskirts of the cluster.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    ndensity : fitter.core.data.Dataset
        Number density profile dataset used to compute probability distribution
        and evaluate log likelihood

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        assumes

    Returns
    -------
    float
        Log likelihood value

    Notes
    -----
    As the translation between discrete number density and surface-brightness
    observations is difficult to quantify, the model is actually only fit on
    the shape of the number density profile data.
    To accomplish this the modelled number density is scaled to have the
    same mean value as the surface brightness data (K scaling factor).
    The chosen K factor minimizes chi-squared:

    .. math:: K = \frac{\sum \Sigma_{obs} \Sigma_{model} / \delta\Sigma_{obs}^2}
                       {\sum \Sigma_{model}^2 / \delta\Sigma_{obs}^2}

    References
    ----------
    [1]Hénault-Brunet, V., Gieles, M., Strader, J., Peuten, M., Balbinot, E.,
        and Douglas, K. E. K., “On the black hole content and initial mass
        function of 47 Tuc”, Monthly Notices of the Royal Astronomical Society,
        vol. 491, no. 1, pp. 113–128, 2020.

    '''
    # TODO the units are all messed up on this one, simply being ignored

    if mass_bin is None:
        if 'm' in ndensity.mdata:
            mass_bin = np.where(model.mj == ndensity.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Set cutoff to avoid fitting flat end of data
    # TODO should do this cutoff based on a flatness, rather than a set value
    valid = (ndensity['Σ'].value > 0.1)

    obs_r = ndensity['r'][valid]
    obs_Σ = ndensity['Σ'][valid].value
    obs_err = ndensity['ΔΣ'][valid].value

    # TODO the model Sigma is in /pc^2, and is not being converted to match obs?
    model_r = model.r.to(obs_r.unit)
    model_Σ = (model.Sigmaj[mass_bin] / model.mj[mass_bin]).value

    # Interpolated the model data at the measurement locations
    interpolated = np.interp(obs_r, model_r, model_Σ)

    # Calculate K scaling factor
    K = (np.sum(obs_Σ * interpolated / obs_err**2)
         / np.sum(interpolated**2 / obs_err**2))

    interpolated *= K

    # Now nuisance parameter
    yerr = np.sqrt(obs_err**2 + model.s2)

    # Now regular gaussian likelihood
    return -0.5 * np.sum(
        (obs_Σ - interpolated)**2 / yerr**2 + np.log(yerr**2)
    )


@_angular_units
def likelihood_pm_tot(model, pm, *, mass_bin=None):
    r'''Compute the log likelihood of the cluster total proper motion dispersion

    Computes the log likelihood component of a cluster's proper motion
    dispersion profile for the total proper motion, that is, the combined radial
    and tangential components, assuming a gaussian likelihood.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    pm : fitter.core.data.Dataset
        Proper motion dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`

    Returns
    -------
    float
        Log likelihood value

    Notes
    -----
    The "total" combined proper motion is given by the averaged vector:

    .. math:: PM_{tot} = \sqrt{\frac{(PM_{T}^2 + PM_{R}^2)}{2}}

    '''

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Get model values
    model_tot = np.sqrt(0.5 * (model.v2Tj[mass_bin] + model.v2Rj[mass_bin]))

    # Convert model units
    model_r = model.r.to(pm['r'].unit)
    model_tot = model_tot.to(pm['PM_tot'].unit)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_tot', model_r, model_tot)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_tot)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_tot'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


@_angular_units
def likelihood_pm_ratio(model, pm, *, mass_bin=None):
    r'''Compute the log likelihood of the cluster proper motion dispersion ratio

    Computes the log likelihood component of a cluster's proper motion
    dispersion anisotropy profile as the ratio of the tangential to radial
    dispersions, assuming a gaussian likelihood.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    pm : fitter.core.data.Dataset
        Proper motion dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`

    Returns
    -------
    float
        Log likelihood value

    Notes
    -----
    The proper motion ratio, or anisotropy measure, is given by the fraction:

    .. math:: PM_{ratio} = \sqrt{\frac{PM_{T}^2}{PM_{R}^2}}

    '''

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Get model values
    model_ratio = np.sqrt(model.v2Tj[mass_bin] / model.v2Rj[mass_bin])

    # Convert model units
    model_r = model.r.to(pm['r'].unit)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_ratio', model_r, model_ratio)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_ratio.decompose())

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_ratio'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


@_angular_units
def likelihood_pm_T(model, pm, *, mass_bin=None):
    '''Compute the log likelihood of the cluster tangential proper motion

    Computes the log likelihood component of a cluster's proper motion
    dispersion profile for the tangential proper motion, in relation to the
    cluster centre, assuming a gaussian likelihood.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    pm : fitter.core.data.Dataset
        Proper motion dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`

    Returns
    -------
    float
        Log likelihood value

    '''

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Get model values
    model_T = np.sqrt(model.v2Tj[mass_bin])

    # Convert model units
    model_r = model.r.to(pm['r'].unit)
    model_T = model_T.to(pm['PM_T'].unit)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_T', model_r, model_T)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_T)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_T'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


@_angular_units
def likelihood_pm_R(model, pm, *, mass_bin=None):
    '''Compute the log likelihood of the cluster radial proper motion

    Computes the log likelihood component of a cluster's proper motion
    dispersion profile for the radial proper motion, in relation to the
    cluster centre, assuming a gaussian likelihood.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    pm : fitter.core.data.Dataset
        Proper motion dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`

    Returns
    -------
    float
        Log likelihood value

    '''

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Get model values
    model_R = np.sqrt(model.v2Rj[mass_bin])

    # Convert model units
    model_r = model.r.to(pm['r'].unit)
    model_R = model_R.to(pm['PM_R'].unit)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_R', model_r, model_R)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_R)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_R'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


@_angular_units
def likelihood_LOS(model, vlos, *, mass_bin=None):
    '''Compute the log likelihood of the cluster LOS velocity dispersion

    Computes the log likelihood component of a cluster's velocity
    dispersion profile for the line-of-sight velocities, assuming a gaussian
    likelihood.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    vlos : fitter.core.data.Dataset
        Velocity dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`

    Returns
    -------
    float
        Log likelihood value

    '''

    if mass_bin is None:
        if 'm' in vlos.mdata:
            mass_bin = np.where(model.mj == vlos.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Get model values
    model_LOS = np.sqrt(model.v2pj[mass_bin])

    # Convert model units
    model_r = model.r.to(vlos['r'].unit)
    model_LOS = model_LOS.to(vlos['σ'].unit)

    # Build asymmetric error, if exists
    obs_err = vlos.build_err('σ', model_r, model_LOS)

    # Interpolated model at data locations
    interpolated = np.interp(vlos['r'], model_r, model_LOS)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (vlos['σ'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


@_angular_units
def likelihood_mass_func(model, mf):
    r'''Compute the log likelihood of the cluster mass function

    Computes the log likelihood component of a cluster's present day mass
    function distribution of visible stars. Profiles of the relative number
    of stars in each mass bin are compared against the computed mass function
    N of the model, given by it's density profile.

    A Gaussian likelihood is assumed, with a δN Poisson error accompanying the
    mass function nuisance parameter `F`.

    parameters
    ----------
    model : fitter.Model
        Cluster model use to compute probability distribution

    mf : fitter.core.data.Dataset
        Mass function profile dataset used to compute probability distribution
        and evaluate log likelihood

    Returns
    -------
    float
        Log likelihood value

    Notes
    -----
    The model mass function N is given for each stellar mass bin by the
    integral of the surface density profile within each radial bin:

    .. math:: N = \int_{r_0}^{r_1} 2 \pi r \Sigma(r) dr

    '''
    # TODO the units in mf are messy, due to all the interpolations

    tot_likelihood = 0

    rbin_size = 0.4 * u.arcmin

    for annulus_ind in np.unique(mf['bin']):

        # we only want to use the obs data for this r bin
        r_mask = (mf['bin'] == annulus_ind)

        r1 = (rbin_size * annulus_ind).to(model.r.unit)
        r2 = (rbin_size * (annulus_ind + 1)).to(model.r.unit)

        # Get a binned version of N_model (an Nstars for each mbin)
        binned_N_model = np.empty(model.nms)
        for mbin_ind in range(model.nms):

            # Interpolate the model density at the data locations
            density = interp.interp1d(
                model.r, 2 * np.pi * model.r * model.Sigmaj[mbin_ind],
                kind="cubic"
            )

            # Convert density spline into Nstars
            binned_N_model[mbin_ind] = (
                integ.quad(density, r1.value, r2.value)[0]
                / (model.mj[mbin_ind] * model.mes_widths[mbin_ind]).value
            )

        # interpolate a func N_model = f(mean mass) from the binned N_model
        interp_N_model = interp.interp1d(
            model.mj[:model.nms], binned_N_model, fill_value="extrapolate"
        )
        # Grab the interpolated N_model's at the data mean masses
        N_model = interp_N_model(mf['mbin_mean'][r_mask])

        # Grab the N_data (adjusted by width to get an average
        #                   dr of a bin (like average-interpolating almost))
        N_data = (mf['N'][r_mask] / mf['mbin_width'][r_mask]).value
        err_data = (mf['Δmbin'][r_mask] / mf['mbin_width'][r_mask]).value

        # Compute δN_model from poisson error, and nuisance factor
        err = np.sqrt(err_data**2 + (model.F * N_data)**2)

        # compute final gaussian log likelihood
        tot_likelihood += -0.5 * np.sum(
            (N_data - N_model)**2 / err**2 + np.log(err**2)
        )

    return tot_likelihood


# --------------------------------------------------------------------------
# Composite likelihood functions
# --------------------------------------------------------------------------

# Main likelihood function, generates the model(theta) passes it to the
# individual likelihood functions and collects their results.
def log_likelihood(theta, observations, L_components):

    try:
        model = Model(theta, observations)
    except ValueError:
        logging.debug(f"Model did not converge with {theta=}")
        return -np.inf, -np.inf * np.ones(len(L_components))

    # Calculate each log likelihood
    probs = np.array([
        likelihood(model, observations[key], *args)
        for (key, likelihood, *args) in L_components
    ])

    return sum(probs), probs


# Combines the likelihood with the prior
def posterior(theta, observations, fixed_initials=None, L_components=None):
    '''
    theta : array of theta values
    observations : data.Observations
    fixed_initials : dict of any theta values to fix
    L_components : output from determine_components
    '''

    if fixed_initials is None:
        fixed_initials = {}

    if L_components is None:
        L_components = observations.valid_likelihoods

    # get a list of variable params, sorted for the unpacking of theta
    variable_params = DEFAULT_INITIALS.keys() - fixed_initials.keys()
    params = sorted(variable_params, key=list(DEFAULT_INITIALS).index)

    # Update to unions when 3.9 becomes enforced
    theta = dict(zip(params, theta), **fixed_initials)

    # TODO add type check on theta, cause those exceptions aren't very pretty
    # TODO if this fails on a fixed param, the run should be aborted
    # Also these ranges will probably have to change a bunch when we expand
    # Prior probability function
    if not all(priors := (3 < theta['W0'] < 20,
                          0.01 < theta['M'] < 10,
                          0.5 < theta['rh'] < 15,
                          0 < theta['ra'] < 5,
                          0 < theta['g'] < 2.3,
                          0.3 < theta['delta'] < 0.8,
                          0 < theta['s2'] < 15,
                          0 < theta['F'] < 0.7,
                          -2 < theta['a1'] < 6,
                          -2 < theta['a2'] < 6,
                          -2 < theta['a3'] < 6,
                          0 < theta['BHret'] < 100,
                          2 < theta['d'] < 8)):

        inv = {f"{k}={theta[k]}" for i, k in enumerate(theta) if not priors[i]}
        logging.debug(f"Theta outside priors domain: {'; '.join(inv)}")

        return -np.inf, *(-np.inf * np.ones(len(L_components)))

    probability, individuals = log_likelihood(theta, observations, L_components)

    return probability, *individuals
