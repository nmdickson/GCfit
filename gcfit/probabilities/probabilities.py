from .pulsars import *
from .priors import Priors
from .. import util
from ..core.data import (DEFAULT_THETA, DEFAULT_EV_THETA,
                         FittableModel, FittableEvolvedModel)

import numpy as np
import astropy.units as u
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


# TODO standardize which interpolation function we're using, 3 are in play rn


# TODO this messes up the error messages sometimes (when .get(model) fails)
def _angular_units(func):
    '''Decorator for supporting all angular unit equivalencies.
    For use in wrapping likelihood functions, therefore assumes 'model'
    will be first arg, or in kwargs.
    '''
    import functools

    @functools.wraps(func)
    def angular_units_decorator(*args, **kwargs):

        model = kwargs.get('model') or args[0]

        with u.set_enabled_equivalencies(util.angular_width(model.d)):
            return func(*args, **kwargs)

    return angular_units_decorator


# --------------------------------------------------------------------------
# Component likelihood functions
# --------------------------------------------------------------------------


@_angular_units
def likelihood_pulsar_spin(model, pulsars, Pdot_kde, cluster_μ, coords,
                           use_DM=False, *, mass_bin=None, hyperparams=False):
    r'''Compute the loglikelihood of pulsar spin period derivatives.

    Computes the log likelihood component of a cluster's pulsar's spin
    period derivatives, evaluating the observed pulsar timing solutions
    against the combined probability distributions of the clusters acceleration
    field, the pulsars intrinsic spin-down, the proper-motion contribution and
    the galactic potential.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    pulsars : gcfit.core.data.Dataset
        Pulsars dataset used to compute probability distribution and evaluate
        log likelihood.

    Pdot_kde : scipy.stats.gaussian_kde
        Gaussian KDE of the galactic field pulsars Pdot-P distribution, from
        `field_Pdot_KDE`. Should be generated beforehand for speed, but if
        None, will generate at runtime.

    cluster_μ : float
        Total cluster proper motion, in mas/yr.

    coords : 2-tuple of float
        Cluster Galactic (Latitude, Longitude), in degrees.

    use_DM : bool, optional
        Whether to use dispersion measure data in pulsar likelihoods.

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else -1.

    hyperparams : bool, optional
        Not implemented.

    Returns
    -------
    float
        Log likelihood value.

    See Also
    --------
    likelihood_pulsar_orbital : Binary pulsar orbital period likelihood.
    gcfit.probabilities.pulsars : Module containing all pulsar prob. components.

    Notes
    -----
    The combined timing solutions for pulsars embedded in clusters, which
    is used alongside the model acceleration field to compute this likelihood,
    is given by the combination of components:

    .. math::

        \left(\frac{\dot{P}}{P}\right)_{\rm{obs}}
            = \left(\frac{\dot{P}}{P}\right)_{\rm{int}}
            + \frac{a_{\rm{clust}}}{c}
            + \frac{a_{\rm{gal}}}{c} + \frac{\mu^2 D}{c}
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

        if R >= model.rt:

            mssg = (f"Pulsar {pulsars['id'][i]} is outside cluster truncation "
                    f"radius {model.rt}")
            logging.debug(mssg)

            return np.NINF

        P = pulsars['P'][i].to('s')

        Pdot_meas = pulsars['Pdot'][i]
        ΔPdot_meas = np.abs(pulsars['ΔPdot'][i])

        # ------------------------------------------------------------------
        # Compute the cluster component distribution, from the model
        # ------------------------------------------------------------------

        try:
            if use_DM:

                DM = pulsars['DM'][i]
                ΔDM = pulsars['ΔDM'][i]

                DM_mdata = pulsars.mdata

                PdotP_domain, PdotP_c_prob = cluster_component(
                    model, R, mass_bin=mass_bin,
                    DM=DM, ΔDM=ΔDM, DM_mdata=DM_mdata
                )

            else:
                PdotP_domain, PdotP_c_prob = cluster_component(
                    model, R, DM=None, DM_mdata=None, mass_bin=mass_bin
                )

        except ValueError as err:
            # The cluster component shouldn't be crashing nearly as often now,
            # should only happen when Paz fails to integrate to 1.0
            mssg = f"""
            Pulsar `cluster_component` failed with params:
            {model.theta=}, {R=}, {mass_bin=}, DM={DM if use_DM else None},
            with error:
            """
            logging.warning(mssg, exc_info=err)

            return np.NINF

        Pdot_domain = (P * PdotP_domain).decompose()

        # linear to avoid effects around asymptote
        Pdot_c_spl = interp.UnivariateSpline(
            Pdot_domain, PdotP_c_prob, k=1, s=0, ext=1
        )

        # ------------------------------------------------------------------
        # Set up the equally-spaced linear convolution domain
        # ------------------------------------------------------------------

        # TODO both 5000 and 1e-18 need to be computed dynamically
        #   5000 to be enough steps to sample the gaussian and int peaks
        #   1e-18 to be far enough for the int distribution to go to zero
        #   Both balanced so as to use way too much memory unnecessarily
        #   Must be symmetric, to avoid bound effects

        # mirrored/starting at zero so very small gaussians become the δ-func
        lin_domain = np.linspace(0., 3e-18, 5_000 // 2)
        lin_domain = np.concatenate((np.flip(-lin_domain[1:]), lin_domain))

        # ------------------------------------------------------------------
        # Compute gaussian measurement error distribution
        # ------------------------------------------------------------------

        # TODO if width << Pint width, maybe don't bother with first conv.

        # NOTE: this now uses the lin_domain instead of the PdotP domain
        # in order to accommodate pulsar X who's error spline was being cut
        # too soon, giving zero probability to valid regions.
        # if lin_domain gets dynamically computed in the future, make sure it's
        # large enough to accommodate pulsar X.
        err = util.gaussian(x=lin_domain, sigma=ΔPdot_meas, mu=0)

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
            Pdot_int_domain, Pdot_int_prob, k=1, s=0, ext=1
        )

        Pdot_int_prob = util.RV_transform(
            domain=10**Pdot_int_domain, f_X=Pdot_int_spl,
            h=np.log10, h_prime=lambda y: (1 / (np.log(10) * y))
        )

        Pdot_int_spl = interp.UnivariateSpline(
            10**Pdot_int_domain, Pdot_int_prob, k=1, s=0, ext=1
        )

        # ------------------------------------------------------------------
        # Convolve the different distributions
        # ------------------------------------------------------------------

        conv1 = np.convolve(err, Pdot_c_spl(lin_domain), 'same')

        conv2 = np.convolve(conv1, Pdot_int_spl(lin_domain), 'same')

        # Normalize
        conv2 /= interp.UnivariateSpline(
            lin_domain, conv2, k=1, s=0, ext=1
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

    logprobs = np.log(probs)

    # Replace NaNs with -inf
    logprobs[np.isnan(logprobs)] = np.NINF

    return np.sum(logprobs)


@_angular_units
def likelihood_pulsar_orbital(model, pulsars, cluster_μ, coords, use_DM=False,
                              *, mass_bin=None, hyperparams=False):
    r'''Compute the loglikelihood of binary pulsar orbital period derivatives.

    Computes the log likelihood component of a cluster's binary pulsar's orbital
    period derivatives, evaluating the observed orbital timing solutions
    against the combined probability distributions of the clusters acceleration
    field, the proper-motion contribution and the galactic potential.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    pulsars : gcfit.core.data.Dataset
        Pulsars dataset used to compute probability distribution and evaluate
        log likelihood.

    cluster_μ : float
        Total cluster proper motion, in mas/yr.

    coords : 2-tuple of float
        Cluster Galactic (Latitude, Longitude), in degrees.

    use_DM : bool, optional
        Whether to use dispersion measure data in pulsar likelihoods.

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else -1.

    hyperparams : bool, optional
        Not implemented.

    Returns
    -------
    float
        Log likelihood value.

    See Also
    --------
    likelihood_pulsar_spin : Pulsar spin period likelihood.
    gcfit.probabilities.pulsars : Module containing all pulsar prob. components.

    Notes
    -----
    The combined timing solutions for pulsars embedded in clusters, which
    is used alongside the model acceleration field to compute this likelihood,
    is given by the combination of components:

    .. math::

        \left(\frac{\dot{P}}{P}\right)_{\rm{obs}}
            = \left(\frac{\dot{P}}{P}\right)_{\rm{int}}
            + \frac{a_{\rm{clust}}}{c}
            + \frac{a_{\rm{gal}}}{c} + \frac{\mu^2 D}{c}
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

        if R >= model.rt:

            mssg = (f"Pulsar {pulsars['id'][i]} is outside cluster truncation "
                    f"radius {model.rt}")
            logging.debug(mssg)

            return np.NINF

        Pb = pulsars['Pb'][i].to('s')

        Pbdot_meas = pulsars['Pbdot'][i]
        ΔPbdot_meas = pulsars['ΔPbdot'][i]

        # ------------------------------------------------------------------
        # Compute the cluster component distribution, from the model
        # ------------------------------------------------------------------

        try:
            if use_DM:

                DM = pulsars['DM'][i]
                ΔDM = pulsars['ΔDM'][i]

                DM_mdata = pulsars.mdata

                PdotP_domain, PdotP_c_prob = cluster_component(
                    model, R, mass_bin=mass_bin,
                    DM=DM, ΔDM=ΔDM, DM_mdata=DM_mdata
                )

            else:
                PdotP_domain, PdotP_c_prob = cluster_component(
                    model, R, DM=None, DM_mdata=None, mass_bin=mass_bin
                )

        except ValueError as err:
            # The cluster component shouldn't be crashing nearly as often now,
            # should only happen when Paz fails to integrate to 1.0
            mssg = f"""
            Pulsar `cluster_component` failed with params: "
            {model.theta=}, {R=}, {mass_bin=}, DM={DM if use_DM else None},
            with error:
            """
            logging.warning(mssg, exc_info=err)

            return np.NINF

        Pdot_domain = (Pb * PdotP_domain).decompose()

        # linear to avoid effects around asymptote
        Pdot_c_spl = interp.UnivariateSpline(
            Pdot_domain, PdotP_c_prob, k=1, s=0, ext=1
        )

        # ------------------------------------------------------------------
        # Set up the equally-spaced linear convolution domain
        # ------------------------------------------------------------------

        # mirrored/starting at zero so very small gaussians become the δ-func
        lin_domain = np.linspace(0., 1e-9, 10_000 // 2)
        lin_domain = np.concatenate((np.flip(-lin_domain[1:]), lin_domain))

        # ------------------------------------------------------------------
        # Compute gaussian measurement error distribution
        # ------------------------------------------------------------------

        err = util.gaussian(x=lin_domain, sigma=ΔPbdot_meas, mu=0)

        # err_spl = interp.UnivariateSpline(Pdot_domain, err, k=1, s=0, ext=1)

        # ------------------------------------------------------------------
        # Convolve the different distributions
        # ------------------------------------------------------------------

        # conv = np.convolve(err, PdotP_c_prob, 'same')
        conv = np.convolve(err, Pdot_c_spl(lin_domain), 'same')

        # Normalize
        conv /= interp.UnivariateSpline(
            lin_domain, conv, k=1, s=0, ext=1
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
            (lin_domain / Pb) + PdotP_pm + PdotP_gal, conv,
            assume_sorted=True, bounds_error=False, fill_value=0.0
        )

        probs[i] = prob_dist(Pbdot_meas / Pb)

    # ----------------------------------------------------------------------
    # Multiply all the probabilities and return the total log probability.
    # ----------------------------------------------------------------------

    logprobs = np.log(probs)

    # Replace NaNs with -inf
    logprobs[np.isnan(logprobs)] = np.NINF

    return np.sum(logprobs)


@_angular_units
def likelihood_number_density(model, ndensity, *,
                              mass_bin=None, hyperparams=False):
    r'''Compute the loglikelihood of the cluster number density profile.

    Computes the log likelihood component of a cluster's number density profile,
    assuming a Gaussian likelihood. The model profile is scaled to fit the shape
    of the observation data, and a nuisance parameter is introduced to
    add a constant error component and minimize the background effects present
    near the outskirts of the cluster.

    Optionally, a background level can be provided in the dataset metadata
    (`ndensity.mdata["background"]`) which will be subtracted from all
    observations before calculation of the likelihood. By default, will be
    assumed to have same units as Σ.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    ndensity : gcfit.core.data.Dataset
        Number density profile dataset used to compute probability distribution
        and evaluate log likelihood.

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        assumes.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters.

    Returns
    -------
    float
        Log likelihood value.

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

    if mass_bin is None:
        if 'm' in ndensity.mdata:
            mass_bin = np.where(model.mj == ndensity.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    if hyperparams:
        likelihood = util.hyperparam_likelihood
    else:
        likelihood = util.gaussian_likelihood

    background = ndensity.mdata.get('background', 0.0) << ndensity['Σ'].unit

    obs_r = ndensity['r']
    obs_Σ = ndensity['Σ'] - background
    obs_err = ndensity['ΔΣ']

    # Now nuisance parameter (from θ, not the model velocity scale)
    s2 = model.theta['s2'] << u.arcmin**-4
    yerr = np.sqrt(obs_err**2 + s2)

    model_r = model.r.to(obs_r.unit)
    model_Σ = model.Sigmaj[mass_bin] / model.mj[mass_bin]

    # Interpolated the model data at the measurement locations
    interpolated = np.interp(obs_r, model_r, model_Σ).to(obs_Σ.unit)

    # Calculate K scaling factor
    K = (np.sum(obs_Σ * interpolated / yerr**2)
         / np.sum(interpolated**2 / yerr**2))

    if np.isnan(K):
        K = 1

    interpolated *= K

    return likelihood(obs_Σ, interpolated, yerr)


@_angular_units
def likelihood_pm_tot(model, pm, *, mass_bin=None, hyperparams=False):
    r'''Compute the loglikelihood of the cluster total proper motion dispersion.

    Computes the log likelihood component of a cluster's proper motion
    dispersion profile for the total proper motion, that is, the combined radial
    and tangential components, assuming a gaussian likelihood.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    pm : gcfit.core.data.Dataset
        Proper motion dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood.

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters.

    Returns
    -------
    float
        Log likelihood value.

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

    if hyperparams:
        likelihood = util.hyperparam_likelihood
    else:
        likelihood = util.gaussian_likelihood

    # Get model values
    model_tot = np.sqrt(0.5 * (model.v2Tj[mass_bin] + model.v2Rj[mass_bin]))

    # TODO some datasets have Δr, should probably account for those as well
    # Convert model units
    model_r = model.r.to(pm['r'].unit)
    model_tot = model_tot.to(pm['PM_tot'].unit)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_tot', model_r, model_tot)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_tot)

    return likelihood(pm['PM_tot'], interpolated, obs_err)


@_angular_units
def likelihood_pm_ratio(model, pm, *, mass_bin=None, hyperparams=False):
    r'''Compute the loglikelihood of the cluster proper motion dispersion ratio.

    Computes the log likelihood component of a cluster's proper motion
    dispersion anisotropy profile as the ratio of the tangential to radial
    dispersions, assuming a gaussian likelihood.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    pm : gcfit.core.data.Dataset
        Proper motion dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood.

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters.

    Returns
    -------
    float
        Log likelihood value.

    Notes
    -----
    The proper motion ratio, or anisotropy measure, is given by the fraction:

    .. math:: PM_{ratio} = \sqrt{\frac{PM_{T}^2}{PM_{R}^2}}
    '''
    # TODO is there some way we could be using model.betaj instead of this frac

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    if hyperparams:
        likelihood = util.hyperparam_likelihood
    else:
        likelihood = util.gaussian_likelihood

    # Get model values
    model_ratio = np.sqrt(model.v2Tj[mass_bin] / model.v2Rj[mass_bin])

    # Fix the occasional 0/0 error at the edge of the model
    model_ratio[np.isnan(model_ratio)] = 1.0

    # Convert model units
    model_r = model.r.to(pm['r'].unit)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_ratio', model_r, model_ratio)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_ratio.decompose())

    return likelihood(pm['PM_ratio'], interpolated, obs_err)


@_angular_units
def likelihood_pm_T(model, pm, *, mass_bin=None, hyperparams=False):
    '''Compute the loglikelihood of the cluster tangential proper motion.

    Computes the log likelihood component of a cluster's proper motion
    dispersion profile for the tangential proper motion, in relation to the
    cluster centre, assuming a gaussian likelihood.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    pm : gcfit.core.data.Dataset
        Proper motion dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood.

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters.

    Returns
    -------
    float
        Log likelihood value.
    '''

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    if hyperparams:
        likelihood = util.hyperparam_likelihood
    else:
        likelihood = util.gaussian_likelihood

    # Get model values
    model_T = np.sqrt(model.v2Tj[mass_bin])

    # Convert model units
    model_r = model.r.to(pm['r'].unit)
    model_T = model_T.to(pm['PM_T'].unit)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_T', model_r, model_T)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_T)

    return likelihood(pm['PM_T'], interpolated, obs_err)


@_angular_units
def likelihood_pm_R(model, pm, *, mass_bin=None, hyperparams=False):
    '''Compute the loglikelihood of the cluster radial proper motion.

    Computes the log likelihood component of a cluster's proper motion
    dispersion profile for the radial proper motion, in relation to the
    cluster centre, assuming a gaussian likelihood.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    pm : gcfit.core.data.Dataset
        Proper motion dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood.

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters.

    Returns
    -------
    float
        Log likelihood value.
    '''

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    if hyperparams:
        likelihood = util.hyperparam_likelihood
    else:
        likelihood = util.gaussian_likelihood

    # Get model values
    model_R = np.sqrt(model.v2Rj[mass_bin])

    # Convert model units
    model_r = model.r.to(pm['r'].unit)
    model_R = model_R.to(pm['PM_R'].unit)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_R', model_r, model_R)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_R)

    return likelihood(pm['PM_R'], interpolated, obs_err)


@_angular_units
def likelihood_LOS(model, vlos, *, mass_bin=None, hyperparams=False):
    '''Compute the loglikelihood of the cluster LOS velocity dispersion.

    Computes the log likelihood component of a cluster's velocity
    dispersion profile for the line-of-sight velocities, assuming a gaussian
    likelihood.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    vlos : gcfit.core.data.Dataset
        Velocity dispersions profile dataset used to compute probability
        distribution and evaluate log likelihood.

    mass_bin : int, optional
        Index of `model.mj` mass bin to use in all calculations.
        If None (default), attempts to read 'm' from `pulsars.mdata`, else
        uses largest of the main sequence bins, given by `model.nms`.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters.

    Returns
    -------
    float
        Log likelihood value.
    '''

    if mass_bin is None:
        if 'm' in vlos.mdata:
            mass_bin = np.where(model.mj == vlos.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    if hyperparams:
        likelihood = util.hyperparam_likelihood
    else:
        likelihood = util.gaussian_likelihood

    # Get model values
    model_LOS = np.sqrt(model.v2pj[mass_bin])

    # Convert model units
    model_r = model.r.to(vlos['r'].unit)
    model_LOS = model_LOS.to(vlos['σ'].unit)

    # Build asymmetric error, if exists
    obs_err = vlos.build_err('σ', model_r, model_LOS)

    # Interpolated model at data locations
    interpolated = np.interp(vlos['r'], model_r, model_LOS)

    return likelihood(vlos['σ'], interpolated, obs_err)


@_angular_units
def likelihood_mass_func(model, mf, fields, *, hyperparams=False):
    r'''Compute the loglikelihood of the cluster's PDMF.

    Computes the log likelihood component of a cluster's present day mass
    function (PDMF) distribution of visible stars. Radial profiles of the
    relative number of stars counted in each mass bin, within each observation's
    boundary polygons, are compared against the computed mass function
    N of the model, given by it's density profile and integrated over the same
    field.

    A Gaussian likelihood is assumed, with a δN Poisson error accompanying the
    mass function nuisance parameter `F`.

    Parameters
    ----------
    model : gcfit.FittableModel
        Cluster model used to compute probability distribution.

    mf : gcfit.core.data.Dataset
        Mass function profile dataset used to compute probability distribution
        and evaluate log likelihood.

    fields : dict
        Dictionary of `gcfit.util.mass.Field` field, as given by
        `gcfit.util.mass.initialize_fields`.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters.

    Returns
    -------
    float
        Log likelihood value.

    See Also
    --------
    util.mass.Field.MC_integrate :
        Monte Carlo integration method used to integrate the surface density
        profile.

    Notes
    -----
    The model mass function N is given for each stellar mass bin by the
    integral of the surface density profile within each radial bin, within the
    relevant field boundaries:

    .. math:: N = \int_{r_0}^{r_1} \Sigma(r) dr
    '''

    if hyperparams:
        likelihood = util.hyperparam_likelihood
    else:
        likelihood = util.gaussian_likelihood

    # TODO could probably do the radial slicing beforehand as well, if its slow
    # if not field:
    #     cen = (obs.mdata['RA'], obs.mdata['DEC'])
    #     field = mass.Field(mf['fields'], cen)

    if model.nms <= 1:
        mssg = f"Model must have more than one stellar mass bin ({model.nms=})"
        logging.debug(mssg)
        return -np.inf
        # raise ValueError(mssg)

    # ----------------------------------------------------------------------
    # Generate the mass splines before the loops, to save repetition
    # ----------------------------------------------------------------------

    densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                for j in range(model.nms)]

    # ------------------------------------------------------------------
    # Determine the observation data which corresponds to this field
    # ------------------------------------------------------------------

    rbins = np.c_[mf['r1'], mf['r2']]

    mbin_mean = (mf['m1'] + mf['m2']) / 2.
    mbin_width = mf['m2'] - mf['m1']

    N = mf['N'] / mbin_width
    ΔN = mf['ΔN'] / mbin_width

    # ------------------------------------------------------------------
    # Iterate over each radial bin in this field, slicing out the radial
    # shell from the field
    # ------------------------------------------------------------------

    N_data = np.empty_like(N)
    N_model = np.empty_like(N)
    err = np.empty_like(N)

    for field_slice, (r_in, r_out) in zip(fields, np.unique(rbins, axis=0)):
        r_mask = (mf['r1'] == r_in) & (mf['r2'] == r_out)

        # --------------------------------------------------------------
        # Sample this slice of the field M times, and integrate to get N
        # --------------------------------------------------------------

        sample_radii = field_slice._prev_sample.to(u.pc)

        binned_N_model = np.empty(model.nms) << N_data.unit
        for j in range(model.nms):
            Nj = field_slice.MC_integrate(densityj[j], sample=sample_radii)
            widthj = (model.mj[j] * model.mbin_widths[j])
            binned_N_model[j] = Nj / widthj

        # TODO if a mass bin is depleted it should be 0, not extrapolated!
        N_spline = util.QuantitySpline(model.mj[:model.nms],
                                       binned_N_model,
                                       ext=0, k=1)

        # --------------------------------------------------------------
        # Add the error and compute the log likelihood
        # --------------------------------------------------------------

        N_data[r_mask] = N[r_mask]
        N_model[r_mask] = N_spline(mbin_mean[r_mask])
        err[r_mask] = model.theta['F'] * ΔN[r_mask]

    return likelihood(N_data, N_model, err)


# --------------------------------------------------------------------------
# Composite likelihood functions
# --------------------------------------------------------------------------


def log_likelihood(theta, observations, L_components, hyperparams, evolved):
    r'''Compute log likelihood of given `theta`, based on component likelihoods.

    Main likelihood function, which generates the relevant model based on
    the given set of parameters (`theta`) and passes it to the
    various individual likelihood functions (as given in `L_components`)
    and collects and sums their results.

    Parameters
    ----------
    theta : dict
        The model input parameters (W0, M, rh, ra, g, delta, a1, a2, a3,
        BHret, s2, F and d). Passed directly to `gcfit.FittableModel` to
        generate the model used in all likelihood functions.

    observations : Observations
        The `Observations` instance corresponding to this cluster, used to
        initialize the model and to read in all datasets specified by
        `L_components`.

    L_components : list of lists
        List of likelihood components to compute. Must be a list of lists
        in the same format as `Observations.valid_likelihoods` (dataset name,
        likelihood function, *function params).

    hyperparams : bool
        Whether to include bayesian hyperparameters in all likelihood functions.

    Returns
    -------
    lnL : float
        The total log likelihood.

    lnL_i : np.ndarray
        An array of individual log likelihoods for each component, sorted to
        match `L_components`.

    See Also
    --------
    gcfit.FittableModel : Model class initialized using `theta`.

    Notes
    -----
    The total likelihood is computed as the product of all individual
    likelihoods:

    .. math:: \ln(\mathcal{L})
                 = \sum_i^{\rm{datasets}} \ln(P(\mathcal{D_i} \mid \Theta))
                 = \sum_i \ln(\mathcal{L}_i(\Theta)))
    '''

    if evolved:
        model_cls = FittableEvolvedModel
    else:
        model_cls = FittableModel

    try:
        model = model_cls(theta, observations)
    except ValueError as err:
        logging.debug(f"Model did not converge with {theta=} ({err})")
        return -np.inf, -np.inf * np.ones(len(L_components))

    # Calculate each log likelihood
    probs = np.empty(len(L_components))
    for ind, (key, likelihood, *args) in enumerate(L_components):

        kwargs = {'hyperparams': hyperparams}

        probs[ind] = likelihood(model, observations[key], *args, **kwargs)

    return sum(probs), probs


def posterior(theta, observations, fixed_initials=None,
              L_components=None, prior_likelihood=None, *,
              hyperparams=False, return_indiv=True, evolved=False):
    '''Compute the full posterior probability given `theta` and `observations`.

    Combines the various likelihood functions (through `log_likelihood`)
    and the priors on the given `theta` parameters to compute and return
    the full posterior probability.

    This function is designed for use by a bayesian sampler class, like
    `emcee.EnsembleSampler` or `dynesty.DynamicNestedSampler`. The accepted
    arguments are therefore slightly different than what might be expected in
    other parts of the package (e.g. `theta` must be an array).

    Parameters
    ----------
    theta : np.ndarray
        An array of model input parameters, which must be in the expected order
        (W0, M, rh, ra, g, delta, a1, a2, a3, BHret, s2, F and d).
        Only parameters which are specified in `fixed_initials` may be excluded
        here. A dictionary of parameters is not allowed.

    observations : Observations
        The `Observations` instance corresponding to this cluster, to provide
        the "data" for this posterior calculation.

    fixed_initials : dict, optional
        An optional dictionary of parameters which provides fixed values for
        specific parameters used to fill out the `theta` array. This is useful
        for allowing samplers to explore a smaller set of parameters by fixing
        certain usually free ones.

    L_components : list of lists, optional
        List of likelihood components to compute. Must be a list of lists
        in the same format as `observations.valid_likelihoods` (dataset name,
        likelihood function, *function params). Will default to using those
        given by `observations.valid_likelihoods`.

    prior_likelihood : Priors or "ignore", optional
        The Priors object corresponding to the desired sets of priors for this
        given parameter set. If None (the default), will use the default priors
        defined in `priors.DEFAULT_PRIORS`. If "ignore", will assume the priors
        have already been dealt with (e.g. as done in `dynesty`) and will
        ignore them.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters in all likelihood functions.

    return_indiv : bool, optional
        If True (default) will also return all individual likelihood values
        alongside the posterior probability. Can be used by some sampler
        classes, such as the `blobs` functionality of `emcee`.

    Returns
    -------
    float or tuple
        The total posterior probability.
        If `return_indiv` is True, will return a tuple where the first element
        is the posterior probability and the remaining elements are the
        individual log likelihood values for each likelihood function.
    '''

    if fixed_initials is None:
        fixed_initials = {}

    if L_components is None:
        L_components = observations.valid_likelihoods

    if prior_likelihood is None:
        prior_likelihood = Priors(dict(), evolved=evolved)

    if evolved:
        default_θ = DEFAULT_EV_THETA
    else:
        default_θ = DEFAULT_THETA

    # Check if any values of theta are not finite, probably caused by invalid
    # prior transforms, and indicating we should return -inf
    if not np.all(np.isfinite(theta)):

        if return_indiv:
            return -np.inf, *(-np.inf * np.ones(len(L_components)))
        else:
            return -np.inf

    # get a list of variable params, sorted for the unpacking of theta
    variable_params = default_θ.keys() - fixed_initials.keys()
    params = sorted(variable_params, key=list(default_θ).index)

    # TODO add type check on theta, cause those exceptions aren't very pretty
    theta = dict(zip(params, theta)) | fixed_initials

    # prior likelihoods
    if prior_likelihood != 'ignore':

        if not np.isfinite(log_Pθ := prior_likelihood(theta)):

            if return_indiv:
                return -np.inf, *(-np.inf * np.ones(len(L_components)))
            else:
                return -np.inf

    else:
        log_Pθ = 0

    log_L, individuals = log_likelihood(theta, observations, L_components,
                                        hyperparams=hyperparams,
                                        evolved=evolved)

    probability = log_L + log_Pθ

    if return_indiv:
        return probability, *individuals
    else:
        return probability
