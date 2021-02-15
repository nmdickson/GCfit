from . import util
from .new_Paz import vec_Paz
from .data import DEFAULT_INITIALS, Model

import numpy as np
import astropy.units as u
from astropy.constants import c
import scipy.integrate as integ
import scipy.interpolate as interp

import logging
import fnmatch


# TODO standardize which interpolation funciton we're using, 3 are in play rn
# TODO look into just using `u.set_enabled_equivalencies`


# --------------------------------------------------------------------------
# Component likelihood functions
# --------------------------------------------------------------------------


def likelihood_pulsar_spin(model, pulsars, Pdot_kde, cluster_μ, coords, *,
                           mass_bin=None):

    # ----------------------------------------------------------------------
    # Get the pulsar P-Pdot_int kde
    # ----------------------------------------------------------------------

    if Pdot_kde is None:
        Pdot_kde = util.pulsar_Pdot_KDE()

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

        R = pulsars['r'][i].to(u.pc, util.angular_width(model.d))

        P = pulsars['P'][i]

        Pdot_meas = pulsars['Pdot_meas'][i]
        ΔPdot_meas = pulsars['ΔPdot_meas'][i]

        # ------------------------------------------------------------------
        # Compute the cluster component distribution of Pdot, from the model
        # ------------------------------------------------------------------

        a_domain, Pdot_c_prob = vec_Paz(model, R, mass_bin, logspaced=True)
        Pdot_domain = (P * a_domain / c).decompose()

        # linear to avoid effects around asymptote
        Pdot_c_spl = interp.UnivariateSpline(
            Pdot_domain, Pdot_c_prob, k=1, s=0, ext=1
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

        lin_domain = np.linspace(-1e-18, 1e-18, 5_000)

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

        pm = (cluster_μ * u.Unit("mas/yr")).to("1/s", u.dimensionless_angles())

        D = model.d.to('m')

        PdotP_pm = (pm**2 * D / c).decompose()

        # ------------------------------------------------------------------
        # Compute the galactic potential component
        # ------------------------------------------------------------------

        PdotP_gal = util.galactic_pot(*(coords * u.deg), D=model.d)

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

    return np.sum(np.log(probs))


def likelihood_pulsar_orbital(model, pulsars, cluster_μ, coords, *,
                              mass_bin=None):
    '''
    Like isolated pulsar, but using binary orbit timings rather than
    pulsar timings
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

        R = pulsars['r'][i].to(u.pc, util.angular_width(model.d))

        Pb = pulsars['Pb'][i]

        Pbdot_meas = pulsars['Pbdot_meas'][i]
        ΔPbdot_meas = pulsars['ΔPbdot_meas'][i]

        # ------------------------------------------------------------------
        # Compute the cluster component distribution of Pdot, from the model
        # ------------------------------------------------------------------

        a_domain, Pdot_c_prob = vec_Paz(model, R, mass_bin, logspaced=True)
        Pdot_domain = (Pb * a_domain / c).decompose()

        # ------------------------------------------------------------------
        # Compute gaussian measurement error distribution
        # ------------------------------------------------------------------

        err = util.gaussian(x=Pdot_domain, sigma=ΔPbdot_meas, mu=0)

        # ------------------------------------------------------------------
        # Convolve the different distributions
        # ------------------------------------------------------------------

        conv = np.convolve(err, Pdot_c_prob, 'same')

        # Normalize
        conv /= interp.UnivariateSpline(
            Pdot_domain, conv, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        # ------------------------------------------------------------------
        # Compute the Shklovskii (proper motion) effect component
        # ------------------------------------------------------------------

        pm = (cluster_μ * u.Unit("mas/yr")).to("1/s", u.dimensionless_angles())

        D = model.d.to('m')

        PdotP_pm = (pm**2 * D / c).decompose()

        # ------------------------------------------------------------------
        # Compute the galactic potential component
        # ------------------------------------------------------------------

        PdotP_gal = util.galactic_pot(*(coords * u.deg), D=model.d)

        # ------------------------------------------------------------------
        # Interpolate the likelihood value from the overall distribution
        # ------------------------------------------------------------------

        prob_dist = interp.interp1d(
            (Pdot_domain / Pb) + PdotP_pm + PdotP_gal, conv,
            assume_sorted=True, bounds_error=False, fill_value=0.0
        )

        probs[i] = prob_dist(Pbdot_meas / Pb)

    # ----------------------------------------------------------------------
    # Multiply all the probabilities and return the total log probability.
    # ----------------------------------------------------------------------

    return np.sum(np.log(probs))


def likelihood_number_density(model, ndensity, *, mass_bin=None):
    # TODO the units are all messed up on this one, simply being ignored

    if mass_bin is None:
        if 'm' in ndensity.mdata:
            mass_bin = np.where(model.mj == ndensity.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Set cutoff to avoid fitting flat end of data
    valid = (ndensity['Σ'].value > 0.1)

    obs_r = ndensity['r'][valid]
    obs_Σ = ndensity['Σ'][valid].value
    obs_err = ndensity['ΔΣ'][valid].value

    # TODO the model Sigma is in /pc^2, and is not being converted to match obs?
    model_r = model.r.to(u.arcmin, util.angular_width(model.d))
    model_Σ = (model.Sigmaj[mass_bin] / model.mj[mass_bin]).value

    # Interpolated the model data at the measurement locations
    interpolated = np.interp(obs_r, model_r, model_Σ)

    # K Scaling Factor

    # Because the translation between number density and surface-brightness
    # data is hard we actually only fit on the shape of the number density
    # profile. To do this we scale the number density
    # data to have the same mean value as the surface brightness data:

    # This minimizes chi^2
    # Sum of observed * model / observed**2
    # Divided by sum of model**2 / observed**2

    # Calculate scaling factor
    K = (np.sum(obs_Σ * interpolated / obs_Σ**2)
         / np.sum(interpolated**2 / obs_Σ**2))

    interpolated *= K

    # Now nuisance parameter
    # This allows us to add a constant error component to the data which
    # allows us to fit on the data while not worrying too much about the
    # outermost points where background effects are most prominent.

    yerr = np.sqrt(obs_err**2 + model.s2)

    # Now regular gaussian likelihood
    return -0.5 * np.sum(
        (obs_Σ - interpolated)**2 / yerr**2 + np.log(yerr**2)
    )


def likelihood_pm_tot(model, pm, *, mass_bin=None):

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    model_tot = np.sqrt(0.5 * (model.v2Tj[mass_bin] + model.v2Rj[mass_bin]))

    # Convert model units
    model_r = model.r.to(u.arcsec, util.angular_width(model.d))
    model_tot = model_tot.to(u.mas / u.yr, util.angular_speed(model.d))

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_tot', model_r, model_tot)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_tot)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_tot'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


def likelihood_pm_ratio(model, pm, *, mass_bin=None):

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Convert model units
    model_r = model.r.to(u.arcsec, util.angular_width(model.d))
    model_ratio = np.sqrt(model.v2Tj[mass_bin] / model.v2Rj[mass_bin])

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_ratio', model_r, model_ratio)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_ratio)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_ratio'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


def likelihood_pm_T(model, pm, *, mass_bin=None):

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    model_T = np.sqrt(model.v2Tj[mass_bin])

    # Convert model units
    model_r = model.r.to(u.arcsec, util.angular_width(model.d))
    model_T = model_T.to(u.mas / u.yr, util.angular_speed(model.d))

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_T', model_r, model_T)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_T)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_T'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


def likelihood_pm_R(model, pm, *, mass_bin=None):

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    model_R = np.sqrt(model.v2Rj[mass_bin])

    # Convert model units
    model_r = model.r.to(u.arcsec, util.angular_width(model.d))
    model_R = model_R.to(u.mas / u.yr, util.angular_speed(model.d))

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_R', model_r, model_R)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_R)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_R'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


def likelihood_LOS(model, vlos, *, mass_bin=None):

    if mass_bin is None:
        if 'm' in vlos.mdata:
            mass_bin = np.where(model.mj == vlos.mdata['m'] * u.Msun)[0][0]
        else:
            mass_bin = model.nms - 1

    # Convert model units
    model_r = model.r.to(u.arcsec, util.angular_width(model.d))
    model_LOS = np.sqrt(model.v2pj[mass_bin])

    # Build asymmetric error, if exists
    obs_err = vlos.build_err('σ', model_r, model_LOS)

    # Interpolated model at data locations
    interpolated = np.interp(vlos['r'], model_r, model_LOS)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (vlos['σ'] - interpolated)**2 / obs_err**2
        + np.log((obs_err / obs_err.unit)**2)
    )


def likelihood_mass_func(model, mf):

    tot_likelihood = 0

    for annulus_ind in np.unique(mf['bin']):

        # we only want to use the obs data for this r bin
        r_mask = (mf['bin'] == annulus_ind)

        r1 = 0.4 * annulus_ind.to(u.arcsec, util.angular_width(model.d))
        r2 = 0.4 * (annulus_ind + 1).to(u.arcsec, util.angular_width(model.d))

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
                integ.quad(density, r1, r2)[0]
                / (model.mj[mbin_ind] * model.mes_widths[mbin_ind])
            )

        # interpolate a func N_model = f(mean mass) from the binned N_model
        interp_N_model = interp.interp1d(
            model.mj[:model.nms], binned_N_model, fill_value="extrapolate"
        )
        # Grab the interpolated N_model's at the data mean masses
        N_model = interp_N_model(mf['mbin_mean'][r_mask])

        # Grab the N_data (adjusted by width to get an average
        #                   dr of a bin (like average-interpolating almost))
        N_data = (mf['N'][r_mask] / mf['mbin_width'][r_mask])

        # Compute δN_model from poisson error, and nuisance factor
        err = np.sqrt(mf['Δmbin'][r_mask]**2 + (model.F * N_data)**2)

        # compute final gaussian log likelihood
        tot_likelihood += -0.5 * np.sum(
            (N_data - N_model)**2 / err**2 + np.log((err / err.unit)**2)
        )

    return tot_likelihood


# --------------------------------------------------------------------------
# Composite likelihood functions
# --------------------------------------------------------------------------


def determine_components(obs):
    '''from observations, determine which likelihood functions will be computed
    and return a dict of the relevant obs dataset keys, and tuples of the
    functions and any other required args
    I really don't love this
    This should really go in data.Observations, at least
    '''

    comps = []
    for key in obs.datasets:

        # ------------------------------------------------------------------
        # Parse each key to determine if it matches with one of our
        # likelihood functions.
        # fnmatch is used to properly handle relevant subgroups
        # such as proper_motion/high_mass and etc, where they exist
        #
        # Each component is a tuple of where the first two elements are,
        # respectively, the observation key and likelihood function, and all
        # remaining elements are the extra arguments to pass to the function
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Pulsar likelihoods
        # ------------------------------------------------------------------

        if fnmatch.fnmatch(key, '*pulsar*'):

            mdata = obs.mdata['μ'], (obs.mdata['b'], obs.mdata['l'])

            if 'Pdot_meas' in obs[key]:

                kde = util.pulsar_Pdot_KDE()

                comps.append((key, likelihood_pulsar_spin, kde, *mdata))

            if 'Pb' in obs[key]:

                comps.append((key, likelihood_pulsar_orbital, *mdata))

        # ------------------------------------------------------------------
        # Line-of-sight velocity dispersion likelihoods
        # ------------------------------------------------------------------

        elif fnmatch.fnmatch(key, '*velocity_dispersion*'):

            comps.append((key, likelihood_LOS, ))

        # ------------------------------------------------------------------
        # Number density likelihoods
        # ------------------------------------------------------------------

        elif fnmatch.fnmatch(key, '*number_density*'):

            comps.append((key, likelihood_number_density, ))

        # ------------------------------------------------------------------
        # Proper motion dispersion likelihoods
        # ------------------------------------------------------------------

        elif fnmatch.fnmatch(key, '*proper_motion*'):

            if 'PM_tot' in obs[key]:
                comps.append((key, likelihood_pm_tot, ))

            if 'PM_ratio' in obs[key]:
                comps.append((key, likelihood_pm_ratio, ))

            if 'PM_R' in obs[key]:
                comps.append((key, likelihood_pm_R, ))

            if 'PM_T' in obs[key]:
                comps.append((key, likelihood_pm_T, ))

        # ------------------------------------------------------------------
        # Stellar mass function likelihoods
        # ------------------------------------------------------------------

        elif fnmatch.fnmatch(key, '*mass_function*'):

            comps.append((key, likelihood_mass_func, ))

    return comps


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
def posterior(theta, observations, fixed_initials, L_components):
    '''
    theta : array of theta values
    observations : data.Observations
    fixed_initials : dict of any theta values to fix
    L_components : output from determine_components
    '''

    # get a list of variable params, sorted for the unpacking of theta
    variable_params = DEFAULT_INITIALS.keys() - fixed_initials.keys()
    params = sorted(variable_params, key=list(DEFAULT_INITIALS).index)

    # Update to unions when 3.9 becomes enforced
    theta = dict(zip(params, theta), **fixed_initials)

    # TODO make this prettier
    # Prior probability function
    if not (3 < theta['W0'] < 20 and 0.5 < theta['rh'] < 15
            and 0.01 < theta['M'] < 10 and 0 < theta['ra'] < 5
            and 0 < theta['g'] < 2.3 and 0.3 < theta['delta'] < 0.5
            and 0 < theta['s2'] < 10 and 0.1 < theta['F'] < 0.5
            and -2 < theta['a1'] < 6 and -2 < theta['a2'] < 6
            and -2 < theta['a3'] < 6 and 0 < theta['BHret'] < 100
            and 4 < theta['d'] < 8):

        logging.debug("Theta outside priors domain")

        return -np.inf, *(-np.inf * np.ones(len(L_components)))

    probability, individuals = log_likelihood(theta, observations, L_components)

    return probability, *individuals
