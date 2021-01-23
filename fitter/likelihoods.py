from .data import DEFAULT_INITIALS
from .new_Paz import vec_Paz

import numpy as np
import limepy as lp
import scipy.stats
import scipy.signal
import scipy.integrate as integ
import scipy.interpolate as interp
from ssptools import evolve_mf_3 as emf3

import logging
import fnmatch


# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------


def pc2arcsec(r, d):
    d *= 1000
    return 206265 * 2 * np.arctan(r / (2 * d))


def as2pc(theta, d):
    d *= 1000
    return np.tan(theta * 1 / 3600 * np.pi / 180 / 2) * 2 * d


def kms2masyr(kms, d):
    kmyr = kms * 3.154e7
    pcyr = kmyr * 3.24078e-14
    asyr = pc2arcsec(pcyr, d)
    masyr = 1000 * asyr
    return masyr


# --------------------------------------------------------------------------
# Component likelihood functions
# --------------------------------------------------------------------------


def likelihood_pulsar(model, pulsars, mass_bin=None):

    # Simple gaussian implementation
    def gaussian(x, sigma, mu):
        norm = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = np.exp(-0.5 * (((x - mu) / sigma) ** 2))
        return norm * exponent

    if mass_bin is None:
        if 'm' in pulsars.mdata:
            mass_bin = np.where(model.mj == pulsars.mdata['m'])[0][0]
        else:
            logging.debug("No mass bin provided for pulsars, using -1")
            mass_bin = -1

    # Generate the probability distributions that we will convolve with the
    #   pre-generated error distributions
    # pre_prob_dist = gen_prob_dists(model, A_SPACE, pulsars['r'], mass_bin)

    accel_domains, accel_prob_dists = zip(*[
        vec_Paz(model=model, R=R, mass_bin=mass_bin) for R in pulsars['r']
    ])

    error_dists = [
        gaussian(x=accel_domains[i], sigma=width, mu=0)
        for i, width in enumerate(pulsars['Δa_los'])
    ]

    prob_dists = []
    for i in range(len(accel_prob_dists)):
        conved = scipy.signal.convolve(error_dists[i], accel_prob_dists[i],
                                       mode="same")

        spl = interp.UnivariateSpline(accel_domains[i], conved, k=3, s=0, ext=1)
        conved /= spl.integral(accel_domains[i].min(), accel_domains[i].max())

        prob_dists.append(conved)

    # For each distribution we want to interpolate the probability from the
    #   corresponding a_los measurement
    probs = np.zeros(len(pulsars['r']))

    # Select the corresponding distributions
    for i in range(len(pulsars['r'])):

        # Interpolate the probability value from the convolved distributions
        interpolated = interp.interp1d(
            accel_domains[i], prob_dists[i], assume_sorted=True,
            bounds_error=False, fill_value=0.0
        )

        # evaluate at the measured a_los
        probs[i] = interpolated(pulsars['a_los'][i])

    # Multiply all the probabilities and return the total log probability.
    return np.log(np.prod(probs))


def likelihood_number_density(model, ndensity, mass_bin=None):
    # TODO don't forget to revert or better compute this flatness cutoff [:100]

    if mass_bin is None:
        if 'm' in ndensity.mdata:
            mass_bin = np.where(model.mj == ndensity.mdata['m'])[0][0]
        else:
            mass_bin = model.nms - 1

    # Interpolated the model data at the measurement locations
    interpolated = np.interp(
        ndensity['r'][:100], pc2arcsec(model.r, model.d) / 60,
        model.Sigmaj[mass_bin] / model.mj[mass_bin],
    )

    # K Scaling Factor

    # Because the translation between number density and surface-brightness
    # data is hard we actually only fit on the shape of the number density
    # profile. To do this we scale the number density
    # data to have the same mean value as the surface brightness data:

    # This minimizes chi^2
    # Sum of observed * model / observed**2
    # Divided by sum of model**2 / observed**2

    # Calculate scaling factor
    K = (np.sum(ndensity['Σ'][:100] * interpolated / ndensity['Σ'][:100] ** 2)
         / np.sum(interpolated ** 2 / ndensity['Σ'][:100] ** 2))

    # Apply scaling factor to interpolated points
    interpolated *= K

    # Now nuisance parameter
    # This allows us to add a constant error component to the data which
    # allows us to fit on the data while not worrying too much about the
    # outermost points where background effects are most prominent.

    # TODO could ΔΣ ever be asymmetric? I doubt it right?
    yerr = np.zeros(len(ndensity['ΔΣ'][:100]))

    # Add the nuisance parameter in quadrature
    for i in range(len(ndensity['ΔΣ'][:100])):
        yerr[i] = np.sqrt(ndensity['ΔΣ'][:100][i] ** 2 + model.s2)

    # Now regular gaussian likelihood
    return -0.5 * np.sum((ndensity['Σ'][:100] - interpolated) ** 2 / yerr ** 2
                         + np.log(yerr ** 2))


def likelihood_pm_tot(model, pm, mass_bin=None):

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'])[0][0]
        else:
            mass_bin = model.nms - 1

    model_tot = np.sqrt(0.5 * (model.v2Tj[mass_bin] + model.v2Rj[mass_bin]))

    # Convert model units
    model_r = pc2arcsec(model.r, model.d)
    model_tot = kms2masyr(model_tot, model.d)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_tot', model_r, model_tot)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_tot)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_tot'] - interpolated) ** 2 / obs_err ** 2 + np.log(obs_err ** 2)
    )


def likelihood_pm_ratio(model, pm, mass_bin=None):

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'])[0][0]
        else:
            mass_bin = model.nms - 1

    # Convert model units
    model_r = pc2arcsec(model.r, model.d)
    model_ratio = np.sqrt(model.v2Tj[mass_bin] / model.v2Rj[mass_bin])

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_ratio', model_r, model_ratio)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_ratio)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_ratio'] - interpolated) ** 2 / obs_err ** 2
        + np.log(obs_err ** 2)
    )


def likelihood_pm_T(model, pm, mass_bin=None):

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'])[0][0]
        else:
            mass_bin = model.nms - 1

    # Convert model units
    model_r = pc2arcsec(model.r, model.d)
    model_T = kms2masyr(np.sqrt(model.v2Tj[mass_bin]), model.d)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_T', model_r, model_T)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_T)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_T'] - interpolated) ** 2 / obs_err ** 2 + np.log(obs_err ** 2)
    )


def likelihood_pm_R(model, pm, mass_bin=None):

    if mass_bin is None:
        if 'm' in pm.mdata:
            mass_bin = np.where(model.mj == pm.mdata['m'])[0][0]
        else:
            mass_bin = model.nms - 1

    # Convert model units
    model_r = pc2arcsec(model.r, model.d)
    model_R = kms2masyr(np.sqrt(model.v2Rj[mass_bin]), model.d)

    # Build asymmetric error, if exists
    obs_err = pm.build_err('PM_R', model_r, model_R)

    # Interpolated model at data locations
    interpolated = np.interp(pm['r'], model_r, model_R)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_R'] - interpolated) ** 2 / obs_err ** 2 + np.log(obs_err ** 2)
    )


def likelihood_LOS(model, vlos, mass_bin=None):

    if mass_bin is None:
        if 'm' in vlos.mdata:
            mass_bin = np.where(model.mj == vlos.mdata['m'])[0][0]
        else:
            mass_bin = model.nms - 1

    # Convert model units
    model_r = pc2arcsec(model.r, model.d)
    model_LOS = np.sqrt(model.v2pj[mass_bin])

    # Build asymmetric error, if exists
    obs_err = vlos.build_err('σ', model_r, model_LOS)

    # Interpolated model at data locations
    interpolated = np.interp(vlos['r'], model_r, model_LOS)

    # Gaussian likelihood
    return -0.5 * np.sum(
        (vlos['σ'] - interpolated) ** 2 / obs_err ** 2 + np.log(obs_err ** 2)
    )


def likelihood_mass_func(model, mf):

    tot_likelihood = 0

    for annulus_ind in np.unique(mf['bin']):

        # we only want to use the obs data for this r bin
        r_mask = (mf['bin'] == annulus_ind)

        r1 = as2pc(60 * 0.4 * annulus_ind, model.d)
        r2 = as2pc(60 * 0.4 * (annulus_ind + 1), model.d)

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
        tot_likelihood += (-0.5 * np.sum((N_data - N_model)**2
                                         / err**2 + np.log(err**2)))

    return tot_likelihood


# --------------------------------------------------------------------------
# Composite likelihood functions
# --------------------------------------------------------------------------


def create_model(theta, observations=None, *, strict=False, verbose=False):
    '''if observations are provided, will use it for a few parameters and
    check if need to add any mass bins. Otherwise will just use defaults.
    '''

    # Construct the model with current theta (parameters)
    if isinstance(theta, dict):
        W0, M, rh, ra, g, delta, s2, F, a1, a2, a3, BHret, d = (
            theta['W0'], theta['M'], theta['rh'], theta['ra'], theta['g'],
            theta['delta'], theta['s2'], theta['F'],
            theta['a1'], theta['a2'], theta['a3'], theta['BHret'], theta['d']
        )
    else:
        W0, M, rh, ra, g, delta, s2, F, a1, a2, a3, BHret, d = theta

    m123 = [0.1, 0.5, 1.0, 100]  # Slope breakpoints for initial mass function
    a12 = [-a1, -a2, -a3]  # Slopes for initial mass function
    nbin12 = [5, 5, 20]

    # Output times for the evolution (age)
    tout = np.array([11000])

    # TODO figure out which of these are cluster dependant, store them in files
    # Integration settings
    N0 = 5e5  # Normalization of stars
    tcc = 0  # Core collapse time
    NS_ret = 0.1  # Initial neutron star retention
    BH_ret_int = 1  # Initial Black Hole retention
    BH_ret_dyn = BHret / 100  # Dynamical Black Hole retention

    # Metallicity
    try:
        FeHe = observations.mdata['FeHe']
    except (AttributeError, KeyError):
        FeHe = -1.02

    # Regulates low mass objects depletion, default -20, 0 for 47 Tuc
    try:
        Ndot = observations.mdata['Ndot']
    except (AttributeError, KeyError):
        Ndot = 0

    # Generate the mass function
    mass_func = emf3.evolve_mf(
        m123=m123,
        a12=a12,
        nbin12=nbin12,
        tout=tout,
        N0=N0,
        Ndot=Ndot,
        tcc=tcc,
        NS_ret=NS_ret,
        BH_ret_int=BH_ret_int,
        BH_ret_dyn=BH_ret_dyn,
        FeHe=FeHe,
    )

    # Set bins that should be empty to empty
    cs = mass_func.Ns[-1] > 10 * mass_func.Nmin
    cr = mass_func.Nr[-1] > 10 * mass_func.Nmin

    # Collect mean mass and total mass bins
    mj = np.r_[mass_func.ms[-1][cs], mass_func.mr[-1][cr]]
    Mj = np.r_[mass_func.Ms[-1][cs], mass_func.Mr[-1][cr]]

    # append tracer mass bins (must be appended to end to not affect nms)

    if observations is not None:

        tracer_mj = [
            dataset.mdata['m'] for dataset in observations.datasets.values()
            if 'm' in dataset.mdata
        ]

        mj = np.concatenate((mj, tracer_mj))
        Mj = np.concatenate((Mj, 0.1 * np.ones_like(tracer_mj)))

    # In the event that a limepy model does not converge, return -inf.
    try:
        model = lp.limepy(
            phi0=W0,
            g=g,
            M=M * 1e6,
            rh=rh,
            ra=10**ra,
            delta=delta,
            mj=mj,
            Mj=Mj,
            project=True,
            verbose=verbose,
        )
    except ValueError as err:
        logging.debug(f"Model did not converge with {theta=}")
        if strict:
            raise ValueError(err)
        else:
            return None

    # TODO not my favourite way to store this info
    #   means models *have* to be created here for the most part

    # store some necessary mass function info in the model
    model.nms = len(mass_func.ms[-1][cs])
    model.mes_widths = mass_func.mes[-1][1:] - mass_func.mes[-1][:-1]

    # store some necessary theta parameters in the model
    model.d = d
    model.F = F
    model.s2 = s2

    return model


def determine_components(obs):
    '''from observations, determine which likelihood functions will be computed
    and return a dict of the relevant obs dataset keys, and tuples of the
    functions and any other required args
    I really don't love this
    '''

    L_components = []
    for key in obs.datasets:

        # fnmatch is to correctly find subgroup stuff like pm/high_mass, etc
        if fnmatch.fnmatch(key, '*pulsar*'):

            # now that we need to do this every time, its actually quicker
            # to just compute it manually
            # Well actually this may change once we are also pulsar-vectorized
            # a_width = np.abs(obs[key]['Δa_los'])
            # pulsar_edist = scipy.stats.norm.pdf(A_SPACE, 0, np.c_[a_width])

            L_components.append((key, likelihood_pulsar))

        elif fnmatch.fnmatch(key, '*velocity_dispersion*'):
            L_components.append((key, likelihood_LOS, ))

        elif fnmatch.fnmatch(key, '*number_density*'):
            L_components.append((key, likelihood_number_density, ))

        elif fnmatch.fnmatch(key, '*proper_motion*'):
            if 'PM_tot' in obs[key]:
                L_components.append((key, likelihood_pm_tot, ))

            if 'PM_ratio' in obs[key]:
                L_components.append((key, likelihood_pm_ratio, ))

            if 'PM_R' in obs[key]:
                L_components.append((key, likelihood_pm_R, ))

            if 'PM_T' in obs[key]:
                L_components.append((key, likelihood_pm_T, ))

        elif fnmatch.fnmatch(key, '*mass_function*'):
            L_components.append((key, likelihood_mass_func, ))

    return L_components


# Main likelihood function, generates the model(theta) passes it to the
# individual likelihood functions and collects their results.
def log_likelihood(theta, observations, L_components):

    # TODO Having this as a try/excpt might be better than returning None
    model = create_model(theta, observations)

    # If the model does not converge, return -np.inf
    if model is None or not model.converged:
        return -np.inf, -np.inf * np.ones(len(L_components))

    # Calculate each log likelihood

    probs = np.array([
        likelihood(model, observations[key], *args)
        for (key, likelihood, *args) in L_components
    ])

    return sum(probs), probs


# Combines the likelihood with the prior
def posterior(theta, observations, fixed_initials, L_components):

    # get a list of variable params, sorted for the unpacking of theta
    variable_params = DEFAULT_INITIALS.keys() - fixed_initials.keys()
    params = sorted(variable_params, key=list(DEFAULT_INITIALS).index)

    # Update to unions when 3.9 becomes enforced
    theta = dict(zip(params, theta), **fixed_initials)

    # TODO make this prettier
    # Prior probability function
    if not (3 < theta['W0'] < 20
            and 0.5 < theta['rh'] < 15
            and 0.01 < theta['M'] < 10
            and 0 < theta['ra'] < 5
            and 0 < theta['g'] < 2.3
            and 0.3 < theta['delta'] < 0.5
            and 0 < theta['s2'] < 10
            and 0.1 < theta['F'] < 0.5
            and -2 < theta['a1'] < 6
            and -2 < theta['a2'] < 6
            and -2 < theta['a3'] < 6
            and 0 < theta['BHret'] < 100
            and 4 < theta['d'] < 8):
        return -np.inf, *(-np.inf * np.ones(len(L_components)))

    probability, individuals = log_likelihood(theta, observations, L_components)

    return probability, *individuals
