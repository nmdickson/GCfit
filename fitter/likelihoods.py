from .data import A_SPACE
from .new_Paz import vec_Paz

import numpy as np
import limepy as lp
import scipy.signal
import scipy.integrate as integ
import scipy.interpolate as interp
from ssptools import evolve_mf_3 as emf3

import sys
import logging

# TODO figure out all these mass bins and why functions choose certain ones


# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------

def pc2arcsec(r, d):
    # d = 4.45 * 1000  # pc
    d *= 1000
    return 206265 * 2 * np.arctan(r / (2 * d))


def as2pc(theta, d):
    # d = 4.45 * 1000  # pc
    d *= 1000
    return np.tan(theta * 1 / 3600 * np.pi / 180 / 2) * 2 * d


def kms2masyr(kms, d):
    kmyr = kms * 3.154e7
    pcyr = kmyr * 3.24078e-14
    asyr = pc2arcsec(pcyr, d)
    masyr = 1000 * asyr
    return masyr


# --------------------------------------------------------------------------
# Asymmetric error constructions
# --------------------------------------------------------------------------

# TODO these should be build into data probably?
# TODO be careful of the units here maybe

# We have two datasets with asymmetric error, these functions allow us to choose
# either the upper or lower error bound depending on where our test
# point is in relation to the observed point.
def build_asym_err(model, r, quantity, sigmaup, sigmalow, d):
    sigma = np.zeros(len(sigmaup))
    interpolated = np.interp(r, pc2arcsec(model.r, d), np.sqrt(model.v2p))
    for i in range(len(r)):
        if interpolated[i] > quantity[i]:
            sigma[i] = sigmaup[i]
        else:
            sigma[i] = sigmalow[i]
    return sigma


# --------------------------------------------------------------------------
# Component likelihood functions
# --------------------------------------------------------------------------

# Calculates the likelihood from pulsars
def likelihood_pulsars(model, pulsars, error_dist, return_dist=False):

    # Generate the probability distributions
    def gen_prob_dists(model, a_space, r_data, jns):
        dists = []
        # Create a distribution at each r value that we have a pulsar
        for i in range(len(r_data)):
            # vec_Paz allows us to run get_Paz() over an
            # array of az points without looping through
            # calculations that are specific only to the r point.
            prob_dist = vec_Paz(
                model=model, az_data=a_space, R_data=r_data[i],
                jns=jns, current_pulsar=i,
            )
            dists.append(prob_dist)
        return dists

    # Convolve the gaussian errors and the generated probability distributions
    def gen_convolved_dists(error, prob):

        # The ordering of the input arrays here is actually important:
        # see comment in likelihood_pulsars()
        dists = []
        for i in range(len(prob)):
            conved = scipy.signal.convolve(error[i], prob[i], mode="same")

            # The convolution method here keeps the same number of points which
            # is important for our later interpolations but it doesn't normalize
            # the resulting function. Here we set up a spline to manually
            # normalize the area under the function to 1. This removes most of
            # the effects of the somewhat inconsistent peaks in the probability
            # distribution while maintaining the overall shape.
            spl = interp.UnivariateSpline(A_SPACE, conved, k=3, s=0, ext=1)
            conved /= spl.integral(-15e-9, 15e-9)

            dists.append(conved)
        return dists

    # Generate the probability distributions that we will convolve with the
    #   pre-generated error distributions
    pre_prob_dist = gen_prob_dists(model, A_SPACE, pulsars['r'], -1)

    # The ordering of the arrays here is actually important: In the case of a
    # z2 interpolation error the probability value will be dropped so in some
    # cases the prob dist will be shorter than the a_space. This convolution
    # returns an array of the same length as the first input so by supplying
    # the error dist first (which is built with the a_space) we can ensure
    # that the convolved dist will be the same length as the a_space. This
    # is needed for the interpolation below.
    prob_dist = gen_convolved_dists(error_dist, pre_prob_dist)

    # For each distribution we want to interpolate the probability from the
    #   corresponding a_los measurement
    probs = np.zeros(len(pulsars['r']))
    # Select the corresponding distributions
    for i in range(len(pulsars['r'])):
        # Interpolate the probability value from the convolved distributions
        interpolated = interp.interp1d(A_SPACE, prob_dist[i])
        # evaluate at the measured a_los
        probs[i] = interpolated(pulsars['a_los'][i])

    if return_dist:
        return prob_dist
    else:
        # Multiply all the probabilities and return the total log probability.
        return np.log(np.prod(probs))


# Calculates likelihood from number density data.
def likelihood_number_density(model, ndensity, mass_bin, s, d):

    # Interpolated the model data at the measurement locations
    interpolated = np.interp(
        ndensity['r'], pc2arcsec(model.r, d) / 60,
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
    K = (np.sum(ndensity['Σ'] * interpolated / ndensity['Σ'] ** 2)
         / np.sum(interpolated ** 2 / ndensity['Σ'] ** 2))

    # Apply scaling factor to interpolated points
    interpolated *= K

    # Now nuisance parameter
    # This allows us to add a constant error component to the data which
    # allows us to fit on the data while not worrying too much about the
    # outermost points where background effects are most prominent.
    yerr = np.zeros(len(ndensity['ΔΣ']))

    # Add the nuisance parameter in quadrature
    for i in range(len(ndensity['ΔΣ'])):
        yerr[i] = np.sqrt(ndensity['ΔΣ'][i] ** 2 + s)

    # Now regular gaussian likelihood
    return -0.5 * np.sum((ndensity['Σ'] - interpolated) ** 2 / yerr ** 2
                         + np.log(yerr ** 2))


def likelihood_pm_tot(model, pm, mass_bin, d):
    # TODO is v2j what I should use for tot?

    # Build asymmetric error, if exists
    try:
        obs_err = build_asym_err(model, pm['r'], pm['PM_tot'],
                                 pm['ΔPM_tot,up'], pm['ΔPM_tot,down'], d)
    except KeyError:
        obs_err = pm['ΔPM_tot']

    # Interpolated model at data locations
    interpolated = np.interp(
        pm['r'], pc2arcsec(model.r, d),
        kms2masyr(np.sqrt(model.v2j[mass_bin]), d)
    )

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_tot'] - interpolated) ** 2 / obs_err ** 2
        + np.log(obs_err ** 2)
    )


def likelihood_pm_ratio(model, pm, mass_bin, d):

    model_ratio = np.sqrt(model.v2Tj[mass_bin] / model.v2Rj[mass_bin])

    # Build asymmetric error, if exists
    try:
        obs_err = build_asym_err(model, pm['r'], pm['PM_ratio'],
                                 pm['ΔPM_ratio,up'], pm['ΔPM_ratio,down'], d)
    except KeyError:
        obs_err = pm['ΔPM_ratio']

    # Interpolated model at data locations
    interpolated = np.interp(
        pm['r'], pc2arcsec(model.r, d),
        model_ratio
    )

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_tot'] - interpolated) ** 2 / obs_err ** 2
        + np.log(obs_err ** 2)
    )


def likelihood_pm_T(model, r, pm, mass_bin, d):

    # Build asymmetric error, if exists
    try:
        obs_err = build_asym_err(model, pm['r'], pm['PM_T'],
                                 pm['ΔPM_T,up'], pm['ΔPM_T,down'], d)
    except KeyError:
        obs_err = pm['ΔPM_T']

    # Interpolated model at data locations
    interpolated = np.interp(
        pm['r'], pc2arcsec(model.r, d),
        kms2masyr(np.sqrt(model.v2Tj[mass_bin]), d)
    )

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_T'] - interpolated) ** 2 / obs_err ** 2
        + np.log(obs_err ** 2)
    )


def likelihood_pm_R(model, r, pm, mass_bin, d):

    # Build asymmetric error, if exists
    try:
        obs_err = build_asym_err(model, pm['r'], pm['PM_R'],
                                 pm['ΔPM_R,up'], pm['ΔPM_R,down'], d)
    except KeyError:
        obs_err = pm['ΔPM_R']

    # Interpolated model at data locations
    interpolated = np.interp(
        pm['r'], pc2arcsec(model.r, d),
        kms2masyr(np.sqrt(model.v2Rj[mass_bin]), d)
    )

    # Gaussian likelihood
    return -0.5 * np.sum(
        (pm['PM_R'] - interpolated) ** 2 / obs_err ** 2
        + np.log(obs_err ** 2)
    )


def likelihood_LOS(model, vlos, mass_bin, d):
    # most massive main-sequence bin, mass_bin

    # Build asymmetric error, if exists
    try:
        obs_err = build_asym_err(model, vlos['r'], vlos['σ'],
                                 vlos['Δσ_up'], vlos['Δσ_down'], d)
    except KeyError:
        obs_err = vlos['Δσ']

    # Interpolated model at data locations
    interpolated = np.interp(
        vlos['r'], pc2arcsec(model.r, d),
        np.sqrt(model.v2pj[mass_bin])
    )

    # Gaussian likelihood
    return -0.5 * np.sum(
        (vlos['σ'] - interpolated) ** 2 / obs_err ** 2
        + np.log(obs_err ** 2)
    )


def likelihood_mf_tot(model, mf, N_ms, mes_widths, F, d):

    tot_likelihood = 0

    for annulus_ind in np.unique(mf['bin']):

        # we only want to use the obs data for this r bin
        r_mask = (mf['bin'] == annulus_ind)

        r1, r2 = as2pc(0.4 * annulus_ind, d), as2pc(0.4 * (annulus_ind + 1), d)

        # Get a binned version of N_model (an Nstars for each mbin)
        binned_N_model = np.empty(N_ms)
        for mbin_ind in range(N_ms):

            # Interpolate the model density at the data locations
            density = interp.interp1d(
                model.r, 2 * np.pi * model.r * model.Sigmaj[mbin_ind],
                kind="cubic"
            )

            # Convert density spline into Nstars
            binned_N_model[mbin_ind] = (
                integ.quad(density, r1, r2)[0]
                / (model.mj[mbin_ind] * mes_widths[mbin_ind])
            )

        # interpolate a func N_model = f(mean mass) from the binned N_model
        interp_N_model = interp.interp1d(
            model.mj[:N_ms], binned_N_model, fill_value="extrapolate"
        )
        # Grab the interpolated N_model's at the data mean masses
        N_model = interp_N_model(mf['mbin_mean'][r_mask])

        # Grab the N_data (adjusted by width to get an average
        #                   dr of a bin (like average-interpolating almost))
        N_data = (mf['N'][r_mask] / mf['mbin_width'][r_mask])

        # Compute δN_model from poisson error, and nuisance factor
        err = np.sqrt(mf['Δmbin'][r_mask]**2
                      + (F * N_data / mf['mbin_width'][r_mask]**2))

        # compute final gaussian (log?) likelihood
        tot_likelihood += (-0.5 * np.sum((N_data - N_model)**2
                                         / err**2 + np.log(err**2)))

    return tot_likelihood


# --------------------------------------------------------------------------
# Composite likelihood functions
# --------------------------------------------------------------------------

# Log prior is what we use to define what regions of parameter space we
#   will consider valid.
# TODO how do we know these ranges, do they change per cluster, are they good
def log_prior(theta):
    W0, M, rh, ra, g, delta, s, F, a1, a2, a3, BHret, d = theta
    if (
        3 < W0 < 20
        and 0.5 < rh < 15
        and 0.01 < M < 10
        and 0 < ra < 5
        and 0 < g < 2.3
        and 0.3 < delta < 0.5
        and 0 < s < 10
        and 0.1 < F < 0.5
        and -2 < a1 < 6
        and -2 < a2 < 6
        and -2 < a3 < 6
        and 0 < BHret < 100
        and 4 < d < 7
    ):
        # If its within the valid space don't add anything
        return 0.0
    # If its outside the valid space add -inf to prevent further movement
    #   in that direction
    return -np.inf


# Combines the likelihood with the prior
def log_probability(theta, observations, error_dist):
    lp = log_prior(theta)
    # This line was inserted while debugging, may not be needed anymore.
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, observations, error_dist)


# Main likelihood function, generates the model(theta) passes it to the
# individual likelihood functions and collects their results.
def log_likelihood(theta, observations, pulsar_edist):

    # Construct the model with current theta (parameters)
    W0, M, rh, ra, g, delta, s2, F, a1, a2, a3, BHret, d = theta

    m123 = [0.1, 0.5, 1.0, 100]  # Slope breakpoints for initial mass function
    a12 = [-a1, -a2, -a3]  # Slopes for initial mass function
    nbin12 = [5, 5, 20]

    # Output times for the evolution
    tout = np.array([11000])

    # Integration settings
    N0 = 5e5  # Normalization of stars
    Ndot = 0  # Regulates low mass objects depletion, default -20, 0 for 47 Tuc
    tcc = 0  # Core collapse time
    NS_ret = 0.1  # Initial neutron star retention
    BH_ret_int = 1  # Initial Black Hole retention
    BH_ret_dyn = BHret / 100  # Dynamical Black Hole retention
    FeHe = -1.02  # Metallicity

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

    # Collect bin-widths
    # mes_widths = mass_func.mes[-1][1:] - mass_func.mes[-1][:-1]

    # Get main-sequence turnoff bin
    nms = len(mass_func.ms[-1][cs])

    # Add in bin of 0.38 MSol to model Heyl data
    # mj = np.append(mj, 0.38)
    # Mj = np.append(Mj, 0.1)

    # Add in bin of 1.6 MSol to use to model MSPs
    # mj = np.append(mj, 1.6)
    mj = np.append(mj, 1.4)  # for M62 (See lynch, table3)
    Mj = np.append(Mj, 0.1)

    # In the event that a limepy model does not converge, return -inf.
    try:
        model = lp.limepy(
            phi0=W0,
            g=g,
            M=M * 1e6,
            rh=rh,
            ra=10 ** ra,
            delta=delta,
            mj=mj,
            Mj=Mj,
            project=True,
            verbose=False,
        )
    except Exception:
        e = str(sys.exc_info()[0]) + " : " + str(sys.exc_info()[1])
        print("INFO: Exception raised by limepy, returning -np.inf. ", e)
        return -np.inf

    # If the model does not converge return -np.inf
    if not model.converged:
        logging.debug("Model ({model}) did not converge")
        return -np.inf

    # Calculate each log likelihood

    # TODO need to change how calling all L, which ones used depends on cluster

    log_pulsar = likelihood_pulsars(
        model,
        observations['pulsar'],
        pulsar_edist
    )

    log_LOS = likelihood_LOS(
        model,
        observations['velocity_dispersion'],
        nms - 1,
        d
    )

    log_numdens = likelihood_number_density(
        model,
        observations['number_density'],
        nms - 1,
        s2,
        d
    )

    log_pm_tot = likelihood_pm_tot(
        model,
        observations['proper_motion'],
        nms - 1,
        d,
    )

    log_pm_ratio = likelihood_pm_ratio(
        model,
        observations['proper_motion'],
        nms - 1,
        d,
    )

    # log_pmR_high = likelihood_pm_R(
    #     model,
    #     observations['proper_motion/high_mass'],
    #     nms - 1,
    #     d,
    # )

    # log_pmT_high = likelihood_pm_T(
    #     model,
    #     observations['proper_motion/high_mass'],
    #     nms - 1,
    #     d,
    # )

    # log_pmR_low = likelihood_pm_R(
    #     model,
    #     observations['proper_motion/low_mass'],
    #     -2,
    #     d,
    # )

    # log_pmT_low = likelihood_pm_T(
    #     model,
    #     observations['proper_motion/low_mass'],
    #     -2,
    #     d,
    # )

    # log_mf = likelihood_mf_tot(
    #     model,
    #     observations['mass_function'],
    #     nms,
    #     mes_widths,
    #     F,
    #     d
    # )

    return (
        log_pulsar
        + log_LOS
        + log_numdens
        + log_pm_tot
        + log_pm_ratio
        # + log_pmR_high
        # + log_pmT_high
        # + log_pmR_low
        # + log_pmT_low
        # + log_mf
    )
