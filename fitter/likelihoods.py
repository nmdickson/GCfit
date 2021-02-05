from .data import DEFAULT_INITIALS
from .new_Paz import vec_Paz

import numpy as np
import limepy as lp
import astropy.units as u
import scipy.stats
import scipy.signal
import scipy.integrate as integ
import scipy.interpolate as interp
from ssptools import evolve_mf_3 as emf3

import logging
import fnmatch
import pathlib
from importlib import resources


# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------


def pc2arcsec(r, d):
    '''Convert r from [pc] to [arcsec]. `d` must be given in [kpc]'''
    d *= 1000
    return 206265 * 2 * np.arctan(r / (2 * d))


def as2pc(theta, d):
    '''Convert theta from [as] to [pc]. `d` must be given in [kpc]'''
    d *= 1000
    return np.tan(theta * 1 / 3600 * np.pi / 180 / 2) * 2 * d


def kms2masyr(kms, d):
    '''Convert kms from [kms] to [masyr `d` must be given in [kpc]'''
    kmyr = kms * 3.154e7
    pcyr = kmyr * 3.24078e-14
    asyr = pc2arcsec(pcyr, d)
    masyr = 1000 * asyr
    return masyr


def masyr2kms(masyr, d):
    '''Convert masyr from [masyr] to [kms]. `d` must be given in [kpc]'''
    asyr = masyr / 1000.
    pcyr = as2pc(asyr, d)
    kmyr = pcyr / 3.24078e-14
    kms = kmyr / 3.154e7
    return kms

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------


# Simple gaussian implementation
def _gaussian(x, sigma, mu):
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * (((x - mu) / sigma) ** 2))
    return norm * exponent


def _galactic_pot(lat, lon, D):
    '''b, l, d'''
    import gala.potential as pot

    from astropy.constants import c
    from astropy.coordinates import SkyCoord

    # Mikly Way Potential
    mw = pot.BovyMWPotential2014()

    # TODO chck that these dont already have units (or maybe require them first)
    # Pulsar position in galactocentric coordinates
    b_pulsar, l_pulsar, D_pulsar = lat * u.deg, lon * u.deg, D * u.kpc

    crd = SkyCoord(b=b_pulsar, l=l_pulsar, distance=D_pulsar, frame='galactic')
    XYZ = crd.galactocentric.cartesian.xyz

    # Sun position in galactocentric coordinates
    b_sun = np.zeros_like(lat) * u.deg
    l_sun = np.zeros_like(lon) * u.deg
    D_sun = np.zeros_like(D) * u.kpc

    sun = SkyCoord(b=b_sun, l=l_sun, distance=D_sun, frame='galactic')
    XYZ_sun = sun.galactocentric.cartesian.xyz

    PdotP = mw.acceleration(XYZ).si / c

    # scalar projection of PdotP along the position vector from pulsar to sun
    LOS = XYZ_sun - XYZ
    # PdotP_LOS = np.dot(PdotP, LOS) / np.linalg.norm(LOS)
    PdotP_LOS = np.einsum('i...,i...->...', PdotP, LOS) / np.linalg.norm(LOS)

    return PdotP_LOS


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
    # TODO dont use value, make everything else be units
    Pdot_int = Pdot_pm - _galactic_pot(lat, lon, D).value

    # TODO logging might not be necessary, if use a logspace later or something
    P = np.log10(P)
    Pdot_int = np.log10(Pdot_int)

    # TODO this shouldnt be necessary but sometimes gal > pm -> neg Pdot_int
    #   Check the new gal_pot against old Phinney, this never used to happen
    finite = np.isfinite(Pdot_int)

    # Create Gaussian P-Pdot_int KDE
    return scipy.stats.gaussian_kde(np.vstack([P[finite], Pdot_int[finite]]))


# --------------------------------------------------------------------------
# Component likelihood functions
# --------------------------------------------------------------------------


def likelihood_pulsar(model, pulsars, Pdot_kde, cluster_μ, coords, *,
                      mass_bin=None):

    c = 299_792_458  # m/s

    # ----------------------------------------------------------------------
    # Get the pulsar P-Pdot_int kde
    # ----------------------------------------------------------------------

    if Pdot_kde is None:
        Pdot_kde = pulsar_Pdot_KDE()

    Pdot_min, Pdot_max = Pdot_kde.dataset[1].min(), Pdot_kde.dataset[1].max()

    # ----------------------------------------------------------------------
    # Get pulsar mass bins
    # ----------------------------------------------------------------------

    if mass_bin is None:
        if 'm' in pulsars.mdata:
            mass_bin = np.where(model.mj == pulsars.mdata['m'])[0][0]
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

        R_i = pulsars['r'][i]

        P_i = pulsars['P'][i]
        ΔP_i = pulsars['ΔP'][i]

        Pdot_meas_i = pulsars['Pdot_meas'][i]
        ΔPdot_meas_i = pulsars['ΔPdot_meas'][i]

        # ------------------------------------------------------------------
        # Create a slice of the P-Pdot space, along this pulsars P
        # ------------------------------------------------------------------

        lg_P = np.log10(P_i)

        # TODO might want to max the grid at like -15, depending on conv tests
        P_i_grid, Pdot_int_grid = np.mgrid[lg_P:lg_P:1j, Pdot_min:Pdot_max:100j]

        # ------------------------------------------------------------------
        # Compute the Pdot_int distribution from the KDE
        # ------------------------------------------------------------------

        Pdot_int_prob_i = np.reshape(
            Pdot_kde(np.vstack([P_i_grid.ravel(), Pdot_int_grid.ravel()])),
            P_i_grid.shape
        )[0]

        # Normalize
        Pdot_int_prob_i /= interp.UnivariateSpline(
            Pdot_int_grid.ravel(), Pdot_int_prob_i, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        # ------------------------------------------------------------------
        # Compute the cluster component distribution of Pdot, from the model
        # ------------------------------------------------------------------

        a_domain, Pdot_c_prob_i = vec_Paz(model=model, R=R_i, mass_bin=mass_bin)
        Pdot_domain = a_domain / c

        # Normalize
        Pdot_c_prob_i /= interp.UnivariateSpline(
            Pdot_domain, Pdot_c_prob_i, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        # ------------------------------------------------------------------
        # Compute gaussian measurement error distribution
        # ------------------------------------------------------------------

        width = np.sqrt(ΔPdot_meas_i**2 * (1 / P_i)**2
                        + ΔP_i**2 * (-Pdot_meas_i / P_i**2)**2)
        err_i = _gaussian(x=Pdot_domain, sigma=width, mu=0)

        # ------------------------------------------------------------------
        # Convolve the different distributions
        # ------------------------------------------------------------------

        conv1 = np.convolve(err_i, Pdot_c_prob_i, 'same')

        # Normalize
        conv1 /= interp.UnivariateSpline(
            Pdot_domain, conv1, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        conv2 = np.convolve(conv1, Pdot_int_prob_i, 'same')

        # Normalize
        conv2 /= interp.UnivariateSpline(
            Pdot_domain, conv2, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        # ------------------------------------------------------------------
        # Compute the Shklovskii (proper motion) effect component
        # ------------------------------------------------------------------

        pm = cluster_μ * 4.84e-9 / 31557600  # mas/yr -> rad/s

        D = model.d / 3.24078e-20  # kpc -> m

        PdotP_pm = pm**2 * D / c

        # ------------------------------------------------------------------
        # Compute the galactic potential component
        # ------------------------------------------------------------------

        # TODO cahnge everything to use units
        PdotP_gal = _galactic_pot(*coords, model.d).value

        # ------------------------------------------------------------------
        # Interpolate the likelihood value from the overall distribution
        # ------------------------------------------------------------------

        prob_dist = interp.interp1d(
            Pdot_domain + PdotP_pm + PdotP_gal, conv2, assume_sorted=True,
            bounds_error=False, fill_value=0.0
        )

        probs[i] = prob_dist(Pdot_meas_i / P_i)

    # ----------------------------------------------------------------------
    # Multiply all the probabilities and return the total log probability.
    # ----------------------------------------------------------------------

    return np.log(np.prod(probs))


def likelihood_number_density(model, ndensity, *, mass_bin=None):

    if mass_bin is None:
        if 'm' in ndensity.mdata:
            mass_bin = np.where(model.mj == ndensity.mdata['m'])[0][0]
        else:
            mass_bin = model.nms - 1

    # Set cutoff to avoid fitting flat end of data
    valid = (ndensity['Σ'] > 0.1)

    obs_Σ = ndensity['Σ'][valid]
    obs_err = ndensity['ΔΣ'][valid]

    # Interpolated the model data at the measurement locations
    # TODO the numdens data should be converted to arcsec in storage or data
    interpolated = np.interp(
        ndensity['r'][valid], pc2arcsec(model.r, model.d) / 60,
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
    K = (np.sum(obs_Σ * interpolated / obs_Σ ** 2)
         / np.sum(interpolated ** 2 / obs_Σ ** 2))

    interpolated *= K

    # Now nuisance parameter
    # This allows us to add a constant error component to the data which
    # allows us to fit on the data while not worrying too much about the
    # outermost points where background effects are most prominent.

    yerr = np.sqrt(obs_err ** 2 + model.s2)

    # Now regular gaussian likelihood
    return -0.5 * np.sum(
        (obs_Σ - interpolated) ** 2 / yerr ** 2 + np.log(yerr ** 2)
    )


def likelihood_pm_tot(model, pm, *, mass_bin=None):

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


def likelihood_pm_ratio(model, pm, *, mass_bin=None):

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


def likelihood_pm_T(model, pm, *, mass_bin=None):

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


def likelihood_pm_R(model, pm, *, mass_bin=None):

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


def likelihood_LOS(model, vlos, *, mass_bin=None):

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


def create_model(theta, observations=None, *, verbose=False):
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
        raise ValueError(err)

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

        if fnmatch.fnmatch(key, '*pulsar*'):

            mdata = obs.mdata['μ'], (obs.mdata['b'], obs.mdata['l'])

            comps.append((key, likelihood_pulsar, pulsar_Pdot_KDE(), *mdata))

        elif fnmatch.fnmatch(key, '*velocity_dispersion*'):
            comps.append((key, likelihood_LOS, ))

        elif fnmatch.fnmatch(key, '*number_density*'):
            comps.append((key, likelihood_number_density, ))

        elif fnmatch.fnmatch(key, '*proper_motion*'):
            if 'PM_tot' in obs[key]:
                comps.append((key, likelihood_pm_tot, ))

            if 'PM_ratio' in obs[key]:
                comps.append((key, likelihood_pm_ratio, ))

            if 'PM_R' in obs[key]:
                comps.append((key, likelihood_pm_R, ))

            if 'PM_T' in obs[key]:
                comps.append((key, likelihood_pm_T, ))

        elif fnmatch.fnmatch(key, '*mass_function*'):
            comps.append((key, likelihood_mass_func, ))

    return comps


# Main likelihood function, generates the model(theta) passes it to the
# individual likelihood functions and collects their results.
def log_likelihood(theta, observations, L_components):

    try:
        model = create_model(theta, observations)
    except ValueError:
        # Model did not converge
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
