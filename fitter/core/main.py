from .data import Observations
from ..probabilities import posterior, priors

import h5py
import emcee
import schwimmbad
import numpy as np

import sys
import time
import shutil
import logging
import pathlib
from fnmatch import fnmatch


__all__ = ['fit']


_here = pathlib.Path()


def fit(cluster, Niters, Nwalkers, Ncpu=2, *,
        mpi=False, initials=None, param_priors=None,
        fixed_params=None, excluded_likelihoods=None,
        cont_run=False, savedir=_here, backup=False, verbose=False):
    '''Main MCMC fitting pipeline

    Execute the full MCMC cluster fitting algorithm.

    Based on the given clusters `Observations`, determines the relevant
    likelihoods used to construct an MCMC ensemble sampler (`emcee`) and
    initializes it based on the `initials` stored in the `Observations`.

    MCMC chains and information is stored using an HDF file backend, within
    the `savedir` directory under the filename "{cluster}_sampler.hdf". Also
    stored within this file is various statistics and metadata surrounding the
    fitter run. Alongside each walker's posterior probability profile is
    stored the individual likelihood values of each used likelihood function.

    The MCMC sampler is sampled for `Niters` iterations, parallelized over
    `Ncpu` or using `mpi`, with calls to `fitter.posterior`.

    parameters
    ----------
    cluster : str
        Cluster common name, as used to load `fitter.Observations`

    Niters : int
        Number of sampler iterations

    Nwalkers : int
        Number of sampler walkers

    Ncpu : int, optional
        Number of CPU's to parallelize the sampling computation over. Is
        ignored if `mpi` is True.

    mpi : bool, optional
        Parallelize sampling computation using mpi rather than multiprocessing.
        Parallelization is handled by `schwimmbad`.

    initials : dict, optional
        Dictionary of initial walker positions for each parameter. If
        None (default), uses the `initials` stored in the cluster's
        `Observations`. Any missing parameters in `initials` will be filled
        with the values stored in the `Observations`.

    param_priors : dict, optional
        Dictionary of prior bounds/args for each parameter.
        See `probabilities.priors` for formatting of args and defaults.

    fixed_params : list of str, optional
        List of parameters to fix to the initial value, and not allow to be
        varied through the sampler.

    excluded_likelihoods : list of str, optional
        List of component likelihoods to exclude from the posterior probability
        function. Each likelihood can be specified using either the name of
        the function (as given by __name__) or the name of the relevant dataset.

    cont_run : bool, optional
        Not Implemented

    savedir : path-like, optional
        The directory within which the HDF output file is stored, defaults to
        the current working directory.

    backup : bool, optional
        If True, create a continuous backup HDF file every 100 iterations

    verbose : bool, optional
        Increase verbosity (currently only affects output of run final summary)

    See Also
    --------
    emcee : MCMC Ensemble sampler implementation
    schwimmbad : Interface to parallel processing pools
    '''

    logging.info("BEGIN")

    # ----------------------------------------------------------------------
    # Check arguments
    # ----------------------------------------------------------------------

    if fixed_params is None:
        fixed_params = []

    if excluded_likelihoods is None:
        excluded_likelihoods = []

    if param_priors is None:
        param_priors = {}

    if cont_run:
        raise NotImplementedError

    savedir = pathlib.Path(savedir)
    if not savedir.is_dir():
        raise ValueError(f"Cannot access '{savedir}': No such directory")

    # ----------------------------------------------------------------------
    # Load obeservational data, determine which likelihoods are valid/desired
    # ----------------------------------------------------------------------

    logging.info(f"Loading {cluster} data")

    observations = Observations(cluster)

    logging.debug(f"Observation datasets: {observations}")

    likelihoods = []
    for component in observations.valid_likelihoods:
        key, func, *_ = component
        func_name = func.__name__

        if not any(fnmatch(key, pattern) or fnmatch(func_name, pattern)
                   for pattern in excluded_likelihoods):

            likelihoods.append(component)

    blobs_dtype = [(f'{key}/{func.__qualname__}', float)
                   for (key, func, *_) in likelihoods]

    logging.debug(f"Likelihood components: {likelihoods}")

    # ----------------------------------------------------------------------
    # Initialize the walker positions
    # ----------------------------------------------------------------------

    # *_params -> list of keys, *_initials -> dictionary

    spec_initials = initials

    # get supplied initials, or read them from the data files if not given
    if initials is None:
        initials = observations.initials
    else:
        # fill manually supplied dict with defaults (change to unions in 3.9)
        initials = {**observations.initials, **initials}

    logging.debug(f"Inital initals: {initials}")

    if extraneous_params := (set(fixed_params) - initials.keys()):
        raise ValueError(f"Invalid fixed parameters: {extraneous_params}")

    variable_params = (initials.keys() - set(fixed_params))
    if not variable_params:
        raise ValueError(f"No non-fixed parameters left, fix less parameters")

    # variable params sorting matters for setup of theta, but fixed does not
    fixed_initials = {key: initials[key] for key in fixed_params}
    variable_initials = {key: initials[key] for key in
                         sorted(variable_params, key=list(initials).index)}

    logging.debug(f"Fixed initals: {fixed_initials}")
    logging.debug(f"Variable initals: {variable_initials}")

    init_pos = np.fromiter(variable_initials.values(), np.float64)
    init_pos = 1e-4 * np.random.randn(Nwalkers, init_pos.size) + init_pos

    # ----------------------------------------------------------------------
    # Setup and check param_priors
    # ----------------------------------------------------------------------

    spec_priors = param_priors

    prior_likelihood = priors.Priors(param_priors)

    # check if initials are outside priors, if so then error right here
    if not np.isfinite(prior_likelihood(initials)):
        raise ValueError("Initial positions outside prior boundaries")

    # ----------------------------------------------------------------------
    # Setup MCMC backend
    # ----------------------------------------------------------------------

    # TODO if not cont_run, need to make sure this file doesnt already exist
    backend_fn = f"{savedir}/{cluster}_sampler.hdf"

    logging.debug(f"Using hdf backend at {backend_fn}")

    backend = emcee.backends.HDFBackend(backend_fn)

    accept_rate = np.empty((Niters, Nwalkers))
    iter_rate = np.empty(Niters)

    # ----------------------------------------------------------------------
    # Setup multi-processing pool
    # ----------------------------------------------------------------------

    logging.info("Beginning pool")

    with schwimmbad.choose_pool(mpi=mpi, processes=Ncpu) as pool:

        logging.debug(f"Pool class: {pool}, with {mpi=}, {Ncpu=}")

        if mpi and not pool.is_master():
            logging.debug("This process is not master")
            pool.wait()
            sys.exit(0)

        # ----------------------------------------------------------------------
        # Write run metadata to output (backend) file
        # ----------------------------------------------------------------------

        with h5py.File(backend_fn, 'a') as backend_hdf:

            meta_grp = backend_hdf.require_group(name='metadata')

            meta_grp.attrs['cluster'] = cluster

            # parallelization setup
            meta_grp.attrs['mpi'] = mpi
            meta_grp.attrs['Ncpu'] = Ncpu

            # Fixed parameters
            fix_dset = meta_grp.create_dataset("fixed_params", dtype="f")
            for k, v in fixed_initials.items():
                fix_dset.attrs[k] = v

            # Excluded likelihoods
            ex_dset = meta_grp.create_dataset("excluded_likelihoods", dtype='f')
            for i, L in enumerate(excluded_likelihoods):
                ex_dset.attrs[str(i)] = L

            # Specified initial values
            init_dset = meta_grp.create_dataset("specified_initials", dtype="f")
            if spec_initials is not None:
                for k, v in spec_initials.items():
                    init_dset.attrs[k] = v

            # Specified priors
            bnd_dset = meta_grp.create_dataset("specified_priors", dtype="f")
            if spec_priors:
                for k, v in spec_priors.items():
                    # TODO I don't even think theres a good way to do this
                    bnd_dset.attrs[f'{k}_type'] = np.array(v[0]).astype('|S10')
                    bnd_dset.attrs[f'{k}_args'] = np.array(v[1:]).astype('|S10')

        # ------------------------------------------------------------------
        # Initialize the MCMC sampler
        # ------------------------------------------------------------------

        logging.info("Initializing sampler")

        sampler = emcee.EnsembleSampler(
            nwalkers=Nwalkers,
            ndim=init_pos.shape[-1],
            log_prob_fn=posterior,
            args=(observations, fixed_initials, likelihoods, prior_likelihood),
            pool=pool,
            backend=backend,
            blobs_dtype=blobs_dtype
        )

        logging.debug(f"Sampler class: {sampler}")

        # ------------------------------------------------------------------
        # Run the sampler
        # ------------------------------------------------------------------

        logging.info(f"Iterating sampler ({Niters} iterations)")

        t0 = t = time.time()

        # TODO implement cont_run
        # Set initial state to None if resuming run (`cont_run`)
        for _ in sampler.sample(init_pos, iterations=Niters, progress=True):

            # --------------------------------------------------------------
            # Store some iteration metadata
            # --------------------------------------------------------------

            t_i = time.time()
            iter_rate[sampler.iteration - 1] = t_i - t
            t = t_i

            accept_rate[sampler.iteration - 1, :] = sampler.acceptance_fraction

            if sampler.iteration % 100 == 0:
                logging.debug(f"{sampler.iteration=}")
                if backup:
                    shutil.copyfile(f"{savedir}/{cluster}_sampler.hdf",
                                    f"{savedir}/.backup_{cluster}_sampler.hdf")

        try:
            # Attempt to get autocorrelation time
            tau = sampler.get_autocorr_time()
        except emcee.autocorr.AutocorrError:
            tau = np.nan

    logging.info(f'Autocorrelation time: {tau}')

    logging.info("Finished iteration")

    # ----------------------------------------------------------------------
    # Write extra metadata and statistics to output (backend) file
    # ----------------------------------------------------------------------

    with h5py.File(backend_fn, 'r+') as backend_hdf:

        # Store run metadata

        meta_grp = backend_hdf.require_group(name='metadata')

        meta_grp.attrs['runtime'] = time.time() - t0
        meta_grp.attrs['autocorr'] = tau

        # Store run statistics

        stat_grp = backend_hdf.require_group(name='statistics')

        stat_grp.create_dataset(name='iteration_rate', data=iter_rate)
        stat_grp.create_dataset(name='acceptance_rate', data=accept_rate)

    logging.debug(f"Final state: {sampler}")

    # ----------------------------------------------------------------------
    # Print some verbose results to stdout
    # ----------------------------------------------------------------------

    if verbose:
        from .. import visualize as viz
        viz.RunVisualizer(backend_fn, observations).print_summary()

    logging.info("FINISHED")
