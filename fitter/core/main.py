from .data import Observations
from ..probabilities import posterior, priors

import h5py
import emcee
import schwimmbad
import numpy as np
import dynesty.dynamicsampler as dysamp

import sys
import time
import shutil
import logging
import pathlib
from fnmatch import fnmatch
from collections import abc


__all__ = ['MCMC_fit', 'nested_fit']


_here = pathlib.Path()
_str_types = (str, bytes)


class Output:
    # TODO careful this doesnt conflict with h5py's `create_dataset`
    def create_dataset(self, key, data, file=None, group='statistics'):
        '''currently only works for adding a full array once'''

        data = np.asanyarray(data)

        if data.dtype.kind == 'U':
            data = data.astype('S')

        hdf = file or self.open('a')

        grp = hdf.require_group(name=group)

        grp[key] = data

        # TODO is this really the best way to allow for passing open files?
        if file:
            hdf.close()

    def add_metadata(self, key, value, file=None, value_postfix=''):

        hdf = file or self.open('a')

        meta_grp = hdf.require_group(name='metadata')

        if isinstance(value, abc.Mapping):

            dset = meta_grp.require_dataset(key, data=h5py.Empty("f"))

            for k, v in value.items():
                dset.attrs[f'{k}{value_postfix}'] = v

        elif isinstance(value, abc.Collection) \
                and not isinstance(value, _str_types):

            dset = meta_grp.require_dataset(key, data=h5py.Empty("f"))

            for i, v in enumerate(value):
                dset.attrs[f'{i}{value_postfix}'] = v

        else:

            meta_grp.attrs[f'{key}{value_postfix}'] = value

        if file:
            hdf.close()


class MCMCOutput(emcee.backends.HDFBackend, Output):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NestedSamplingOutput(Output):

    def __init__(self, filename, group='nested', overwrite=False):
        self.filename = filename
        self.group = group

        mode = 'w' if overwrite else 'a'

        with self.open(mode):
            self.file.create_group(self.group)

    def open(self, mode="r"):
        return h5py.File(self.filename, mode)

    def add_results(self, results, overwrite=True):
        '''add a `dynesty.Results` dict to the file.
        if not overwrite, will fail if data already exists
        currently doesnt support appending/adding data/results so make sure
        to `combine_runs` your sampler first
        '''

        for key, data in results.items():
            with self.open('a'):
                self.create_dataset(key, data, group=self.group)


def MCMC_fit(cluster, Niters, Nwalkers, Ncpu=2, *,
             mpi=False, initials=None, param_priors=None, moves=None,
             fixed_params=None, excluded_likelihoods=None, hyperparams=True,
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

    moves : list of emcee.moves.Move, optional
        List of MCMC proposal algorithms, or "moves", as defined within `emcee`.
        This list is simply passed to `emcee.EnsembleSampler`.

    fixed_params : list of str, optional
        List of parameters to fix to the initial value, and not allow to be
        varied through the sampler.

    excluded_likelihoods : list of str, optional
        List of component likelihoods to exclude from the posterior probability
        function. Each likelihood can be specified using either the name of
        the function (as given by __name__) or the name of the relevant dataset.

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters (see Hobson et al., 2002)
        in all likelihood functions.

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

    spec_priors_type = {k: v[0] for k, v in param_priors.items()}
    spec_priors_args = {k: v[1:] for k, v in param_priors.items()}

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

    # backend = emcee.backends.HDFBackend(backend_fn)
    backend = MCMCOutput(backend_fn)

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

        backend.add_metadata('cluster', cluster)

        backend.add_metadata('mpi', mpi)
        backend.add_metadata('Ncpu', Ncpu)

        backend.add_metadata('fixed_params', fixed_initials)
        backend.add_metadata('excluded_likelihoods', excluded_likelihoods)

        if spec_initials is not None:
            backend.add_metadata('specified_initials', spec_initials)

        if spec_priors_type:
            backend.add_metadata('specified_priors', spec_priors_type, '_type')
            backend.add_metadata('specified_priors', spec_priors_args, '_args')

        # ------------------------------------------------------------------
        # Initialize the MCMC sampler
        # ------------------------------------------------------------------

        logging.info("Initializing sampler")

        sampler = emcee.EnsembleSampler(
            nwalkers=Nwalkers,
            ndim=init_pos.shape[-1],
            log_prob_fn=posterior,
            args=(observations, fixed_initials, likelihoods, prior_likelihood),
            kwargs={'hyperparams': hyperparams},
            pool=pool,
            moves=moves,
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
        for _ in sampler.sample(init_pos, iterations=Niters):

            # TODO it would be nice if iteration num was added to log preamble

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

        # Attempt to get autocorrelation time
        tau = sampler.get_autocorr_time(quiet=True)

        logging.info(f'Autocorrelation times: {tau}')

        if ((τ := tau.max()) * 50) > sampler.iteration:
            logging.warning('Chain not long enough for reliable autocorrelation'
                            f', should run for atleast {50 * τ=:g} iterations.')

    logging.info("Finished iteration")

    # ----------------------------------------------------------------------
    # Write extra metadata and statistics to output (backend) file
    # ----------------------------------------------------------------------

    backend.add_metadata('runtime', time.time() - t0)

    backend.add_metadata('autocorr', tau)
    backend.add_metadata('reliable_autocorr', np.any((tau * 50) > Niters))

    backend.create_dataset('iteration_rate', iter_rate)
    backend.create_dataset('acceptance_rate', accept_rate)

    logging.debug(f"Final state: {sampler}")

    # ----------------------------------------------------------------------
    # Print some verbose results to stdout
    # ----------------------------------------------------------------------

    if verbose:
        from .. import visualize as viz
        viz.RunVisualizer(backend_fn, observations).print_summary()

    logging.info("FINISHED")


def nested_fit(cluster, *,
               Ncpu=2, mpi=False, initials=None, param_priors=None,
               fixed_params=None, excluded_likelihoods=None, hyperparams=True,
               savedir=_here, verbose=False):
    '''nsted sampling fitter
    '''

    logging.info("BEGIN NESTED SAMPLING")

    # ----------------------------------------------------------------------
    # Check arguments
    # ----------------------------------------------------------------------

    if fixed_params is None:
        fixed_params = []

    if excluded_likelihoods is None:
        excluded_likelihoods = []

    if param_priors is None:
        param_priors = {}

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

    # ----------------------------------------------------------------------
    # Setup and check param_priors
    # ----------------------------------------------------------------------

    spec_priors_type = {k: v[0] for k, v in param_priors.items()}
    spec_priors_args = {k: v[1:] for k, v in param_priors.items()}

    prior_likelihood = priors.Priors(param_priors, transform=True)

    # ----------------------------------------------------------------------
    # Setup MCMC backend
    # ----------------------------------------------------------------------

    backend_fn = f"{savedir}/{cluster}_sampler.hdf"

    logging.debug(f"Using hdf backend at {backend_fn}")

    backend = NestedSamplingOutput(backend_fn)

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

        backend.add_metadata('cluster', cluster)

        backend.add_metadata('mpi', mpi)
        backend.add_metadata('Ncpu', Ncpu)

        backend.add_metadata('fixed_params', fixed_initials)
        backend.add_metadata('excluded_likelihoods', excluded_likelihoods)

        if spec_initials is not None:
            backend.add_metadata('specified_initials', spec_initials)

        if spec_priors_type:
            backend.add_metadata('specified_priors', spec_priors_type, '_type')
            backend.add_metadata('specified_priors', spec_priors_args, '_args')

        # ------------------------------------------------------------------
        # Initialize the MCMC sampler
        # ------------------------------------------------------------------

        logging.info("Initializing sampler")

        sampler = dysamp.DynamicNestedSampler(
            ndim=len(variable_initials),
            loglikelihood=posterior,  # or should it be log_likelihood????
            prior_transform=prior_likelihood,
            logl_args=(observations, fixed_initials, likelihoods, 'ignore'),
            logl_kwargs={'hyperparams': hyperparams},
            pool=pool,
        )

        logging.debug(f"Sampler class: {sampler}")

        # ------------------------------------------------------------------
        # Run the sampler
        # ------------------------------------------------------------------

        logging.info("Initializing sampler run")

        t0 = time.time()

        stop_kw = {'pfrac': 1.0}

        # runs an initial set of set samples, as if using `NestedSampler`
        for results in sampler.sample_initial():
            pass

        logging.info("Beginning dynamic batch sampling")

        # run the dynamic sampler in batches, until the stop condition is met
        while not dysamp.stopping_function(sampler.results, stop_kw):

            logl_bounds = dysamp.weight_function(sampler.results, stop_kw)

            for _ in sampler.sample_batch(logl_bounds=logl_bounds):
                pass

            logging.info("Combining batch with existing results")

            # add new samples to previous results, save in backend
            sampler.combine_runs()

            backend.add_results(sampler.results)

    logging.info("Finished sampling")

    # ----------------------------------------------------------------------
    # Write extra metadata and statistics to output (backend) file
    # ----------------------------------------------------------------------

    backend.add_metadata('runtime', time.time() - t0)

    logging.debug(f"Final state: {sampler}")

    # ----------------------------------------------------------------------
    # Print some verbose results to stdout
    # ----------------------------------------------------------------------

    if verbose:
        from .. import visualize as viz
        viz.RunVisualizer(backend_fn, observations).print_summary()

    logging.info("FINISHED")
