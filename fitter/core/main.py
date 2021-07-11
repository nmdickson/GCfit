from .data import Observations
from ..probabilities import posterior, priors

import h5py
import emcee
import schwimmbad
import numpy as np
import dynesty
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

    def store_dataset(self, key, data, group='statistics', *, file=None):
        '''currently only works for adding a full array once, will overwrite'''

        data = np.asanyarray(data)

        if data.dtype.kind == 'U':
            data = data.astype('S')

        hdf = file or self.open('a')

        grp = hdf.require_group(name=group)

        if key in grp:
            del grp[key]

        grp[key] = data

        # TODO is this really the best way to allow for passing open files?
        if not file:
            hdf.close()

    def store_metadata(self, key, value, type_postfix='', *, file=None):

        hdf = file or self.open('a')

        meta_grp = hdf.require_group(name='metadata')

        if isinstance(value, abc.Mapping):

            dset = meta_grp.require_dataset(key, dtype="f", shape=None)

            for k, v in value.items():

                v = np.asanyarray(v)

                if v.dtype.kind == 'U':
                    v = v.astype('S')

                dset.attrs[f'{k}{type_postfix}'] = v

        elif isinstance(value, abc.Collection) \
                and not isinstance(value, _str_types):

            dset = meta_grp.require_dataset(key, dtype="f", shape=None)

            for i, v in enumerate(value):

                v = np.asanyarray(v)

                if v.dtype.kind == 'U':
                    v = v.astype('S')

                dset.attrs[f'{i}{type_postfix}'] = v

        else:

            meta_grp.attrs[f'{key}{type_postfix}'] = value

        if not file:
            hdf.close()


class MCMCOutput(emcee.backends.HDFBackend, Output):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NestedSamplingOutput(Output):
    # TODO this logic should be straightened out, groups existance standardized

    def __init__(self, filename, group='nested', overwrite=False):
        self.filename = filename
        self.group = group

    def open(self, mode="r"):
        return h5py.File(self.filename, mode)

    def _store_bounds(self, bounds, key='bound', group=None, *, file=None):
        '''store_dataset alternative for the 'bounds' key which is returned in
        ther form of bound objects, which can't be stored directly

        bounds is a list of bound objects
        '''

        hdf = file or self.open('a')

        base_grp = hdf.require_group(name=(group or self.group))

        if key in base_grp:
            del base_grp[key]

        grp = base_grp.require_group(name=key)

        for ind, bnd in enumerate(bounds):

            bnd_grp = grp.create_group(name=str(ind))

            if isinstance(bnd, dynesty.bounding.MultiEllipsoid):
                bnd_grp.attrs['type'] = 'MultiEllipsoid'
                bnd_grp.create_dataset('centres', data=bnd.ctrs)
                bnd_grp.create_dataset('covariances', data=bnd.covs)

            elif isinstance(bnd, dynesty.bounding.UnitCube):
                bnd_grp.attrs['type'] = 'UnitCube'
                bnd_grp.attrs['ndim'] = bnd.n

            elif isinstance(bnd, dynesty.bounding.RadFriends):
                bnd_grp.attrs['type'] = 'RadFriends'
                bnd_grp.attrs['ndim'] = bnd.n
                bnd_grp.create_dataset('covariances', data=bnd.cov)

            elif isinstance(bnd, dynesty.bounding.SupFriends):
                bnd_grp.attrs['type'] = 'SupFriends'
                bnd_grp.attrs['ndim'] = bnd.n
                bnd_grp.create_dataset('covariances', data=bnd.cov)

        if not file:
            hdf.close()

    # TODO be nicer if could figure out how append results rather than overwrite
    def store_results(self, results, overwrite=True):
        '''add a `dynesty.Results` dict to the file.
        if not overwrite, will fail if data already exists
        currently doesnt support appending/adding data/results so make sure
        to `combine_runs` your sampler first
        '''

        with self.open('a') as hdf:
            for key, data in results.items():
                if key == 'bound':
                    self._store_bounds(data, group=self.group, file=hdf)
                else:
                    self.store_dataset(key, data, group=self.group, file=hdf)

    # ----------------------------------------------------------------------
    # Tracking of current batch sampling
    # ----------------------------------------------------------------------

    _current_batch_keys = ('worst', 'ustar', 'vstar', 'loglstar', 'ncall',
                           'worst_orig', 'bound_orig', 'bound_iter', 'eff')

    def reset_current_batch(self):
        '''empty out the current batch group, to start tracking a new batch'''

        with self.open('a') as hdf:

            base_grp = hdf.require_group(name=self.group)

            if 'current_batch' in base_grp:
                del base_grp['current_batch']

            grp = base_grp.create_group(name='current_batch')

            # Two-dimensional datasets
            for key in {'vstar', 'ustar'}:
                grp.create_dataset(key, shape=(0, self.ndim),
                                   maxshape=(None, self.ndim))

            # One-dimensional datasets
            for key in set(self._current_batch_keys) - {'ustar', 'vstar'}:
                grp.create_dataset(key, shape=(0,), maxshape=(None,))

    def _grow_current_batch(self, n_grow=1):
        '''called in the background to dynamically resize the relevant datasets
        '''
        with self.open('r+') as hdf:
            base_grp = hdf.require_group(name=self.group)

            grp = base_grp['current_batch']

            for key in self._current_batch_keys:
                grp[key].resize(grp['vstar'].shape[0] + n_grow, axis=0)

    def update_current_batch(self, results, reset=False):
        '''Append to the "current batch" the results of each sampling iteration

        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample. **Negative values indicate the index
            of a new live point generated when initializing a new batch.**
        ustar : `~numpy.ndarray` with shape (npdim,)
            Position of the sample.
        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.
        loglstar : float
            Ln(likelihood) of the sample.
        nc : int
            Number of likelihood calls performed before the new
            live point was accepted.
        worst_it : int
            Iteration when the live (now dead) point was originally proposed.
        boundidx : int
            Index of the bound the dead point was originally drawn from.
        bounditer : int
            Index of the bound being used at the current iteration.
        eff : float
            The cumulative sampling efficiency (in percent).

        '''

        if reset:
            self.reset_current_batch()

        results = dict(zip(self._current_batch_keys, results))

        n_grow = results['vstar'].shape[0]
        self._grow_current_batch(n_grow)

        with self.open('a') as hdf:
            base_grp = hdf.require_group(name=self.group)
            grp = base_grp['current_batch']

            for key, val in results.items():
                grp[key][-n_grow:] = val


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

    spec_prior_type = {k: v[0] for k, v in param_priors.items()}
    spec_prior_args = {k: v[1:] for k, v in param_priors.items()}

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

        backend.store_metadata('cluster', cluster)

        backend.store_metadata('mpi', mpi)
        backend.store_metadata('Ncpu', Ncpu)

        backend.store_metadata('fixed_params', fixed_initials)
        backend.store_metadata('excluded_likelihoods', excluded_likelihoods)

        if spec_initials is not None:
            backend.store_metadata('specified_initials', spec_initials)

        if spec_prior_type:
            backend.store_metadata('specified_priors', spec_prior_type, '_type')
            backend.store_metadata('specified_priors', spec_prior_args, '_args')

        # ------------------------------------------------------------------
        # Initialize the MCMC sampler
        # ------------------------------------------------------------------

        logging.info("Initializing sampler")

        sampler = emcee.EnsembleSampler(
            nwalkers=Nwalkers,
            ndim=init_pos.shape[-1],
            log_prob_fn=posterior,
            args=(observations, fixed_initials, likelihoods, prior_likelihood),
            kwargs={'hyperparams': hyperparams, 'return_indiv': True},
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

    backend.store_metadata('runtime', time.time() - t0)

    backend.store_metadata('autocorr', tau)
    backend.store_metadata('reliable_autocorr', np.any((tau * 50) > Niters))

    backend.store_dataset('iteration_rate', iter_rate)
    backend.store_dataset('acceptance_rate', accept_rate)

    logging.debug(f"Final state: {sampler}")

    # ----------------------------------------------------------------------
    # Print some verbose results to stdout
    # ----------------------------------------------------------------------

    if verbose:
        from .. import visualize as viz
        viz.RunVisualizer(backend_fn, observations).print_summary()

    logging.info("FINISHED")


def nested_fit(cluster, *, bound_type='multi', sample_type='auto',
               maxiter=None, Nlive_per_batch=500, pfrac=1.0, dlogz=0.05,
               Ncpu=2, mpi=False, initials=None, param_priors=None,
               fixed_params=None, excluded_likelihoods=None, hyperparams=True,
               savedir=_here, verbose=False):
    '''Main nested sampling fitting pipeline

    Execute the full nested sampling cluster fitting algorithm.

    Based on the given clusters `Observations`, determines the relevant
    likelihoods used to construct an dynamic nested sampler (`dynesty`).

    All sampler results are stored using an HDF file backend, within
    the `savedir` directory under the filename "{cluster}_sampler.hdf". Also
    stored within this file is various statistics and metadata surrounding the
    fitter run.

    The nested sampler is sampled in batches, with each batch adding
    `Nlive_per_batch` live points, until reaching a stop condition
    defined by the fractional posterior weighting desired (`pfrac`), or until
    the maximum number of iterations is reached (`maxiter`).
    The sampling is parallelized over `Ncpu` or using `mpi`, with calls to
    `fitter.posterior` defined based on a uniform sampling of the
    `PriorTransforms`.

    parameters
    ----------
    cluster : str
        Cluster common name, as used to load `fitter.Observations`

    bound_type : {'none', 'single', 'multi', 'balls', 'cubes'}, optional
        Method used to approximately bound the prior using the current
        set of live points. Conditions the sampling methods used to propose
        new live points.

    sample_type : {'unif', 'rwalk', 'rstagger',
                   'slice', 'rslice', 'hslice'}, optional
        Method used to sample uniformly within the likelihood constraint.

    maxiter : int, optional
        Maximum number of iterations. Default is no limit.

    Nlive_per_batch : int, optional
        The number of live points used when adding additional samples
        in the batch. Default is 500

    pfrac : float, optional
        Fractional weight of the posterior (versus evidence) for stop function.
        Between 0.0 and 1.0, defaults to 1.0

    Ncpu : int, optional
        Number of CPU's to parallelize the sampling computation over. Is
        ignored if `mpi` is True.

    mpi : bool, optional
        Parallelize sampling computation using mpi rather than multiprocessing.
        Parallelization is handled by `schwimmbad`.

    initials : dict, optional
        Dictionary of initial parameter values. There is no concept
        of "initial positions" in nested sampling, and this argument is only
        used in the case of fixed parameters.

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

    hyperparams : bool, optional
        Whether to include bayesian hyperparameters (see Hobson et al., 2002)
        in all likelihood functions.

    savedir : path-like, optional
        The directory within which the HDF output file is stored, defaults to
        the current working directory.

    verbose : bool, optional
        Increase verbosity (currently only affects output of run final summary)

    See Also
    --------
    dynesty : Nested sampler implementation
    schwimmbad : Interface to parallel processing pools
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
    # Setup param_priors transforms
    # ----------------------------------------------------------------------

    spec_prior_type = {k: v[0] for k, v in param_priors.items()}
    spec_prior_args = {k: v[1:] for k, v in param_priors.items()}

    prior_kwargs = {'fixed_initials': fixed_initials, 'err_on_fail': False}
    prior_transform = priors.PriorTransforms(param_priors, **prior_kwargs)

    # ----------------------------------------------------------------------
    # Setup Nested Sampling backend
    # ----------------------------------------------------------------------

    backend_fn = f"{savedir}/{cluster}_sampler.hdf"

    logging.debug(f"Using hdf backend at {backend_fn}")

    backend = NestedSamplingOutput(backend_fn)

    # ----------------------------------------------------------------------
    # Setup multi-processing pool
    # ----------------------------------------------------------------------

    logging.info("Beginning pool")

    with schwimmbad.choose_pool(mpi=mpi, processes=Ncpu) as pool:

        map_ = pool.map

        logging.debug(f"Pool class: {pool}, with {mpi=}, {Ncpu=}")

        if mpi and not pool.is_master():
            logging.debug("This process is not master")
            pool.wait()
            sys.exit(0)

        # ----------------------------------------------------------------------
        # Write run metadata to output (backend) file
        # ----------------------------------------------------------------------

        backend.store_metadata('cluster', cluster)

        backend.store_metadata('mpi', mpi)
        backend.store_metadata('Ncpu', Ncpu)

        backend.store_metadata('fixed_params', fixed_initials)
        backend.store_metadata('excluded_likelihoods', excluded_likelihoods)

        if spec_initials is not None:
            backend.store_metadata('specified_initials', spec_initials)

        if spec_prior_type:
            backend.store_metadata('specified_priors', spec_prior_type, '_type')
            backend.store_metadata('specified_priors', spec_prior_args, '_args')

        # ------------------------------------------------------------------
        # Initialize the Nested sampler
        # ------------------------------------------------------------------

        logging.info("Initializing sampler")

        ndim = len(variable_initials)

        backend.ndim = ndim

        sampler = dynesty.DynamicNestedSampler(
            ndim=ndim,
            loglikelihood=posterior,  # cause we need the defaults/checks it has
            prior_transform=prior_transform,
            logl_args=(observations, fixed_initials, likelihoods, 'ignore'),
            logl_kwargs={'hyperparams': hyperparams, 'return_indiv': False},
            pool=pool,
            bound=bound_type,
            method=sample_type
        )

        logging.debug(f"Sampler class: {sampler}")

        # ------------------------------------------------------------------
        # Run the sampler
        # ------------------------------------------------------------------

        logging.info("Initializing sampler run")

        t0 = time.time()

        stop_kw = {'pfrac': pfrac}
        initsample_kw = {'maxiter': maxiter, 'nlive': Nlive_per_batch,
                         'dlogz': dlogz}
        sample_kw = {'maxiter': maxiter, 'nlive_new': Nlive_per_batch}

        backend.reset_current_batch()

        # runs an initial set of set samples, as if using `NestedSampler`
        for results in sampler.sample_initial(**initsample_kw):

            # get rid of extra args from initial sampling
            # TODO maybe we shouldn't drop these?
            (worst, ustar, vstar, loglstar, _, _, _, _, _,
             ncall, worst_orig, bound_orig, bound_iter, eff, _) = results

            results = (worst, ustar, vstar, loglstar,
                       ncall, worst_orig, bound_orig, bound_iter, eff)

            backend.update_current_batch(results)

        backend.store_results(sampler.results)

        logging.info("Beginning dynamic batch sampling")

        # run the dynamic sampler in batches, until the stop condition is met
        while not dysamp.stopping_function(sampler.results, stop_kw, M=map_):

            backend.reset_current_batch()

            wts = dysamp.weight_function(sampler.results, stop_kw)

            for results in sampler.sample_batch(logl_bounds=wts, **sample_kw):
                backend.update_current_batch(results)

            logging.info("Combining batch with existing results")

            # add new samples to previous results, save in backend
            sampler.combine_runs()

            backend.store_results(sampler.results)

    logging.info("Finished sampling")

    # ----------------------------------------------------------------------
    # Write extra metadata and statistics to output (backend) file
    # ----------------------------------------------------------------------

    backend.store_metadata('runtime', time.time() - t0)

    logging.debug(f"Final state: {sampler}")

    # ----------------------------------------------------------------------
    # Print some verbose results to stdout
    # ----------------------------------------------------------------------

    if verbose:
        from .. import visualize as viz
        viz.RunVisualizer(backend_fn, observations).print_summary()

    logging.info("FINISHED")
