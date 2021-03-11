from .data import Observations
from ..probabilities import posterior

import h5py
import emcee
import schwimmbad
import numpy as np

import sys
import time
import shutil
import logging
import pathlib


__all__ = ['fit']


_here = pathlib.Path()


def fit(cluster, Niters, Nwalkers, Ncpu=2, *,
        mpi=False, initials=None, fixed_params=None, excluded_likelihoods=None,
        cont_run=False, savedir=_here, outdir=_here, verbose=False):

    logging.info("BEGIN")

    # ----------------------------------------------------------------------
    # Check arguments
    # ----------------------------------------------------------------------

    if fixed_params is None:
        fixed_params = []

    if excluded_likelihoods is None:
        excluded_likelihoods = []

    if cont_run:
        raise NotImplementedError

    savedir = pathlib.Path(savedir)
    if not savedir.is_dir():
        raise ValueError(f"Cannot access '{savedir}': No such directory")

    outdir = pathlib.Path(outdir)
    if not outdir.is_dir():
        raise ValueError(f"Cannot access '{outdir}': No such directory")

    # ----------------------------------------------------------------------
    # Load obeservational data, determine which likelihoods are valid/desired
    # ----------------------------------------------------------------------

    logging.info(f"Loading {cluster} data")

    observations = Observations(cluster)

    logging.debug(f"Observation datasets: {observations}")

    # determine which likelihoods to compute (given data and exclusions)
    likelihoods = [
        component for component in observations.valid_likelihoods
        if not (component[0] in excluded_likelihoods
                or component[1].__name__ in excluded_likelihoods)
    ]

    blobs_dtype = [(f'{key}/{func.__qualname__}', float)
                   for (key, func, *_) in likelihoods]

    logging.debug(f"Likelihood components: {likelihoods}")

    # ----------------------------------------------------------------------
    # Initialize the walker positions
    # ----------------------------------------------------------------------

    # *_params -> list of keys, *_initials -> dictionary

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

        logging.info("Initializing sampler")

        # ------------------------------------------------------------------
        # Initialize the MCMC sampler
        # ------------------------------------------------------------------

        sampler = emcee.EnsembleSampler(
            Nwalkers,
            init_pos.shape[-1],
            posterior,
            args=(observations, fixed_initials, likelihoods,),
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
        for _ in sampler.sample(init_pos, iterations=Niters):

            # --------------------------------------------------------------
            # Store some iteration metadata
            # --------------------------------------------------------------

            t_i = time.time()
            iter_rate[sampler.iteration - 1] = t_i - t
            t = t_i

            accept_rate[sampler.iteration - 1, :] = sampler.acceptance_fraction

            if sampler.iteration % 100 == 0:
                # TODO the backup probably isn't necessary anymore
                logging.debug(f"{sampler.iteration=}: Creating backup")
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

        meta_grp.attrs['mpi'] = mpi
        meta_grp.attrs['Ncpu'] = Ncpu

        meta_grp.attrs['runtime'] = time.time() - t0
        meta_grp.attrs['autocorr'] = tau

        fix_dset = meta_grp.create_dataset("fixed_params", dtype="f")
        for k, v in fixed_initials.items():
            fix_dset.attrs[k] = v

        ex_dset = meta_grp.create_dataset("excluded_likelihoods", dtype='f')
        for i, L in enumerate(excluded_likelihoods):
            ex_dset.attrs[str(i)] = L

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