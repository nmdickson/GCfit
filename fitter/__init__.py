import h5py
import emcee
import schwimmbad
import numpy as np

import sys
import time
import shutil
import logging

from .likelihoods import posterior, determine_components
from .data import Observations


# TODO this should have some defaults probably
def main(cluster, Niters, Nwalkers, Ncpu, mpi,
         priors, cont_run, savedir, outdir, verbose, debug):

    # TODO if not debug, only info if verbose
    logfile = f"{savedir}/fitter_{cluster}.log"
    loglvl = logging.DEBUG if debug else logging.INFO
    logfmt = ('%(process)s|%(asctime)s|'
              '%(name)s:%(module)s:%(funcName)s:%(message)s')
    datefmt = '%H:%M:%S'

    logging.basicConfig(filename=logfile, level=loglvl,
                        format=logfmt, datefmt=datefmt)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.info("BEGIN")

    # Load the observation data here once
    logging.info(f"Loading {cluster} data")

    observations = Observations(cluster)

    logging.debug(f"Observation datasets: {observations}")

    # determine which likelihoods to compute
    L_components = determine_components(observations)

    blobs_dtype = [(f'{key}/{func.__qualname__}', float)
                   for (key, func, *_) in L_components]

    logging.debug(f"Likelihood components: {L_components}")

    # Initialize the walker positions
    Ndim = 13
    init_pos = np.fromiter(observations.priors.values(), np.float64)
    init_pos = 1e-4 * np.random.randn(Nwalkers, Ndim) + init_pos

    # HDF file saving
    logging.debug(f"Using hdf backend at {savedir}/{cluster}_sampler.hdf")

    backend_fn = f"{savedir}/{cluster}_sampler.hdf"
    backend = emcee.backends.HDFBackend(backend_fn)

    logging.info("Beginning pool")

    with schwimmbad.choose_pool(mpi=mpi, processes=Ncpu) as pool:

        logging.debug(f"Pool class: {pool}, with {mpi=}, {Ncpu=}")

        if mpi and not pool.is_master():
            logging.debug("This process is not master")
            pool.wait()
            sys.exit(0)

        logging.info("Initializing sampler")

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(
            Nwalkers,
            Ndim,
            posterior,
            args=(observations, L_components,),
            pool=pool,
            backend=backend,
            blobs_dtype=blobs_dtype
        )

        logging.debug(f"Sampler class: {sampler}")

        # The acceptance rate is unfortunately not auto-stored in the backend
        accept_rate = np.empty((Niters, Nwalkers))
        iter_rate = np.empty(Niters)

        # Start the sampler
        # Need to change initial_state to None if resuming from previous run.

        logging.info(f"Iterating sampler ({Niters} iterations)")

        t = time.time()

        # Set initial state to None if resuming run (`cont_run`)
        for _ in sampler.sample(init_pos, iterations=Niters, progress=verbose):

            t_i = time.time()
            iter_rate[sampler.iteration - 1] = t_i - t
            t = t_i

            accept_rate[sampler.iteration - 1, :] = sampler.acceptance_fraction

            if sampler.iteration % 100 == 0:
                logging.debug(f"{sampler.iteration=}: Creating backup")
                shutil.copyfile(f"{savedir}/{cluster}_sampler.hdf",
                                f"{savedir}/.backup_{cluster}_sampler.hdf")

        # Attempt to get autocorrelation time, chain may not be long enough
        try:
            tau = sampler.get_autocorr_time()
        except emcee.autocorr.AutocorrError:
            tau = np.nan

    logging.info(f'Autocorrelation time: {tau}')

    logging.info("Finished iteration")

    with h5py.File(backend_fn, 'r+') as backend_hdf:
        stat_grp = backend_hdf.require_group(name='statistics')

        stat_grp.create_dataset(name='iteration_rate', data=iter_rate)
        stat_grp.create_dataset(name='acceptance_rate', data=accept_rate)

    logging.debug(f"Final state: {sampler}")

    # Print results to stdout

    if verbose:
        flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)

        mssg = ''
        for ind, key in enumerate(observations.priors):
            perc = np.percentile(flat_samples[:, ind], [16, 50, 84])
            qnt = np.diff(perc)

            mssg += f'{key:>5} = {perc[1]:.3f} (+{qnt[0]:.3f}/-{qnt[1]:.3f})\n'

        sys.stdout.write(mssg)

    logging.info("FINISHED")
