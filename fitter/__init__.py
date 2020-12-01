import h5py
import emcee
import corner
import schwimmbad
import numpy as np
import matplotlib.pyplot as plt

import sys
import time
import shutil
import logging

from .likelihoods import log_probability, determine_components
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

    logging.debug(f"Likelihood components: {L_components}")

    # Initialize the walker positions
    # TODO i dont think its fair to vary all params by the same random range
    Ndim = 13
    init_pos = np.fromiter(observations.priors.values(), np.float64)
    init_pos = 1e-4 * np.random.randn(Nwalkers, Ndim) + init_pos

    # HDF file saving
    # TODO sometimes I think this gives issues, maybe should give unique fn
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
            log_probability,
            args=(observations, L_components,),
            pool=pool,
            backend=backend,
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

    # First attempt, without worrying about MPI communication
    with h5py.File(backend_fn, 'r+') as backend_hdf:
        stat_grp = backend_hdf.require_group(name='statistics')

        stat_grp.create_dataset(name='iteration_rate', data=iter_rate)
        stat_grp.create_dataset(name='acceptance_rate', data=accept_rate)

    logging.debug(f"Final state: {sampler}")

    logging.info("Creating final plots and writing output")

    # Plot Walkers

    samples = sampler.get_chain()
    logging.debug(f"Final chain parameters: {samples[-1]}")

    fig, axes = plt.subplots(13, figsize=(10, 20), sharex=True)

    labels = [
        r"$W_{0}$",
        r"$M/10^6 M_{\odot}$",
        r"$r_h / pc$",
        r"$ log r_a / pc$",
        r"$g$",
        r"$\delta$",
        r"$s^2$",
        r"$F$",
        r"$\alpha_1$",
        r"$\alpha_2$",
        r"$\alpha_3$",
        r"$BH_{ret}$",
        r"$d$",
    ]

    for i in range(Ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    plt.savefig(f"{outdir}/{cluster}_walkers.png", dpi=600)

    # Corner Plots

    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    fig = corner.corner(flat_samples, labels=labels)
    plt.savefig(f"{outdir}/{cluster}_corner.png", dpi=600)

    # Print results to stdout

    if verbose:
        mssg = ''
        for ind, key in enumerate(observations.priors):
            perc = np.percentile(flat_samples[:, ind], [16, 50, 84])
            qnt = np.diff(perc)

            mssg += f'{key:>5} = {perc[1]:.3f} (+{qnt[0]:.3f}/-{qnt[1]:.3f})\n'

        sys.stdout.write(mssg)

    logging.info("FINISHED")
