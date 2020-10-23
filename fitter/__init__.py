import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import emcee
import corner
import schwimmbad

import sys
import shutil
import logging

from .data import A_SPACE, get_dataset, Observations
from .likelihoods import log_probability


# TODO this should have some defaults probably
def main(cluster, Niters, Nwalkers, Ncpu, mpi,
         priors, cont_run, savedir, outdir, verbose, debug):

    logfile = f"{savedir}/fitter_{cluster}.log"
    loglvl = logging.DEBUG if debug else logging.INFO
    logfmt = ('%(process)s|%(asctime)s|'
              '%(name)s:%(module)s:%(funcName)s:%(message)s')
    datefmt = '%H:%M:%S'

    logging.basicConfig(filename=logfile, level=loglvl,
                        format=logfmt, datefmt=datefmt)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.info("BEGIN")

    Ndim = 13

    # TODO these parameters, and other prior stuff should be stored with data
    pos = [
        6.1,    # W0
        1.06,   # M
        8.1,    # rh
        1.23,   # ra
        0.7,    # g
        0.45,   # delta
        0.1,    # s
        0.45,   # F
        0.5,    # a1
        1.3,    # a2
        2.5,    # a3
        0.5,    # BHret
        4.45,   # d
    ]
    pos += 1e-4 * np.random.randn(Nwalkers, Ndim)

    # Generate the error distributions a single time instead of for every model
    # given that they are constants.

    a_width = np.abs(get_dataset(cluster, 'pulsar/Δa_los'))
    pulsar_edist = scipy.stats.norm.pdf(A_SPACE, 0, np.c_[a_width])

    # Load the observation data here once
    logging.info(f"Loading {cluster} data")

    observations = Observations('M62')

    logging.debug(f"Observation datasets: {observations}")

    # HDF file saving
    # TODO sometimes I think this gives issues, maybe should give unique fn
    logging.debug(f"Using hdf backend at {savedir}/{cluster}_sampler.hdf")

    backend = emcee.backends.HDFBackend(f"{savedir}/{cluster}_sampler.hdf")
    # Comment this line out if resuming from previous run, also change initial
    #   state to None where the sampler is run.
    backend.reset(Nwalkers, Ndim)

    logging.info("Beginning pool")

    with schwimmbad.choose_pool(mpi=mpi, processes=Ncpu) as pool:

        logging.debug(f"Pool class: {pool}, with {mpi=}, {Ncpu=}")

        if mpi and not pool.is_master():
            logging.debug("This process is master")
            pool.wait()
            sys.exit(0)

        logging.info("Initializing sampler")

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(
            Nwalkers,
            Ndim,
            log_probability,
            args=(observations, pulsar_edist,),
            pool=pool,
            backend=backend,
        )

        logging.debug(f"Sampler class: {sampler}")

        # Start the sampler
        # Need to change initial_state to None if resuming from previous run.

        logging.info(f"Iterating sampler ({Niters} iterations)")

        for _ in sampler.sample(pos, iterations=Niters, progress=verbose):

            if sampler.iteration % 10 == 0:
                logging.debug(f"{sampler.iteration=}: Creating backup")
                shutil.copyfile(f"{savedir}/{cluster}_sampler.hdf",
                                f"{savedir}/.backup_{cluster}_sampler.hdf")

        # Attempt to print autocorrelation time
        try:
            tau = sampler.get_autocorr_time()
            print("Tau = " + str(tau))
        except Exception:
            # Usually can't print
            print(" WARN: May not have reached full autocorrelation time")

    logging.info("Finished iteration")

    logging.debug(f"Final state: {sampler}")

    logging.info("Creating final plots and writing output")

    # Plot Walkers

    samples = sampler.get_chain()
    logging.debug(f"Final chain parameters: {samples[-1]}")

    fig, axes = plt.subplots(13, figsize=(10, 20), sharex=True)

    labels = [
        r"W_{0}",
        r"M/10^6 M_{\odot}",
        r"r_h / pc",
        r" log r_a / pc",
        r"g",
        r"\delta",
        r"s^2",
        r"F",
        r"\alpha_1",
        r"\alpha_2",
        r"\alpha_3",
        r"BH_{ret}",
        r"d",
    ]

    for i in range(Ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    fig.tight_layout()
    plt.savefig(f"{outdir}/{cluster}_walkers.png", dpi=600)

    # Corner Plots

    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    fig = corner.corner(flat_samples, labels=labels)
    plt.savefig(f"{outdir}/{cluster}_corner.png", dpi=600)

    # Print results

    # This is an absolute mess but it works.
    for i in range(Ndim):
        # Just use simple labels here for printing
        labels = [
            "W0",
            "M",
            "rh",
            "ra",
            "g",
            "delta",
            "s",
            "F",
            "a1",
            "a2",
            "a3",
            "BHret",
            "d",
        ]
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = r"\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        print(
            txt.split("{")[1].split("}")[0],
            " =",
            txt.split("=")[1].split("_")[0],
            " (+",
            txt.split("^")[1].split("{")[1].split("}")[0],
            " ",
            txt.split("_")[1].split("^")[0].split("{")[1].split("}")[0],
            ")"
        )

    logging.info("FINISHED")
