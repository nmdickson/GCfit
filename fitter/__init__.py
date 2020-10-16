import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

import shutil
import multiprocessing as mp

from .data import A_SPACE, get_dataset
from .likelihoods import log_probability


# Generate a list of gaussian distributions with sigma coresponding
# to the error on the pulsar az measurements.
def pregen_pulsar_error(a_space, a_los_err):
    # TODO this can be vectorized easily (80)

    def gaussian(x, sigma, mu):
        '''Simple gaussian implementation'''
        norm = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = np.exp(-0.5 * (((x - mu) / sigma) ** 2))
        return norm * exponent

    dists = []
    for i in range(len(a_los_err)):
        dist = gaussian(x=a_space, sigma=a_los_err[i], mu=0)
        dists.append(dist)
    return dists


def main():

    print("Starting")

    nwalkers, ndim = 32, 13

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
    pos += 1e-4 * np.random.randn(nwalkers, ndim)

    # Generate the error distributions a single time instead of for every model
    # given that they are constants.
    # TODO is this really the only thing that can be "pre"generated?
    print("preGenerating pulsar error distributions")
    pulsar_edist = pregen_pulsar_error(A_SPACE, get_dataset('pulsar/Δa_los'))

    print("Starting sampler")

    # HDF file saving
    backend = emcee.backends.HDFBackend("sample_multi.hdf")
    # Comment this line out if resuming from previous run, also change initial
    #   state to None
    # where the sampler is run.
    backend.reset(nwalkers, ndim)

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(pulsar_edist,),
        pool=mp.Pool(2),
        backend=backend,
    )

    # Star the sampler
    # Need to change initial_state to None if resuming from previous run.

    # Backup counter, we want to have 2 copies just in case but no
    # more than 2 to preserve disk space
    current_backup = 0

    print('starting sampling iterations')

    for _ in sampler.sample(initial_state=pos, iterations=10, progress=True):

        try:
            if sampler.iteration % 10 == 0:
                if current_backup == 0:
                    shutil.copyfile("sample_multi.hdf", "./backup/sampler0.hdf")
                    current_backup = 1
                else:
                    shutil.copyfile("sample_multi.hdf", "./backup/sampler1.hdf")
                    current_backup = 0
        except Exception:
            pass
    # Attempt to print autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
        print("Tau = " + str(tau))
    except Exception:
        # Usually can't print
        print(" WARN: May not have reached full autocorrelation time")

    # Plot Walkers
    print("Plotting walkers")

    fig, axes = plt.subplots(13, figsize=(10, 20), sharex=True)
    samples = sampler.get_chain()
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
        r"BH_ret",
        r"d",
    ]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    fig.tight_layout()
    plt.savefig("walkers.png", dpi=600)

    # Corner Plots
    print("Plotting corner plots")

    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    fig = corner.corner(
        flat_samples,
        labels=[
            r"$W_{0}$",
            r"$M/10^6 M_{\odot}$",
            r"$r_h / pc$",
            r"$ log \ r_a / pc$",
            r"$g$",
            r"$\delta$",
            r"$s^2$",
            r"$F$",
            r"$\alpha_1$",
            r"$\alpha_2$",
            r"$\alpha_3$",
            r"$BH_{ret}$",
            r"d",
        ],
    )
    plt.savefig("corner.png", dpi=600)

    # Print results
    print("Results: ")

    # This is an absolute mess but it works.
    for i in range(ndim):
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


if __name__ == "__main__":
    main()
