=====
Usage
=====

.. TODO add a bunch of plots here and there maybe 

``GCfit`` has two functionalities. The primary, to fit GC equilibrium models
against a number of observables, and the secondary, to analyze and visualize
the statistical fitting process and the resultant models.

Fitting
=======

Python
^^^^^^

The ``GCfit`` fitting process can be accessed through the python interface at

.. code-block:: python
    
    import gcfit

There are two core fitting functions, one for each sampling method. Both come
with many optional arguments, some shared between the two, and some specific
to the method chosen.

Both functions require, to begin, the name of the cluster.

.. code-block:: python

    cluster = 'NGC104'

    # MCMC Sampling
    gcfit.MCMC_fit(cluster, Niters=3000)

    # Nested Sampling
    gcfit.nested_fit(cluster)

The cluster name can be given in a few different formats. See
:func:`gcfit.util.get_std_cluster_name` for info on valid names.

Both methods share a large assortment of keyword arguments, which define the
probability functions and parallelization scheme used, as well as
method-specific arguments which define the samplers themselves. For specific
call signatures and full details, see :func:`gcfit.core.MCMC\_fit`
and :func:`gcfit.core.nested_fit`.

Probabilities
"""""""""""""

First, we may change the makeup of the posterior the sampler moves over.
Any of the 13 typically free parameters can be specified by the
``fixed_params`` argument, which will remove them from the sampling, reducing
the dimensions, and assigning the parameter to it's initial value.

.. code-block:: python
    
    gcfit.nested_fit(cluster, fixed_params=['M', 'rh'])

The initial values that these parameters are fixed to are defined by the
``initials`` argument. These also act as the initial positions for the MCMC
sampler, and the default values for each parameter are set within each data
file.

.. code-block:: python
    
    gcfit.nested_fit(cluster, fixed_params=['M'], initials={'M': 0.5})

The posterior is typically made up of a sum of component likelihoods, which
act on a specific dataset each. The component likelihood functions, and
types of datasets, can be excluded from the posterior using the
``excluded_likelihoods`` argument.

.. code-block:: python

    excluded_L = ['proper_motion/GEDR3', 'pulsar*']  # glob patterns can be used
    gcfit.nested_fit(cluster, excluded_likelihoods=excluded_L)


The posterior also includes prior probabilities on each free parameter. These
probability funnctions may also be specified using the ``param_priors``
argument. Priors are handled by the :class:`gcfit.probabilities.priors.Priors`
class. The ``param_priors`` argument accepts a dict of param-prior pairs,
where each entry must specify the type and relevant parameters of a prior
distribution.

.. code-block:: python

    priors = {
        "d": ("Gaussian", 5.2, 0.01), # Gaussian priors specify mean and width 
        "M": ("Uniform", [(0, 1.2)]), # Uniform priors specify a list of bounds
        "a2": ("Uniform", [(0, 4), ('a1', 4)]), # Other params can be used as bounds
    }

    gcfit.nested_fit(cluster, param_priors=priors)


Parallelization
"""""""""""""""

In the vast majority of cases, the sampler will be too resource-intensive
to be viably run on a single-core computer. The sampling, however, can be
easily parallelized in multiple ways.

Local parallelization (through the multiprocessing module) can be triggered
using the ``Ncpu`` argument, which simply accepts an integer number of processes
to spawn.

.. code-block:: python

    import multiprocessing
    max_cpu = multiprocessing.cpu_count()

    gcfit.nested_fit(cluster, Ncpu=max_cpu)

To run the fitting over multiple nodes, using MPI, the boolean ``mpi`` flag
can be specified. If using ``mpi``, the ``Ncpu`` argument is ignored, and the
number of processes must be specified when running the code using an
MPI-execution utility (``mpirun``, ``mpiexec``, etc.).

.. code-block:: python

    # Run script with e.g. mpiexec -n 4 python script.py
    gcfit.nested_fit(cluster, mpi=True)

The scaling of the fitting functions is not completely trivial. Before scaling
to a very large number of processes naively, users should look into any notes on
parallelization in the relevant sampler documentation (dynesty or emcee).
More is not always more.


MCMC Sampler Specific
"""""""""""""""""""""

.. things specific to MCMC

The MCMC fitting function is primarily defined by a handful of specific
arguments.

The breadth of an MCMC ensemble sampler is defined by the amount of independant
walkers in the system, which can be defined by ``Nwalkers``.

The number of iterations over which the sampler progresses can be set by the
``Niters`` argument. Lacking an obvious inherent stopping condition, this
argument should be set high enough to ensure convergence of the chains.

.. code-block:: python
    
    gcfit.MCMC_fit(cluster, Niters=1500, Nwalkers=100)


Nested Sampler Specific
"""""""""""""""""""""""
.. things specific to nested

The progression of dynamic nested sampling requires defining both the sampler
parameters and methods, the transition to dynamic sampling, and the final
stopping conditions.

The base nested sampling algorithm works by randomly sampling within the
bounds defining a single iso-likleihood contour level. As such, both the random
sampling method, and the shape of the bounds can be specified. ``dynesty``
offers a variety of choices for both, see the source paper
(`2020MNRAS.493.3132S <https://adsabs.harvard.edu/abs/2020MNRAS.493.3132S>`_)
for more information on each.

.. code-block:: python

    # Bounds can be one of {'none', 'single', 'multi', 'balls', 'cubes'}
    bound = 'multi'

    # Sampler can be one of {'unif', 'rwalk', 'rstagger', 'slice', 'rslice'}
    sampler = 'rwalk'

    gcfit.nested_fit(cluster, bound_type=bound, sample_type=sampler)

*Dynamic* nested sampling allows for a targeted focusing of the sampler
algorithm in order to more efficiently probe the posterior or evidence. This
works by beginning with a short "baseline" static run, to define the likelihood
surface, and then iterative batches of sampling in targeted locations of
parameter space.

The exact definition of these targets depends on a number of parameters. Here
the two most important can be specified; ``pfrac``, which defines the fraction
of importance to give to the posterior vs the evidence, and ``maxfrac``, which
determines the size of the targeted space.

.. code-block:: python

    pfrac = 0.9  # 1 = 100% posterior focus, 0 = 100% evidence focus

    maxfrac = 0.8  # percentage of the maximum weight, defining the new bounds

    gcfit.nested_fit(cluster, pfrac=pfrac, maxfrac=maxfrac)

Both of these arguments are described in more detail in the dynesty
documentation.

Furthermore, advanced users may tweak both the initial and dynamic sampling
batches through the ``initial_kwargs`` and ``batch_kwargs`` arguments,
respectively. See ``dynesty`` for more information.

Finally, the overall stopping conditions must be specified. While static nested
sampling, by definition, has a nicely defined stopping condition based on
evidence estimation, *dynamic* nested sampling suffers from the same issue as
MCMC. Namely that defining a single "stopping point" is difficult, and may
depend on the desired uses for the results. A more general stopping condition
is thus allowed by ``dynesty`` in the form of an "effective sample size".

This argument (``eff_samples``) must be set, in similar fashion to the MCMC
``Niters``, high enough to be confident of convergence.

.. code-block:: python

    ESS = 5000

    gcfit.nested_fit(cluster, pfrac=1, eff_samples=ESS)


Command Line
^^^^^^^^^^^^

.. introduce the GCfitter script

In order to facilitate the easy use of ``GCfit``, in particular parallelized
over a high-performance computing cluster, a command line script is provided as
an interface to the above functions.

``GCfitter`` will be installed automatically alongside the ``GCfit`` python
package, and should be automatically placed in the ``bin`` folder of the current
environment, accessible within the user's ``$PATH``.

.. describe things specific to script, how to run it, parallelism

``GCfitter`` is run from the command line, with a specific call structure.
The first argument must be the name of the cluster, in the same way it would be
used by the ``cluster`` argument above.

The second argument must be one of ``nested`` or ``MCMC``. This will define the
sampler used, as well as the valid command line arguments available

From here a number of optional arguments are available, largely consistent with
those discussed above. The largest difference being that any dictionary
arguments must be instead point to the location of a similar JSON file.

.. direct to help page

For more information on all possible arrangements, see the provided help pages:

.. code-block:: bash

    GCfitter --help

    GCfitter NGC9999 MCMC --help

    GCfitter NGC9999 nested --help

.. TODO some examples of how to do things, including in parallel, with job queue


Analysis
========

.. output files

When the fitting described above has finished, all relevant sampler information
and outputs will be stored in an output HDF5 file (in the directory specified
by ``--savedir``). This file provides everything necessary to reconstruct the
sampler evolution and results, and the corresponding models.

``GCfit`` provides utilities to read in, analyze and plot the relevant
quantities from this output, through the ``gcfit.analysis`` module.
The analysis is split into two seperate modules, for analyzing the
fitting runs and for visualizing the best-fitting models.

All fitting functions return their corresponding figure, and multiple plots
can be "stacked" onto one another. See the source API for each to find more
information, and a list of all possible plots.

.. code-block:: python

    from gcfit import analysis
    import matplotlib.pyplot as plt

    obs = gcfit.Observations(cluster)

    gcfit.nested_fit(cluster, savedir='./nested_out')
    gcfit.MCMC_fit(cluster, savedir='./MCMC_out')


Fitting Results
^^^^^^^^^^^^^^^

The run visualizers are split into specific classes once again for the MCMC
(:class:`gcfit.analysis.MCMCRun`) and nested sampler
(:class:`gcfit.analysis.NestedRun`) results.

.. code-block:: python

    nest_run = analysis.NestedRun(f'./nested_out/{cluster}_sampler.hdf', obs)
    mcmc_run = analysis.MCMCRun(f'./MCMC_out/{cluster}_sampler.hdf', obs)

    # Plot nested sampling parameter evolution, weights and final posteriors
    nest_run.plot_params()

    # Plot MCMC walker evolution
    mcmc_run.plot_chains()

    # Plot marginal distributions for both (corner plots)
    nest_run.plot_marginals()


Best Fit Models
^^^^^^^^^^^^^^^

The fitting results can be used to determine the best-fit parameters, and
corresponding confidence intervals, which in turn describe the best-fitting
model. From there, plots of all observables, as well as a number of other
cluster parameters and profiles, can be created.

The median best-fit model can be visualized with the
:class:`gcfit.analysis.ModelVisualizer` class.

.. code-block:: python

    mviz = nest_run.get_model(method='mean')

    # Plot all radial profiles (dispersions, number density, etc)
    mviz.plot_all()

    # Plot all mass functions (with fields shown)
    mviz.plot_massfunc(show_fields=True)

    # Plot cumulative mass in all stellar components
    mviz.plot_plot_cumulative_mass()

    plt.show()


All the same plots can instead be shown with confidence intervals on the
model outputs (:class:`gcfit.analysis.CIModelVisualizer`). The computation
of these intervals may be intensive, and can thus be parallelized (locally)
using the ``Nprocesses`` keyword.

.. code-block:: python

    civiz = nest_run.get_CImodel(N=500, Nprocesses=4)

    civiz.plot_all()

    civiz.plot_massfunc(show_fields=True)

    civiz.plot_plot_cumulative_mass()

    plt.show()

Given the computing time it may require to compute the confidence intervals,
these outputs can also be saved and loaded from the same results file

.. code-block:: python

    out_filename = nest_run.file.filename

    civiz.save(out_filename)

    civiz = analysis.CIModelVisualizer.load(out_filename)

There also exists a handy command-line script for generating and saving
confidence intervals to later be loaded in python. For more information,
see the help page:

.. code-block:: bash

    generate_model_CI --help


Collections of Runs
^^^^^^^^^^^^^^^^^^^

When analyzing multiple runs (for a single or many different clusters),
the :class:`gcfit.analysis.RunCollection` class allows for easy interaction
with, and comparison of, all runs at the same time.

.. code-block:: python

    rc = analysis.RunCollection.from_dir('nested_out')

    # Plot side-by-side comparison of all a3 parameter distributions
    rc.plot_param_violins('a3')

    # Plot a3 vs mass for all clusters
    rc.plot_relation('M', 'a3', annotate=True)

    # Iteratively plot each runs params
    for _ in rc.iter_plots('plot_params'):
        plt.show()

    # Overplot all cluster a3 posterior distributions
    fig = plt.figure()
    for _ in rc.iter_plots('plot_posterior', param='a3', fig=fig, flipped=False, alpha=0.3):
        pass

This class also provides access to collections of corresponding model outputs.

.. code-block:: python

    mc = rc.get_CImodels(load=True)

    # Iteratively plot each models profiles
    for _ in mc.iter_plots('plot_all'):
        plt.show()

    # Compare run parameters to certain model outputs, like remnannt fractions and BH mass
    rc.plot_relation('a3', 'f_rem')

    rc.plot_param_violins('BH_mass')
