=====
Usage
=====

``GCfit`` has two functionalities. The primary, to fit GC equilibrium models
against a number of observables, and the secondary, to analyze and visualize
the statistical fitting process and the resultant models.

Fitting
=======

Python
^^^^^^

The ``GCfit`` fitting process can be accessed through the python interface
using the core fitting functions.

.. code-block:: python

    import fitter
    cluster = 'NGC104'

    # MCMC Sampling
    fitter.MCMC_fit(cluster, Niters=3000, Nwalkers=32)

    # Nested Sampling
    fitter.nested_fit(cluster, eff_samples=5000)

.. things that go into that are in common

Both fitting functions have a large assortment of keyword arguments, which
define the sampler settings, the sampler duration and stopping conditions,
the component likelihoods, the relevant input and ouput file locations and the
parallelization scheme. Some are shared between both functions, while some are
specific to the method used.

Both functions begin with the ``cluster`` argument, which is simply a string
specifying the name of the globular cluster to fit on. The cluster must have
accompanying observations (see TODO link to obs) and the string should match the
standard cluster name. See (TODO link to util func) for more information on
what standards are acceptable.

.. ncpu,mpi
The parallelization schemes can be specified using the ``Ncpu`` and ``mpi``
arguments. ``Ncpu`` sets the number of processes to parallelize the sampling
computation over locally, using the ``multiprocessing`` module. The boolean
``mpi`` flag allows for the fitting to be run over multiple nodes using MPI.
If :code:`True`, the ``Ncpu`` parameter is ignored, and the MPI setup must be
specified when running the code using an MPI-execution utility (e.g. ``mpirun``,
``mpiexec``, etc.). All parallelization is handled in the same way: a pool is
created using the ``schwimmbad`` library, and passed to the relevant ``emcee``
or ``dynesty`` sampler functions.

.. likelihood funcs stuff
The makeup of the probability functions and how they're sampled can be altered
by a few shared arguments.
Any of the 13 typically free parameters can be specified by the ``fixed_params``
argument, in order to fix them to a constant value, set by the initial values,
essentially removing them from the sampling.
Any of the likelihood components can be similarly ignored (excluded from the
overall likelihood sum) using the ``excluded_likelihoods`` argument.
The ``initials`` argument sets the initial position of the sampler in each
parameter dimension. In the MCMC sampler, these initial positions define the
starting positions of the walkers, while in the nested sampling the initial
values are only used for fixed parameters. Any parameters not given to this
argument take their initial values from the initials stored in the relevant
cluster observations data file.

.. priors
The prior distributions, for each free parameter, can be specified using the
``param_priors`` argument. Each entry of the dict should specify the type of
prior distribution, and the relevant parameters of said distribution. See (TODO)
for more details.

.. things specific to MCMC

The MCMC sampler 

Niters, Nwalkers

moves

.. things specific to nested

bounds

samples

init, batch kwargs

pfrac

maxfrac

eff_samples

.. examples of how to do some things

this is examples


Script
^^^^^^

.. introduce the GCfitter script

.. describe things specific to script, how to run it, parallelism

.. direct to help page

.. some examples of how to do things, including in parallel, with job queue


Analysis
========

.. output files

.. run visualizers
.. common plots/stats
.. specifics to each kind

.. model visualizers
.. CI visualizers
