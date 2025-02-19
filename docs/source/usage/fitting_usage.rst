Fitting
=======

While these models can be useful on their own, one of the key objectives of
``GCfit`` is to determine the posterior distributions of the most important
parameters defining these models.

This fitting is based on top of a different model subclass;
:class:`gcfit.core.FittableModel`.

This class is nearly identical to the base ``Model``, except for how it is
initialized: based on an array of sampled values for each of the 13 main fitting
parameters, in a specific order, and an ``Observations`` object.

Note: The order and units required of the parameters for this class may not
match those in :class:`gcfit.core.Model`. It is recommended to only access the base
model directly, and leave this class for use by the fitting functions below.

Python
^^^^^^

The ``GCfit`` fitting process can be accessed through the python interface as
well.

There are two core fitting functions, one for each sampling method. Both come
with many optional arguments, some shared between the two, and some specific
to the method chosen.

Both functions require, to begin, the name of the cluster.

.. code-block:: python

    >>> cluster = 'NGC104'

    >>> # MCMC Sampling
    >>> gcfit.MCMC_fit(cluster, Niters=3000, Nwalkers=32)

    >>> # Nested Sampling
    >>> gcfit.nested_fit(cluster)

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

    >>> gcfit.nested_fit(cluster, fixed_params=['M', 'rh'])

The initial values that these parameters are fixed to are defined by the
``initials`` argument. These also act as the initial positions for the MCMC
sampler, and the default values for each parameter are set within each data
file.

.. code-block:: python

    >>> gcfit.nested_fit(cluster, fixed_params=['M'], initials={'M': 0.5})

The posterior is typically made up of a sum of component likelihoods, which
act on a specific dataset each. The component likelihood functions, and
types of datasets, can be excluded from the posterior using the
``excluded_likelihoods`` argument.

.. code-block:: python

    >>> excluded_L = ['proper_motion/GEDR3', 'pulsar*']  # glob patterns can be used
    >>> gcfit.nested_fit(cluster, excluded_likelihoods=excluded_L)


The posterior also includes prior probabilities on each free parameter. These
probability funnctions may also be specified using the ``param_priors``
argument. Priors are handled by the :class:`gcfit.probabilities.priors.Priors`
class. The ``param_priors`` argument accepts a dict of param-prior pairs,
where each entry must specify the type and relevant parameters of a prior
distribution.

.. code-block:: python

    >>> priors = {
    >>>     "d": ("Gaussian", 5.2, 0.01), # Gaussian priors specify mean and width
    >>>     "M": ("Uniform", [(0, 1.2)]), # Uniform priors specify a list of bounds
    >>>     "a2": ("Uniform", [(0, 4), ('a1', 4)]), # Other params can be used as bounds
    >>> }

    >>> gcfit.nested_fit(cluster, param_priors=priors)


Parallelization
"""""""""""""""

In the vast majority of cases, the sampler will be too resource-intensive
to be viably run on a single-core computer. The sampling, however, can be
easily parallelized in multiple ways.

Local parallelization (through the multiprocessing module) can be triggered
using the ``Ncpu`` argument, which simply accepts an integer number of processes
to spawn.

.. code-block:: python

    >>> import multiprocessing
    >>> max_cpu = multiprocessing.cpu_count()

    >>> gcfit.nested_fit(cluster, Ncpu=max_cpu)

To run the fitting over multiple nodes, using MPI, the boolean ``mpi`` flag
can be specified. If using ``mpi``, the ``Ncpu`` argument is ignored, and the
number of processes must be specified when running the code using an
MPI-execution utility (``mpirun``, ``mpiexec``, etc.).

.. code-block:: python

    >>> # Run script with e.g. mpiexec -n 4 python script.py
    >>> gcfit.nested_fit(cluster, mpi=True)

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

    >>> gcfit.MCMC_fit(cluster, Niters=1500, Nwalkers=100)


Nested Sampler Specific
"""""""""""""""""""""""
.. things specific to nested

The progression of dynamic nested sampling requires defining both the sampler
parameters and methods, the transition to dynamic sampling, and the final
stopping conditions.

The base nested sampling algorithm works by randomly sampling within the
bounds defining a single iso-likelihood contour level. As such, both the random
sampling method, and the shape of the bounds can be specified. ``dynesty``
offers a variety of choices for both, see the source paper
(`2020MNRAS.493.3132S <https://adsabs.harvard.edu/abs/2020MNRAS.493.3132S>`_)
for more information on each.

.. code-block:: python

    >>> # Bounds can be one of {'none', 'single', 'multi', 'balls', 'cubes'}
    >>> bound = 'multi'

    >>> # Sampler can be one of {'unif', 'rwalk', 'rstagger', 'slice', 'rslice'}
    >>> sampler = 'rwalk'

    >>> gcfit.nested_fit(cluster, bound_type=bound, sample_type=sampler)

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

    >>> pfrac = 0.9  # 1 = 100% posterior focus, 0 = 100% evidence focus

    >>> maxfrac = 0.8  # percentage of the maximum weight, defining the new bounds

    >>> gcfit.nested_fit(cluster, pfrac=pfrac, maxfrac=maxfrac)

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

    >>> ESS = 5000

    >>> gcfit.nested_fit(cluster, pfrac=1, eff_samples=ESS)


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
