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

All plotting functions return their corresponding figure, and multiple plots
can be "stacked" onto one another. See the source API for each to find more
information, and a list of all possible plots.

.. code-block:: python

    >>> from gcfit import analysis
    >>> import matplotlib.pyplot as plt

    >>> obs = gcfit.Observations(cluster)

    >>> gcfit.nested_fit(cluster, savedir='./nested_out')
    >>> gcfit.MCMC_fit(cluster, savedir='./MCMC_out')


Fitting Results
^^^^^^^^^^^^^^^

The run visualizers are split into specific classes once again for the MCMC
(:class:`gcfit.analysis.MCMCRun`) and nested sampler
(:class:`gcfit.analysis.NestedRun`) results.

.. code-block:: python

    >>> nest_run = analysis.NestedRun(f'./nested_out/{cluster}_sampler.hdf', obs)
    >>> mcmc_run = analysis.MCMCRun(f'./MCMC_out/{cluster}_sampler.hdf', obs)

    >>> # Plot nested sampling parameter evolution, weights and final posteriors
    >>> nest_run.plot_params()
    <Figure size 640x480 with 30 Axes>

    >>> # Plot MCMC walker evolution
    >>> mcmc_run.plot_chains()
    <Figure size 640x480 with 13 Axes>

    >>> # Plot marginal distributions for both (corner plots)
    >>> nest_run.plot_marginals()
    <Figure size 640x480 with 169 Axes>

    >>> plt.show()


Best Fit Models
^^^^^^^^^^^^^^^

The fitting results can be used to determine the best-fit parameters, and
corresponding confidence intervals, which in turn describe the best-fitting
model. From there, plots of all observables, as well as a number of other
cluster parameters and profiles, can be created.

The median best-fit model can be visualized with the
:class:`gcfit.analysis.ModelVisualizer` class.

.. code-block:: python

    >>> mviz = nest_run.get_model(method='mean')
    >>> mviz
    <gcfit.analysis.models.ModelVisualizer object at 0x7f434e7e9840>

    >>> # Plot all radial profiles (dispersions, number density, etc)
    >>> mviz.plot_all()
    <Figure size 640x480 with 6 Axes>

    >>> # Plot all mass functions (with fields shown)
    >>> mviz.plot_massfunc(show_fields=True)
    <Figure size 640x480 with 9 Axes>

    >>> # Plot cumulative mass in all stellar components
    >>> mviz.plot_plot_cumulative_mass()
    <Figure size 640x480 with 1 Axes>

    >>> plt.show()

Profiles corresponding to any mass bin, not only those comparable to the
observations, can be shown alongside using the `mass_bins` argument to any
plotting function:

.. code-block:: python

    >>> # Plot alongside profiles of lightest stars and heaviest remnants
    >>> extra_masses = [0, -1]
    >>> mviz.plot_pm_tot(mass_bins=extra_masses)
    <Figure size 640x480 with 1 Axes>

    >>> plt.show()

All the same plots can instead be shown with confidence intervals on the
model outputs (:class:`gcfit.analysis.CIModelVisualizer`). The computation
of these intervals may be intensive, and can thus be parallelized (locally)
using the ``Nprocesses`` keyword.

.. code-block:: python

    >>> civiz = nest_run.get_CImodel(N=500, Nprocesses=4)

    >>> civiz.plot_all()
    <Figure size 640x480 with 6 Axes>

    >>> civiz.plot_massfunc(show_fields=True)
    <Figure size 640x480 with 21 Axes>

    >>> civiz.plot_plot_cumulative_mass()
    <Figure size 640x480 with 1 Axes>

    >>> plt.show()

Note that, unless extra tracer masses are used during fitting, profiles for
only a single mass bin will be generated to save on memory and time.

Given the computing time it may require to compute the confidence intervals,
these outputs can also be saved and loaded from the same results file:

.. code-block:: python

    >>> out_filename = nest_run.file.filename

    >>> civiz.save(out_filename)

    >>> civiz = analysis.CIModelVisualizer.load(out_filename)

There also exists a handy command-line script for generating and saving
confidence intervals to later be loaded in python. For more information,
see the help page:

.. code-block:: bash

    generate_model_CI --help


Plotting Specific Models and Observations
"""""""""""""""""""""""""""""""""""""""""

All of these model visualizations can also be used to examine specific models,
not necessarily based on any fitting results, though they will of course not
have any comparisons to observed datasets.

.. code-block:: python

    >>> model = gcfit.Model(W0=6.3, M=5e5, rh=6.7, age=12, FeH=-0.7)
    
    >>> mv = analysis.ModelVisualizer(model)

    >>> mv.plot_cumulative_mass()
    <Figure size 640x480 with 1 Axes>

    >>> plt.show()


Similarly, visualizations of observational datasets, without any corresponding
models, can also be done.

.. code-block:: python

    >>> obs = gcfit.Observations('NGC104')
    
    >>> ov = analysis.ObservationsVisualizer(obs)

    >>> ov.plot_number_density(show_background=True)
    <Figure size 640x480 with 1 Axes>

    >>> plt.show()


Collections of Runs
^^^^^^^^^^^^^^^^^^^

When analyzing multiple runs (for a single or many different clusters),
the :class:`gcfit.analysis.RunCollection` class allows for easy interaction
with, and comparison of, all runs at the same time.

.. code-block:: python

    >>> rc = analysis.RunCollection.from_dir('nested_out')

    >>> # Plot side-by-side comparison of all a3 parameter distributions
    >>> rc.plot_param_violins('a3')
    <Figure size 640x480 with 1 Axes>

    >>> # Plot a3 vs mass for all clusters
    >>> rc.plot_relation('M', 'a3', annotate=True)
    <Figure size 640x480 with 1 Axes>

    >>> # Iteratively plot each runs params
    >>> for _ in rc.iter_plots('plot_params'):
    >>>     plt.show()

    >>> # Overplot all cluster a3 posterior distributions
    >>> fig = plt.figure()
    >>> for _ in rc.iter_plots('plot_posterior', param='a3', fig=fig, flipped=False, alpha=0.3):
    >>>     pass

    >>> plt.show()

This class also provides access to collections of corresponding model outputs.

.. code-block:: python

    >>> mc = rc.get_CImodels(load=True)
    >>> mc
    <gcfit.analysis.models.ModelCollection object at 0x7f49ccb986a0>

    >>> # Iteratively plot each models profiles
    >>> for _ in mc.iter_plots('plot_all'):
    >>>     plt.show()

    >>> # Compare run parameters to certain model outputs, like remnannt fractions and BH mass
    >>> rc.plot_relation('a3', 'f_rem')
    <Figure size 640x480 with 1 Axes>

    >>> rc.plot_param_violins('BH_mass')
    <Figure size 640x480 with 1 Axes>
