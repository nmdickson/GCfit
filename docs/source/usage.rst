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

The MCMC fitting is executed with the ``MCMC_fit`` function:

.. code-block:: python

    MCMC_fit
    
The Nested Sampling is executed with the ``nested_fit`` function:

.. code-block:: python

    nested_fit


.. things that go into that are in common

.. things specific to MCMC
.. things specific to nested

.. examples of how to do some things


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
