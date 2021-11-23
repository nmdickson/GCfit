====================================
Fitting Globular Clusters with GCfit
====================================

The main objective of ``GCfit`` is to determine the best-fit parameters of a
globular cluster (GC) equilibrium model, subject to a number of observed
physical datasets, using statistical sampling over a Bayesian posterior.

Models
======

In systems like globular clusters where the two-body relaxation time
far exceeds the system's dynamical time-scale, the phase-space
distribution (positions and velocities) of stars in a cluster can
be described by distribution-function (DF) models.
DF models are equilibrium models built around a distribution function
which describes the particle density of stars and satisfies the
collisionless-Boltzmann equation

``GCfit`` uses an extension of the `LIMEPY <https://github.com/mgieles/limepy>`_
family of DF models to describe the clusters.
As described in `(Gieles and Zocchi, 2015) <https://ui.adsabs.harvard.edu/abs/
2015MNRAS.454..576G>`_, LIMEPY is:

    a family of self-consistent, spherical, lowered isothermal models,
    consisting of one or more mass components, with parametrized prescriptions
    for the energy truncation and for the amount of radially biased pressure
    anisotropy.

Parameters
^^^^^^^^^^

The models used in ``GCfit`` are defined by 13 free parameters.

6 physical parameters defining the system structure:

#. W0
    The central potential :math:`\hat{phi}_0`. Used as a boundary condition for
    solving Poisson’s equation and defines how concentrated the model is.
#. M
    The total mass of the system, in all mass components. In units of
    :math:`10^6 M_\odot`.
#. rh
    The system half-mass radius, in parsecs.
#. ra
    The anisotropy-radius, which determines the amount of anisotropy in the
    system (higher ra values indicate more isotropy)
#. g
    the truncation parameter g, which controls the sharpness of the truncation
    of the model
#. delta
    sets the mass dependance of the velocity scale for each mass component
    Maximum value of 1/2

4 parameters defining the initial mass function:

#. a1
    The low-mass IMF exponent (0.1 to 0.5 :math:`M_\odot`)
#. a2
    The intermediate-mass IMF exponent (0.5 to 1.0 :math:`M_\odot`)
#. a3
    The high-mass IMF exponent (1.0 to 100 :math:`M_\odot`)
#. BHret
    The percentage of black holes retained after dynamical ejections

and 3 remaining parameters to aid in model fitting:

#. s2
    Nuisance parameter applied as an additional unknown uncertainty to all
    number density profiles, allowing for small deviations between
    the outer parts of the model and observations
#. F
    Nuisance parameter applied as an additional unknown uncertainty to all
    mass function profiles encapsulating possible additional sources of
    uncertainty
#. d
    Distance to the cluster, in kiloparsecs. Mainly used for all conversions
    between observational (angular) and and model (linear) units.

SSPTOOLS, IMFS

Observations
============

Probabilities
=============

MCMC
====

Specifics about MCMC
introduce the MCMC_fit function
how it works, what we use to do it, any specific requirements from the user

Nested Sampling
===============

Specifics about Nested Sampling
the nested_fit function
how it works, what we use to do it, any specific requirements from the user,
prior transforms, plateau weights