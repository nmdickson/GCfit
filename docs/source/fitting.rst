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

* W0
    The central potential :math:`\hat{\phi}_0`. Used as a boundary condition for
    solving Poisson’s equation and defines how concentrated the model is.
* M
    The total mass of the system, in all mass components. In units of
    :math:`10^6 M_\odot`.
* rh
    The system half-mass radius, in parsecs.
* ra
    The anisotropy-radius, which determines the amount of anisotropy in the
    system (higher ra values indicate more isotropy)
* g
    The truncation parameter, which controls the sharpness of the outer density
    truncation of the model
* delta
    Sets the mass dependance of the velocity scale for each mass component.
    Maximum value of 0.5

4 parameters defining the mass function:

* a1
    The low-mass IMF exponent (0.1 to 0.5 :math:`M_\odot`)
* a2
    The intermediate-mass IMF exponent (0.5 to 1.0 :math:`M_\odot`)
* a3
    The high-mass IMF exponent (1.0 to 100 :math:`M_\odot`)
* BHret
    The percentage of black holes retained after dynamical ejections

and 3 remaining parameters to aid in model fitting:

* s2
    Nuisance parameter applied as an additional unknown uncertainty to all
    number density profiles, allowing for small deviations between
    the outer parts of the model and observations
* F
    Nuisance parameter applied as an additional unknown uncertainty to all
    mass function profiles encapsulating possible additional sources of
    uncertainty
* d
    Distance to the cluster, in kiloparsecs. Mainly used for all conversions
    between observational (angular) and and model (linear) units.


Mass Function Evolution
^^^^^^^^^^^^^^^^^^^^^^^

The evolution of stars within the model from initial mass function, over
the age of the cluster, to the present day stellar and remnant mass function
is carried out using the `ssptools` library.

The initial mass function (IMF) is defined by a broken power-law 
distribution function:

.. math::

    \xi (m) \propto \begin{cases}
        m^{-\alpha_1} & 0.1\ M_\odot < m \leq 0.5\ M_\odot \\
        m^{-\alpha_2} & 0.5\ M_\odot < m \leq 1\ M_\odot \\
        m^{-\alpha_3} & 1\ M_\odot < m \leq 100\ M_\odot \\
    \end{cases}

where the :math:`\alpha` parameters are defined by the parameters above, and
:math:`\xi(m) \Delta m` is the number of stars with masses within the range
:math:`m + \Delta m`. This function determines the initial distribution of the
cluster total mass.

To evolve to the present day population of stars, the rate of change of
main-sequence stars in each mass bin is given by the equation:

.. math::

    \dot{N} (m_{to}) = - \left.\frac{dN}{dm}\right|_{m_{to}} \left|\frac{dm_{to}}{dt}\right|

where the rate of change of the turn-off mass (:math:`m_{to}`) can be derived
from an approximation of the main-sequence lifetime of stars, based on stellar
evolution models (Dotter et al. 2007, 2008).

As these stars evolve off of the main sequence, the stellar remnants they will
form will depend (both in type and in mass) on their initial mass and the
metallicity of the cluster. The maximum initial mass which will form a white
dwarf and the minimum initial mass which allows for black hole formation are
determined from stellar evolution models, and vary with metallicity.
Initial-final mass relations (IFMRs) for both are also determined from these
models and are similarly metallicity-dependant. All stars within this mass
range will form neutron stars, always with a mass of :math:`1.4\ M_\odot`.

The other avenue for mass loss is through the escape of stars and
remnants past the cluster tidal radius, lost to the potential of the host
galaxy.
Stellar losses are dominated by the escape of low-mass stars in the outer edges
of the cluster. Lacking a precise method for determining the overall losses,
which will depend on the cluster potential and galactic orbit, we opt to
disallow the escape of any stars.

TODO Remnant losses

Observations
============

All cluster models are fit by comparing model structures against a number of
observed datasets containing measured information on the structure and
kinematics of a specific cluster.

Currently supported observational datasets include:

* Proper Motion Dispersions
    Radial, tangential or overall proper motion velocity dispersion profiles

* LOS Velocity Dispersions
    Velocity dispersion profiles along the line-of-sight
    
* Number Densities
    Radial number density profiles

* Mass Functions
    Present day stellar mass functions (counts), binned radially and in mass

* Pulsars Timing Solutions
    Millisecond-pulsar timing solutions (period and period derivative), used to
    constrain possible acceleration from the cluster potential


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