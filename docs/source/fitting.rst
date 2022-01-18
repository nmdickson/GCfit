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

.. list-table::
    :header-rows: 1
    :widths: 10 10 80

    * - Parameter
      - Name
      - Description
    * - :math:`\hat{\phi}_0`
      - ``W0``
      - The central potential. Used as a boundary condition
        for solving Poisson’s equation and defines how centrally concentrated
        the model is.
    * - M
      - ``M``
      - The total mass of the system, in all mass components. In units of
        :math:`10^6 M_\odot`.
    * - :math:`r_h`
      - ``rh``
      - The system half-mass radius, in parsecs.
    * - :math:`r_a`
      - ``ra``
      - The anisotropy-radius, which determines the amount of anisotropy in the
        system (higher ra values indicate more isotropy)
    * - g
      - ``g``
      - The truncation parameter, which controls the sharpness of the outer
        density truncation of the model
    * - :math:`\delta`
      - ``delta``
      - Sets the mass dependance of the velocity scale for each mass component.
        Maximum value of 0.5
    * - :math:`\alpha_1`
      - ``a1``
      - The low-mass IMF exponent (0.1 to 0.5 :math:`M_\odot`)
    * - :math:`\alpha_2`
      - ``a2``
      - The intermediate-mass IMF exponent (0.5 to 1.0 :math:`M_\odot`)
    * - :math:`\alpha_3`
      - ``a3``
      - The high-mass IMF exponent (1.0 to 100 :math:`M_\odot`)
    * - :math:`\mathrm{BH}_{ret}`
      - ``BHret``
      - The percentage of black holes retained after dynamical ejections
    * - :math:`s^2`
      - ``s2``
      - Nuisance parameter applied as an additional unknown uncertainty to all
        number density profiles, allowing for small deviations between
        the outer parts of the model and observations
    * - F
      - ``F``
      - Nuisance parameter applied as an additional unknown uncertainty to all
        mass function profiles encapsulating possible additional sources of
        uncertainty
    * - d
      - ``d``
      - Distance to the cluster, in kiloparsecs. Mainly used for all conversions
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

where the :math:`\alpha` parameters are defined by the free parameters above,
and :math:`\xi(m) \Delta m` is the number of stars with masses within the range
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

TODO some example plots of each type

* Proper Motion Dispersions
    Radial, tangential or overall proper motion velocity dispersion profiles

* LOS Velocity Dispersions
    Velocity dispersion profiles along the line-of-sight
    
* Number Densities
    Radial number density profiles

* Mass Functions
    Present day stellar mass functions (counts), binned radially and in mass.
    Each dataset corresponds to an observed field on the sky, whose boundaries
    must also be included.

* Pulsars Timing Solutions
    Millisecond-pulsar timing solutions (period and period derivative), used to
    constrain possible acceleration from the cluster potential

While these are the type of observables supported, not all are required at once,
and multiple datasets corresponding to one type are permitted.


Probabilities
=============

The probability associated with a given set of model :math:`M` parameters
:math:`\Theta`, subject to some number of observable datasets :math:`\mathcal{D}` is
given by a simple bayesian posterior:

.. math::
    
    P(\Theta \mid \mathcal{D}, M) = \frac{P(\mathcal{D} \mid \Theta,M)
                                    P(\Theta \mid M)}{P(\mathcal{D} \mid M)}
                        = \frac{\mathcal{L}(\Theta) \pi(\Theta)}{\mathcal{Z}}

where :math:`\mathcal{L}` is the likelihood and :math:`\pi` is the prior
likelihood.

Likelihoods
^^^^^^^^^^^

The total (log) likelihood function :math:`\ln(\mathcal{L})` is given simply by
the summation of all component likelihood functions.

.. math::

    \ln(\mathcal{L}) = \sum_i^{\rm{datasets}} \ln(P(\mathcal{D_i} \mid \Theta))
                     = \sum_i \ln(\mathcal{L}_i(\Theta)))

Every observational dataset has it's own component likelihood function, unique
to the type of observable it is.

All velocity dispersions (LOS and PM) use a simple gaussian log-likelihood over
a number of dispersion measurements at different radial distances:

.. math::

    \ln(\mathcal{L}_i) = \frac{1}{2} \sum_r \left( \frac{(\sigma_{\rm{obs}}(r)
                    - \sigma_{\rm{model}}(r))^2}{\delta\sigma_{\rm{obs}}^2(r)}
                    - \ln(\delta\sigma_{\rm{obs}}^2(r))\right)

where :math:`\sigma(R)` corresponds to the dispersion at a distance
:math:`R` from the cluster centre, with corresponding uncertainties
:math:`\delta\sigma(R)`.

Number density datasets use a modified gaussian likelihood.
As the translation between discrete number density and surface-brightness
observations is difficult to quantify, the model is actually only fit on
the shape of the number density profile data.
To accomplish this the modelled number density is scaled to have the
same mean value as the surface brightness data.
The constant scaling factor K is chosen to minimize the chi-squared distance:

.. math::
    
    K = \frac{\sum\limits_r \Sigma_{obs} \Sigma_{model} / \delta\Sigma^2}
             {\sum\limits_r \Sigma_{model}^2 / \delta\Sigma^2}

The likelihood is then given in similar fashion to the dispersion profiles:

.. math::

    \ln(\mathcal{L}_i) = \frac{1}{2} \sum_r \left( \frac{(\Sigma_{\rm{obs}}(r) - K\Sigma_{\rm{model}}(r))^2}{\delta\Sigma^2(r)} - \ln(\delta\Sigma^2(r))\right)

where :math:`\Sigma(R)` is the number density at distance :math:`R`.

The error :math:`\delta\Sigma` in these equations includes both the
uncertainties from the observed datasets and an added constant error over the
entire profile, defined by the nuisance parameter ``s2`` (:math:`s^2`), which
helps to minimize the background effects present near the outskirts of the
cluster.

.. math::
    \delta\Sigma^2(R) = \delta\Sigma_{\rm{obs}}^2(R) + s^2

To compare against the Mass function datasets, the model surface density is
(Monte Carlo) integrated, within each dataset's corresponding field boundaries,
over each radial bin :math:`j` (with bounds :math:`r0,\ r1`) to get the count
:math:`N_{\rm{model},j}` of stars within this bin slice of the field:

.. math::

    N_{\rm{model},j} = \int_{r_0}^{r_1} \Sigma(r) dr

This count can be used in the usual gaussian likelihood:

.. math::

    \ln(\mathcal{L}_i) = \frac{1}{2} \sum_j^{\rm{bins}}
        \left( \frac{(N_{\rm{obs},j} - N_{\rm{model},j})^2}{\delta N_j^2}
              - \ln(\delta N_j^2) \right)

where the error :math:`\delta N` also includes the nuisance parameter ``F``
which acts to account for unknown sources of error in the mass function counts
by scaling upwards the uncertainties in the counts:

.. math::

    \delta N_j = \delta N_{\rm{model},j} \cdot F

TODO pulsar likelihoods

Priors
^^^^^^

The prior likelihood :math:`\pi` for some set of parameters :math:`\Theta`
is given by the product of individual priors on each parameter in
:math:`\Theta`, designed to influence the possible values for each.
These priors are defined, a priori, by a few arguments specific to each,
which may also be dependant on the values of other parameters.

.. math::
    \pi(\Theta) = \prod_i^{N_{\rm{params}}} \pi_i (\theta_i)

Individual parameter priors can take a few possible forms:

* Uniform (L, U)
    A uniform (flat) distribution defined between two bounds (L, U), with a
    value normalized to unity

.. math::

    \pi_i (\theta_i) =
    \begin{cases}
        \frac{1}{U-L} & {\text{for }} \theta_i \in [L,U] \\
        0 & {\text{otherwise}}
    \end{cases}

* Gaussian (:math:`\mu`, :math:`\sigma`)
    A Gaussian normal distribution centred on :math:`\mu` with a width of
    :math:`\sigma`

.. math::
    \pi_i (\theta_i)  = \frac{1}{\sigma \sqrt{2\pi}}
    e^{-\frac{1}{2} \left(\frac{\theta_i-\mu}{\sigma}\right)^{2}}

* TODO other kinds


Sampling
========

The posterior distribution of the parameter set :math:`\Theta` must be
determined through a statistical sampling technique. Two such set of
algorithms are available in ``GCfit``.

TODO link to *_fit functions ref
TODO also link to a good "for more information" or at least a relevant paper

MCMC
^^^^

The first is **Markov Chain Monte Carlo (MCMC)** sampling.

MCMC sampling approximates the posterior distribution by
generating random samples within parameter space. Each sample is proposed
randomly, dependant only on the preceeding sample in the "chain" of samples
(resulting in a *Markov Chain*).

Chains must be initialized to initial positions
within parameter space, from which they will evolve over time towards areas of
high probability. There are a number of algorithms available
dictating the proposal and acceptance of new samples, which determines the
random path taken by chains. Samplers which utilize multiple chains run in
parallel are known as ensemble samplers.

``GCfit`` utilizes the `emcee <https://emcee.readthedocs.io>`_
MCMC ensemble sampler library.

.. Specifics about MCMC
.. introduce the MCMC_fit function
.. how it works, what we use to do it, any specific requirements from the user

Nested Sampling
^^^^^^^^^^^^^^^

The second is **Dynamic Nested Sampling**.

Nested sampling
(`Skilling 2004 <https://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S>`_)
is a Monte Carlo integration method, first proposed for estimating the Bayesian
evidence integral :math:`\mathcal{Z}`, which works by iteratively integrating
the posterior over the shells of prior volume contained within nested,
increasing iso-likelihood contours.

Samples are proposed randomly at each step, subject to a minimum likelihood
constraint corresponding to the current likelihood contour. These samples, and
their importance weights (a function of shell amplitude and volume, analogous
to the contribution to the typical set), can be used to estimate the posterior,
alongside the evidence integral.

Nested sampling has the benefit of flexibility, as the independantly generated
samples are able to probe complex posterior shapes, with little danger of
falling into local minimums, or of missing distant modes. The sampling also has
well defined stopping criterion based on the remaining evidence.

Dynamic Nested Sampling is an extension of the typical nested sampling algorithm
designed to retune the sampling to more efficiently estimate the posterior,
by spending less time probing the "outer" sections of the prior volume which
have little impact on the posterior. In practice this is done by allowing for
a fine-tuning of the sample "resolution", which is increased around the typical
set.

``GCfit`` utilizes the `dynesty <https://dynesty.readthedocs.io/>`_
Dynamic Nested Sampling package.

.. Specifics about Nested Sampling
.. the nested_fit function
.. how it works, what we use to do it, any specific requirements from the user,
.. prior transforms, plateau weights