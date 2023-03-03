===================================
Globular Cluster Equilibrium Models
===================================

In systems like globular clusters (GCs) where the two-body relaxation time
far exceeds the system's dynamical time-scale, the phase-space
distribution (positions and velocities) of stars in a cluster can
be described by distribution-function (DF) models.
DF models are equilibrium models built around a distribution function
which describes the particle density of stars and satisfies the
collisionless-Boltzmann equation

``GCfit`` uses an extension of the `LIMEPY <https://github.com/mgieles/limepy>`_
family of DF models to describe the phase-space distribution of stars and
remnants within the clusters, alongside a mass-evolution algorithm to determine
the makeup of said populations.
As described in `Gieles and Zocchi (2015) <https://ui.adsabs.harvard.edu/abs/
2015MNRAS.454..576G>`_, LIMEPY is:

    a family of self-consistent, spherical, lowered isothermal models,
    consisting of one or more mass components, with parametrized prescriptions
    for the energy truncation and for the amount of radially biased pressure
    anisotropy.

Models
======

``GCfit`` provides easy access to a few different versions of the GC models,
based on LIMEPY, through the :class:`gcfit.core.data.Model` class.
This includes both single and multimass cluster models (though
we recommend only using multimass models, as singlemass models are insufficient
to accurately describe the processes in real GCs), as well as isotropic and
anisotropic models.

The typically used historical DF models (King, Woolley, etc.) can also be
easily recovered by controlling the shape of the model truncation, as described
below.

One of the main objective of ``GCfit`` is to determine the best-fit parameters
of these models, subject to a number of observed physical datasets, using
statistical sampling techniques. To accomplish this, a subset of these models
defined using only the 13 key parameters is also provided.


Parameters
^^^^^^^^^^

The fittable models used in ``GCfit`` are defined by 13 free parameters,
detailed below.

These parameters comprise the main set used to define all of the models,
while any other available arguments are documented in
:class:`gcfit.core.data.Model`.


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
      - The total mass of the system, in all mass components.
        .. In units of :math:`10^6 M_\odot`.
    * - :math:`r_h`
      - ``rh``
      - The system half-mass radius, in parsecs.
    * - :math:`\log(\hat{r}_a)`
      - ``ra``
      - The (log) anisotropy-radius, which determines the amount of anisotropy
        in the system (higher ra values indicate more isotropy).
    * - g
      - ``g``
      - The truncation parameter, which controls the sharpness of the outer
        density truncation of the model. Certain specific values will recover
        typically used DF models (i.e. ``g=1`` results in King (1966) models).
    * - :math:`\delta`
      - ``delta``
      - Sets the mass dependance of the velocity scale for each mass component.
        Maximum value of 0.5, typical of fully mass-segregated systems.
    * - :math:`\alpha_1`
      - ``a1``
      - The low-mass IMF exponent (0.1 to 0.5 :math:`M_\odot`).
    * - :math:`\alpha_2`
      - ``a2``
      - The intermediate-mass IMF exponent (0.5 to 1.0 :math:`M_\odot`).
    * - :math:`\alpha_3`
      - ``a3``
      - The high-mass IMF exponent (1.0 to 100 :math:`M_\odot`).
    * - :math:`\mathrm{BH}_{ret}`
      - ``BHret``
      - The percentage of black holes retained after dynamical ejections and
        natal kicks.
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
        between observational (angular) and model (linear) units.


Mass Function Evolution
^^^^^^^^^^^^^^^^^^^^^^^

The evolution of stars within the model from initial mass function, over
the age of the cluster, to the present day stellar and remnant mass function
is carried out using the `ssptools <https://github.com/SMU-clusters/ssptools>`_
library. This library is a fork based on the original algorithms of
`Balbinot and Gieles (2018) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.2479B>`_.

The initial mass function (IMF) is defined by a broken power-law 
distribution function:

.. math::

    \xi (m) \propto \begin{cases}
        m^{-\alpha_1} & 0.1\ M_\odot < m \leq 0.5\ M_\odot \\
        m^{-\alpha_2} & 0.5\ M_\odot < m \leq 1\ M_\odot \\
        m^{-\alpha_3} & 1\ M_\odot < m \leq 100\ M_\odot \\
    \end{cases}

where the :math:`\alpha_i` parameters are defined by the free parameters above,
and :math:`\xi(m) \Delta m` is the number of stars with masses within the
interval :math:`m + \Delta m`. This function determines the initial distribution
of the cluster total mass.

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
The amount and final mass of these remnants must then be scaled downwards to
mimic the loss of newly formed remnants. By default a neutron star retention
fraction of 10% is assumed.

This algorithm includes two more complicated
prescriptions for the loss of black holes, accounting for dynamical
ejections on top of the typical natal kicks.
Firstly the ejection of, primarily low-mass, BHs through natal kicks is
simulated. Beginning with the assumption that the kick velocity is drawn from a
Maxwellian distribution with a dispersion of 265 km/s (scaled down by a
"fallback fraction" interpolated from a grid of SSE models), the fraction of
black holes retained in each mass bin is then found by integrating the
kick velocity distribution from 0 to the estimated initial system escape
velocity.
Black holes are also ejected over time from the core of GCs due to dynamical
interactions with one another. This
process is simulated through the removal of BHs, beginning with the heaviest
mean-mass bins through to the lighest. This is carried
out iteratively until the combination of mass lost through both the natal
kicks and these dynamical ejections equals the fraction of BHs specified by
the :math:`\mathrm{BH}_{ret}` parameter.

The final avenue for cluster mass loss is through the escape of stars and
remnants driven by two-body relaxation and lost to the potential of the host
galaxy. Such losses, in a mass segregated cluster, are dominated by
the escape of low-mass objects from the outer regions of the cluster.
Determining the overall losses through this process is a complicated task,
dependent on the dynamical history and orbital evolution of the cluster,
which we do not attempt to model here.
By default, we opt to ignore this preferential
loss of low-mass stars and do not further model the escape of any
stars, apart from through the processes described above.
