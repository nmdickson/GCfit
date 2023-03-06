====================================
Fitting Globular Clusters with GCfit
====================================

One of the main objective of ``GCfit`` is to determine the best-fit parameters
of a globular cluster equilibrium model, subject to a number of observed
physical datasets, using statistical sampling over a Bayesian posterior.

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
    Present day stellar mass functions (counts), binned radially and in mass.
    Each dataset corresponds to an observed field on the sky, whose boundaries
    must also be included.

* Pulsars Timing Solutions
    Millisecond-pulsar timing solutions (period and period derivative), used to
    constrain possible acceleration from the cluster potential

While these are the type of observables supported, not all are required at once,
and multiple datasets corresponding to each type are permitted.


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

Velocity Dispersions
""""""""""""""""""""

All velocity dispersions (LOS and PM) use a simple gaussian log-likelihood over
a number of dispersion measurements at different projected radial distances:

.. math::

    \ln(\mathcal{L}_i) = \frac{1}{2} \sum_j
        \left[
            \frac{(\sigma_{j,\mathrm{obs}}
            - \sigma_{j,\mathrm{model}})^2}
            {\delta\sigma_{j,\mathrm{obs}}^2}
            - \ln(\delta\sigma_{j,\mathrm{obs}}^2)
        \right],

where :math:`\sigma_j \equiv \sigma(r_j)` corresponds to the dispersion at a distance
:math:`r_j` from the cluster centre, with corresponding uncertainties
:math:`\delta\sigma_j \equiv \delta\sigma(r_j)`.
Dispersions with subscript *obs* correspond to the observed dispersions and
uncertainties, while subscript *model* corresponds to the predicted model
dispersions.

Number Densities
""""""""""""""""

Number density datasets use a modified gaussian likelihood.
As the translation between discrete number density and surface-brightness
observations is difficult to quantify, the model is actually only fit on
the shape of the number density profile data.
To accomplish this the modelled number density is scaled to have the
same mean value as the surface brightness data.
The constant scaling factor K is chosen to minimize the chi-squared distance:

.. math::
    
        K = \frac{\sum\limits_j \Sigma_{j,\mathrm{obs}} \Sigma_{j,\mathrm{model}}
                  / \delta\Sigma^2_j}
                 {\sum\limits_j \Sigma_{j,\mathrm{model}}^2 / \delta\Sigma^2_j},

where :math:`\Sigma_j \equiv \Sigma(r_j)` are the modelled and observed
number density, with respective subscripts, at a distance :math:`r_j` from the
cluster centre.

We also introduce an extra nuisance parameter (:math:`s^2`) to the fitting.
This parameter is added in quadrature, as a constant error over the entire
profile, to the observational uncertainties to give the overall error
:math:`\delta\Sigma`:

.. math::

        \delta\Sigma^2_j = \delta\Sigma_{j,\mathrm{obs}}^2 + s^2.


This parameter adds a constant uncertainty component over the entire radial
extent of the number density profile, effectively allowing for small
deviations in the observed profiles near the outskirts of the cluster.
This enables us to account for certain processes not captured by our models,
such as the effects of potential escapers.

The likelihood is then given in similar fashion to the dispersion profiles:

.. math::

        \ln(\mathcal{L}_i) = \frac{1}{2} \sum_j
            \left[
                \frac{(\Sigma_{j,\mathrm{obs}}
                - K\Sigma_{j,\mathrm{model}})^2}{\delta\Sigma^2_j}
                - \ln(\delta\Sigma^2_j)
            \right].


Mass Functions
""""""""""""""

To compare the models against the mass function datasets,
the local stellar mass functions are extracted from the models within
specific areas in order to match the observed MF data at different projected
radial distances from the cluster centre within their respective HST fields.

To compute the stellar mass functions, the model surface density in a given
mass bin :math:`\Sigma_k(r)` is integrated, using a Monte Carlo method,
over the area :math:`A_j`, which covers
a radial slice of the corresponding HST field from the projected distances
:math:`r_j` to :math:`r_{j+1}`. This gives the count
:math:`N_{\mathrm{model},k,j}` of stars within this footprint :math:`j`
in the mass bin :math:`k`:

.. math::

    N_{\mathrm{model}, k, j} = \int_{A_j} \Sigma_k(r) dA_j.


This star count can then be used to compute the Gaussian likelihood:

.. math::

    \ln(\mathcal{L}_i) = \frac{1}{2}
        \sum_j^{\substack{\mathrm{radial}\\\mathrm{bins}}}
        \sum_k^{\substack{\mathrm{mass}\\\mathrm{bins}}}
        \left[
            \frac{(N_{\mathrm{obs},k,j} - N_{\mathrm{model},k,j})^2}
                    {\delta N_{k,j}^2}
              - \ln(\delta N_{k,j}^2)
        \right],

which is computed separately for each HST program considered.

The error term :math:`\delta N_{k,j}` must also account for unknown and
unaccounted for sources of error in the mass function counts, as well as the
fact that our assumed parametrization of the global mass function may not be
a perfect representation of the data.
Therefore we include another nuisance parameter (:math:`F`) which scales up
the uncertainties:

.. math::

    \delta N_{k,j} = F \cdot \delta N_{\mathrm{obs},k,j}.

This scaling, rather than adding in quadrature as with the \(s^2\)
nuisance parameter, boosts the errors by a constant factor.
This allows it to capture additional unaccounted-for uncertainties
(e.g. in the completeness correction or limitations due to the simple
parametrization of the mass function) across the full range of values of
star counts, while simply adding the same error in quadrature to all values
of star counts would lead to negligible error inflation in regions with
higher counts.


Pulsar Timings
""""""""""""""

Millisecond pulsars have been discovered, in small numbers, in dozens of
MW globular clusters. Through extremely precise pulse measurements, the period
and the time-derivative of the period is known for a number of these pulsars.

These timing solutions, for pulsars embedded in clusters, follow a specific
relation:

.. math::
    \left(\frac{\dot{P}}{P}\right)_{\rm{obs}}
        = \left(\frac{\dot{P}}{P}\right)_{\rm{int}} + \frac{a_{\rm{clust}}}{c}
        + \frac{a_{\rm{gal}}}{c} + \frac{\mu^2 D}{c}

where the intrinsic spin-down of pulsars
:math:`\left(\frac{\dot{P}}{P}\right)_{\rm{int}}`, the potential (acceleration)
fields of the host cluster and galaxy, and the Shklovskii (proper motion) all
combine in the observed spin-down of the pulsar timing solution. 

The intrinsic spin-down of the observed pulsars is assumed to be identical to
pulsars found in the galaxy, outside of clusters, and dependant only on their
period. The field pulsars, as they are unaffected by the cluster potential,
can have their intrinsic timing solutions determined directly. A gaussian
kernel density estimator is then computed in the field :math:`P`-:math:`\dot{P}`
space, which is slice along each cluster pulsar's period to extract a
distribution of possible intrinsic values.

The cluster acceleration component, dependant on the model, is complicated by
the fact that the 3D position of the pulsar cannot be easily determined, and
the line-of-sight position of the pulsar within the cluster potential well is
unknown. Instead, a probability distribution of the acceleration can be
computed using the relation:

.. math::
    P(a_{\hat{z}}|z) = \frac{dm}{da(z)} = \frac{dm}{dz} \frac{dz}{da(z)}
                     = \frac{\rho(z)}{\left| \frac{da(z)}{dz} \right|}

These two distributions are then convolved, alongside a gaussian error
distribution representing the measurement uncertainties. Shifting by the
galactic and proper motion components (which are small and constant), a final
normalized probability distribution is obtained.

The measured timing solution is then interpolated onto this distribution,
computing a final likelihood value, for this pulsar. All pulsars in the cluster
have their likelihoods summed in the usual manner.

.. TODO DM stuff (this is maybe a bit out of date)


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


Sampling
========

The posterior distribution of the parameter set :math:`\Theta` must be
determined through a statistical sampling technique. Two such set of
algorithms are available in ``GCfit``.

.. TODO might want to expand on these?

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
