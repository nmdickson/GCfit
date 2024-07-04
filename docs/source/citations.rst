=========
Citations
=========

``GCfit`` was first introduced in
`Dickson et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5320D>`_
and updated in
`Dickson et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.529..331D>`_.

If you find this package useful in your research, please consider citing the
relevant papers below:

Observational Data
==================

Each observational dataset should come with it's own ``source`` metadata,
typically in the form of a
`bibcode <https://adsabs.harvard.edu/help/actions/bibcode>`_ identifier.

While these can be accessed directly through the metadata attribute of each
``Dataset``:

.. code-block:: python
    
    >>> dset = obs['number_density']
    >>> dset.mdata['source']
    '2019MNRAS.485.4906D'

``GCfit`` also comes equipped with some utility functions to automatically
convert bibcodes to useful formats, like bibtex. This functionality requires
the `ads <https://github.com/andycasey/ads>`_ package to be installed correctly,
with a valid ``ADS_DEV_KEY`` set.

The ``Observations`` and ``Dataset`` objects can provide some of the available
formats directly:

.. code-block:: python

    >>> sources = obs.get_sources()
    >>> print(sources['number_density'][0])
    @ARTICLE{2019MNRAS.485.4906D,
           author = {{de Boer}, T.~J.~L. and {Gieles}, M. and {Balbinot}, E. and {H{\'e}nault-Brunet}, V. and {Sollima}, A. and {Watkins}, L.~L. and {Claydon}, I.},
            title = "{Globular cluster number density profiles using Gaia DR2}",
    ...
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    >>> print(dset.cite())
    de Boer et al. (2019)

Or the utility methods can be used directly:

.. code-block:: python

    >>> print(gcfit.util.bibcode2bibtex(dset.mdata['source'])[0])
    @ARTICLE{2019MNRAS.485.4906D,
           author = {{de Boer}, T.~J.~L. and {Gieles}, M. and {Balbinot}, E. and {H{\'e}nault-Brunet}, V. and {Sollima}, A. and {Watkins}, L.~L. and {Claydon}, I.},
            title = "{Globular cluster number density profiles using Gaia DR2}",
    ...
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    >>> print(gcfit.util.bibcode2cite(dset.mdata['source']))
    de Boer et al. (2019)


Models
======

The equilibrium models used should be cited from the ``limepy`` paper:
`2015MNRAS.454..576G <https://adsabs.harvard.edu/abs/2015MNRAS.454..576G>`_.

The mass evolution algorithm (`ssptools <https://github.com/SMU-clusters/ssptools>`_)
is based off of the algorithm first introduced in
`2018MNRAS.474.2479B <https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.2479B>`_
(and updated in
`Dickson et al., 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5320D>`_).


Samplers
========

If you are using the MCMC fitter (``MCMC_fit``), the sampler source software
should be cited as the ``emcee`` paper:
`2013PASP..125..306F <https://adsabs.harvard.edu/abs/2013PASP..125..306F>`_.
Specific proposal algorithm citations can be found within.

Nested sampling fits (``nested_fit``) should cite the ``dynesty`` paper:
`2020MNRAS.493.3132S <https://adsabs.harvard.edu/abs/2020MNRAS.493.3132S>`_.
For specific bound and sampler algorithm sources, see the
`dynesty documentation <https://dynesty.readthedocs.io/en/latest/references.html>`_.

Other
=====

If Bayesian hyperparameters are used (``hyperparams=True`` in any fitting),
the source paper
`2002MNRAS.335..377H <https://adsabs.harvard.edu/abs/2002MNRAS.335..377H>`_
can be cited.

``GCfit`` makes extensive use of the
`numpy <https://adsabs.harvard.edu/abs/2020Natur.585..357H>`_,
`scipy <https://adsabs.harvard.edu/abs/2020NatMe..17..261V>`_ and
`astropy <https://adsabs.harvard.edu/abs/2018AJ....156..123A>`_
libraries. All plotting functionality is enabled by the
`matplotlib <https://adsabs.harvard.edu/abs/2007CSE.....9...90H>`_ library.
Parallelization pools are handled by the
`schwimmbad <https://adsabs.harvard.edu/abs/2017JOSS....2..357P>`_ library