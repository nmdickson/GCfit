Models
======

The ``GCfit`` Models can be accessed through the core module at

.. code-block:: python

    >>> import gcfit

All models are based off of the single base class :class:`gcfit.core.Model`.

To begin, we can start by exploring a model with some arbitrary default
parameters. The :class:`gcfit.core.Model` class gives default values for many
arguments, which you may want to adjust yourself. See the documentation of said
class for more explanation of the meaning of all available parameters.

.. code-block:: python

    >>> model = gcfit.Model(W0=6.3, M=5e5, rh=6.7, age=12, FeH=-0.7)
    >>> model
    <gcfit.core.data.Model object at 0x7f43609cb9a0

The model will automatically generate a number of mass bins, containing either
stars or remnants of a certain type, which are used to solve the multimass
version of the LIMEPY DF.

.. code-block:: python

    >>> # Mean masses per bin
    >>> model.mj
    <Quantity [ 0.11766481,  0.16234563,  0.22399308, ...,  7.11333146,
                8.94500201, 11.60259068] solMass>

    >>> # Total mass per bin
    >>> model.Mj
    <Quantity [2.97895583e+04, 3.73135159e+04, 4.67358943e+04, ...,
               3.16648907e+01, 6.09462856e+01, 4.72391331e+01] solMass>

    >>> # Stellar object types (MS, NS, WD, BH)
    >>> model.star_types
    array(['MS', 'MS', 'MS', 'MS', 'MS', 'MS', 'MS', 'MS', 'MS', 'WD', 'WD',
           'WD', 'WD', 'WD', 'WD', 'NS', 'BH', 'BH', 'BH', 'BH'], dtype='<U2')

    >>> # Total mass and number of black holes, in their repective bins
    >>> model.BH.Mj
    <Quantity [12.09032806, 31.66489069, 60.94628557, 47.23913309] solMass>
    >>> model.BH.Nj
    <Quantity [2.04506045, 4.45148534, 6.81344571, 4.07142977]>

Notice that the majority of interesting quantities in :class:`gcfit.core.Model`
are stored as :class:`astropy.Quantity` objects, with their respective units.

The radial profiles of a number of system properties, such as velocity
dispersion, density and energy, are available for each mass bin, as well as a
number of useful radii.

.. code-block:: python

    >>> # Radial profile domain
    >>> model.r
    <Quantity [0.00000000e+00, 1.87338537e-07, 2.06072391e-06, ...,
               6.01602811e+01, 6.01602836e+01, 6.01602872e+01] pc>

    >>> # Density profile of the most massive main-sequence stars
    >>> model.rhoj[model.nms - 1]
    <Quantity [1.22558007e+05, 1.22558004e+05, 1.22557541e+05, ...,
               2.28136910e-24, 3.06566189e-25, 0.00000000e+00] solMass / pc3>

    >>> # Half-mass radius of each mass bin
    >>> model.rhj
    <Quantity [9.17577916, 8.97214788, 8.67214036, ..., 0.03528696, 0.02549436,
               0.01402568] pc>

See :class:`gcfit.core.Model` for further description of all available properties.

Models matching a number of historical DF formulations can also be created
easily using the relevant generator functions. These functions mostly
consist of setting a specific default value for the truncation parameter ``g``.

.. code-block:: python

    >>> # Generate a King (1966) model
    >>> king = gcfit.Model.king(6.3, 5e5, 6.7, age=12, FeH=-0.7)

    >>> model.g, king.g
    (1.5, 1)


Sampled Models
^^^^^^^^^^^^^^

These (multimass) models can also be sampled, in order to return a random
distribution of stars matching the phase-space distribution of the models.

.. code-block:: python

    >>> sampled = model.sample()
    >>> sampled
    <gcfit.core.data.SampledModel object at 0x7f4360a3fa30>

    >>> # Total number of stars in the system
    >>> sampled.Nstars
    <Quantity 1244835>

    >>> # Cartesian coordinates of all stars, centred on the cluster centre
    >>> sampled.pos.x
    <Quantity [ 2.22175514e+01,  9.32170721e-01, -2.69610089e-01, ...,
                2.50494632e-03, -3.79114439e-03, -1.48577896e-02] pc>
    >>> sampled.pos.z
    <Quantity [-1.44490672e+01, -2.08278389e+01, -2.43389757e+00, ...,
                1.17980765e-02,  5.97614136e-03, -1.18399874e-04] pc>
    >>> sampled.pos._fields
    ('x', 'y', 'z', 'r', 'theta', 'phi')

    >>> # Radial and tangential velocities of each star
    >>> sampled.vel.r
    <Quantity [ 0.20798123,  6.36224581, -9.7605177 , ...,  1.81204356,
                1.86281209,  2.67130275] km / s>
    >>> sampled.vel.t
    <Quantity [ 7.30988878,  6.18068915, 12.31156676, ...,  3.81179933,
                1.77385682,  4.47794275] km / s>
    >>> sampled.vel._fields
    ('x', 'y', 'z', 'r', 't', 'theta', 'phi')

If a centre coordinate on the sky is given (as an :class:`astropy.SkyCoord`
with both position and velocity),
the projected positions and velocities on the sky can also be computed.

.. code-block:: python

    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord

    >>> deg, masyr, kms = u.deg, u.Unit('mas/yr'), u.Unit('km/s')
    >>> cen = SkyCoord(l=45. * deg, b=55. * deg,
    >>>                pm_l_cosb=5 * masyr, pm_b=3 * masyr, radial_velocity=2 * kms,
    >>>                frame='galactic')

    >>> p_sampled = model.sample(centre=cen)

    >>> p_sampled.galactic.lon
    <Quantity [45.05209111, 45.07594013, 44.9869397 , ..., 44.99989859,
               45.00023154, 45.00003136] deg>
    >>> p_sampled.galactic.pm_b
    <Quantity [3.09281663, 2.9219263 , 3.46475291, ..., 2.93432464, 3.00739841,
               3.10851683] mas / yr>
    >>> p_sampled.galactic._fields
    ('lat', 'lon', 'distance', 'pm_l_cosb', 'pm_b', 'v_los')


Observations
^^^^^^^^^^^^

Another useful class within ``GCfit`` is the :class:`gcfit.core.Observations` class,
which acts as a container for a number of observational datasets. These
observations are key for all fitting (see below), but are also useful when
working with individual models, as they contain a number of useful metadata
fields about the cluster:

.. code-block:: python

    >>> obs = gcfit.Observations('NGC104')
    Observations(cluster="NGC0104")

    >>> model = gcfit.Model(W0=6.3, M=5e5, rh=6.7, observations=obs)
    >>> model.age, model.FeH
    (<Quantity 11.75 Gyr>, -0.72)

More information on the datafiles underlying this class, and how to create your
own datafiles can be found at (TODO).
