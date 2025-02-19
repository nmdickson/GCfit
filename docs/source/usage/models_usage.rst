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
    <Quantity [ 0.11766481,  0.16234563,  0.22399308, ...,  7.10114089,
                8.94490091, 11.4071349] solMass>

    >>> # Total mass per bin
    >>> model.Mj
    <Quantity [2.97895583e+04, 3.73135159e+04, 4.67358943e+04, ...,
               2.98231834e+01, 5.58230429e+01, 5.05028550e+01] solMass>

    >>> # Stellar object types (MS, NS, WD, BH)
    >>> model.star_types
    array(['MS', 'MS', 'MS', 'MS', 'MS', 'MS', 'MS', 'MS', 'MS', 'WD', 'WD',
           'WD', 'WD', 'WD', 'WD', 'NS', 'BH', 'BH', 'BH', 'BH'], dtype='<U2')

    >>> # Total mass and number of black holes, in their repective bins
    >>> model.BH.Mj
    <Quantity [12.26536265, 29.82318344, 55.82304292, 50.50285502] solMass>
    >>> model.BH.Nj
    <Quantity [2.07936342, 4.19977352, 6.24076705, 4.42730409]>

Notice that the majority of interesting quantities in :class:`gcfit.core.Model`
are stored as :class:`astropy.Quantity` objects, with their respective units.

The radial profiles of a number of system properties, such as velocity
dispersion, density and energy, are available for each mass bin, as well as a
number of useful radii.

.. code-block:: python

    >>> # Radial profile domain
    >>> model.r
    <Quantity [0.00000000e+00, 2.08887713e-07, 2.29776485e-06, ...,
               6.00907685e+01, 6.00908280e+01, 6.00908306e+01] pc>

    >>> # Density profile of the most massive main-sequence stars
    >>> model.rhoj[model.nms - 1]
    <Quantity [1.22615737e+05, 1.22615733e+05, 1.22615270e+05, ...,
               4.10890596e-21, 1.21353818e-25, 0.00000000e+0] solMass / pc3>

    >>> # Half-mass radius of each mass bin
    >>> model.rhj
    <Quantity [9.16723465, 8.96575389, 8.66929104, ..., 0.0351056, 0.0252477,
               0.01440394] pc>

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
    <Quantity 1244880>

    >>> # Cartesian coordinates of all stars, centred on the cluster centre
    >>> sampled.pos.x
    <Quantity [ 1.70500994e+01,  2.86122226e+00, -1.16756636e+00, ...,
                5.24831562e-03, -1.45917202e-03, -9.13248353e-03] pc>
    >>> sampled.pos.z
    <Quantity [-9.56794358e+00, -2.53661084e+00, -6.16515774e+00, ...,
                6.71300176e-03,  2.06363703e-03,  1.76499660e-03] pc>
    >>> sampled.pos._fields
    ('x', 'y', 'z', 'r', 'theta', 'phi')

    >>> # Radial and tangential velocities of each star
    >>> sampled.vel.r
    <Quantity [-0.71188986, -6.86505168,  3.51273965, ..., -3.84003514,
                3.12497992,  2.09300044] km / s>
    >>> sampled.vel.t
    <Quantity [6.83229216, 4.14966349, 4.9182554 , ..., 3.80989784, 4.72867768,
               1.41100603] km / s>
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
    <Quantity [44.99073051, 45.27478485, 44.95426386, ..., 45.0009292 ,
               44.99996491, 45.00047616] deg>
    >>> p_sampled.galactic.pm_b
    <Quantity [2.59835011, 3.18921132, 3.36723663, ..., 3.06262506, 3.0400891 ,
               2.96432599] mas / yr>
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
