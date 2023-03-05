from .. import util

import h5py
import numpy as np
import limepy as lp
from astropy import units as u
from ssptools import EvolvedMF

import fnmatch
import logging
import itertools
from collections import namedtuple


__all__ = ['DEFAULT_THETA', 'Model', 'FittableModel', 'SingleMassModel',
           'Observations']


# The order of this is important!
DEFAULT_THETA = {
    'W0': 6.0,
    'M': 0.69,
    'rh': 2.88,
    'ra': 1.23,
    'g': 0.75,
    'delta': 0.45,
    's2': 0.1,
    'F': 1.1,
    'a1': 0.5,
    'a2': 1.3,
    'a3': 2.5,
    'BHret': 0.5,
    'd': 6.405,
}


# --------------------------------------------------------------------------
# Cluster Observational Data
# --------------------------------------------------------------------------
# TODO maybe define a new excepton for when a req'd thing is not in an obs
# TODO add proposal ids to mass function data, bibcodes don't really match


class Variable(u.Quantity):
    '''Read-only `astropy.Quantity` subclass with metadata support'''

    def __repr__(self):
        prefix = f'<{self.__class__.__name__} '

        if self.is_empty:
            view = 'Empty'
        else:
            view = np.array2string(
                self.view(np.ndarray), separator=', ', prefix=prefix
            )

        return f'{prefix}{view}{self._unitstr:s}>'

    # TODO better way to handle string arrays, and with nicer method failures
    # TODO the "readonly" part of Variable is currently not functional
    def __new__(cls, value, unit=None, mdata=None, *args, **kwargs):

        if is_empty := (value.shape is None):
            value = []

        value = np.asanyarray(value)

        is_str = value.dtype.kind in 'US'

        # If unit is None, look for unit in mdata then assume dimensionless
        if unit is None and mdata is not None:
            try:
                unit = mdata['unit']
            except KeyError:
                pass

        if unit is not None:

            if is_str:
                raise ValueError("value is array of strings, cannot have unit")

            unit = u.Unit(unit)

        # If value is already a quantity, ensure its compatible with given unit
        if isinstance(value, u.Quantity):
            if unit is not None and unit is not value.unit:
                value = value.to(unit)

            unit = value.unit

        # Create the parent object (usually quantity, except if is string)
        if not is_str:
            quant = super().__new__(cls, value, unit, *args, **kwargs)
        else:
            quant = np.asarray(value, *args, **kwargs).view(cls)

        # Store the metadata
        if isinstance(mdata, dict):
            quant.mdata = mdata
        elif mdata is None:
            quant.mdata = dict()
        else:
            raise TypeError('`mdata` must be a dict or None')

        # flag if empty
        quant.is_empty = is_empty

        # quant.flags.writeable = False

        return quant

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.mdata = getattr(obj, 'mdata', dict(defaulted=True))

        self.is_empty = getattr(obj, 'is_empty', False)

        try:

            if self._unit is None:
                unit = getattr(obj, '_unit', None)
                if unit is not None:
                    self._set_unit(unit)

            if 'info' in obj.__dict__:
                self.info = obj.info

        except AttributeError:
            pass

        # nump.arra.view is only one now missing its writeable=False
        # if obj.flags.writeable is not False:
        #    self.flags.writeable = False

    def __quantity_subclass__(self, unit):
        return type(self), True


class Dataset:
    '''Read-only container for all variables associated with a single dataset

    Contains all data representing a single observational dataset,
    i.e. all `Variable`s associated to a single physical process, from a single
    source, along with all relevant metadata.

    Should not be initialized directly, but from an `Observations` instance,
    using the base data file's relevant group.

    Attributes
    ----------
    variables

    mdata : dict
        Dictionary of all "cluster-level" metadata

    '''

    def __repr__(self):
        return f'Dataset("{self._name}")'

    def __str__(self):
        return f'{self._name} Dataset'

    _citation = None

    def __citation__(self):
        if self._citation is not None:
            return self._citation
        else:
            try:
                bibcodes = self.mdata['source'].split(';')

                try:
                    self._citation = util.bibcode2cite(bibcodes)

                except RuntimeError:
                    # Failed to get citation, just return raw source
                    self._citation = '; '.join(bibcodes)

                return self._citation

            except KeyError:
                return None

    def __contains__(self, key):
        return key in self._dict_variables

    def __getitem__(self, key):
        try:
            return self._dict_variables[key]
        except KeyError:
            mssg = f"Variable '{key}' does not exist in '{self._name}'"
            raise KeyError(mssg)

    def __iter__(self):
        raise TypeError(f"'{self.__class__.__name__}' object is not iterable")

    def _init_variables(self, name, var):
        '''used by group.visit'''

        if isinstance(var, h5py.Dataset):
            mdata = dict(var.attrs)
            self._dict_variables[name] = Variable(var, mdata=mdata)

    def __init__(self, group):

        self._dict_variables = {}
        group.visititems(self._init_variables)

        self.mdata = dict(group.attrs)

        self._name = group.name

    @property
    def size(self):
        '''Number of datapoints in this dataset'''
        try:
            return list(self._dict_variables.values())[0].size
        except IndexError:
            mssg = f"No variables stored in {self}, can't retrieve size"
            logging.warning(mssg)
            return None

    @property
    def variables(self):
        '''Dictionary of all `Variables`s contained in this class'''
        return self._dict_variables

    def cite(self):
        '''Return the literature source (citation) of this `Dataset`'''
        return self.__citation__()

    def build_err(self, varname, model_r, model_val):
        '''Return the most relevant uncertainties associated with a variable

        Determines and returns the uncertainty (error) variables corresponding
        to the `varname` variable, which must also exist within this dataset.

        As some uncertainties are not symmetric (i.e. not equal in the positive
        and negative directions), which side of the error bars to utilize must
        be determined.
        To accomplish this, the given `model_r` values are interpolated onto
        this dataset's radial `r` profile, and for each point the closest
        error bar to each `model_val` is chosen.

        parameters
        ----------
        varname : str
            Name of the variable to retrieve the errors for

        model_r : astropy.Quantity
            Quantity representing the desired radial profile to interpolate on.
            Only used for assymetric errors. Must have equivalent units to
            the dataset `r`.

        model_val : astropy.Quantity
            Quantity representing the desired values to interpolate on.
            Only used for assymetric errors. Must have equivalent units to
            the given `varname`.
        '''

        quantity = self[varname]

        # ------------------------------------------------------------------
        # Attempt to convert model values
        # ------------------------------------------------------------------
        # TODO maybe interpolating to model should be optional, not default
        #   and just return Δvar or all upper or lower by default

        model_r = model_r.to(self['r'].unit)
        model_val = model_val.to(quantity.unit)

        # ------------------------------------------------------------------
        # If a single homogenous uncertainty exists, return it
        # ------------------------------------------------------------------

        try:
            return self[f'Δ{varname}']

        # ------------------------------------------------------------------
        # If the uncertainties aren't symmetric, determine which bound to use
        # based on the models value above or below the quantity
        # ------------------------------------------------------------------

        except KeyError:

            try:
                err_up = self[f'Δ{varname},up']
                err_down = self[f'Δ{varname},down']

            except KeyError:
                mssg = f"No existing err (Δ) Variable associated with {varname}"
                raise ValueError(mssg)

            err = np.zeros_like(quantity)

            model_val = np.interp(self['r'], model_r, model_val)

            gt_mask = (model_val > quantity)
            err[gt_mask] = err_up[gt_mask]
            err[~gt_mask] = err_down[~gt_mask]

            return err


class Observations:
    '''Read-only interface for all observational cluster data

    The main interface for reading and interacting with (reading) all
    observational data for the specified globular cluster.

    Defined based on a given cluster datafile, handles creation and access to
    all contained `Dataset`s, as well as the setup and arguments for all
    relevant likelihoods.

    The relevant cluster data files will be found using the
    `gcfit.util.get_cluster_path` function, and can likewise be retricted to
    "core" or "local" files. The data file used *must* be considered valid
    (i.e. pass all tests within `gcfit.utils.data.ClusterFile.test`).

    Parameters
    ----------
    cluster : str
        Name of the globular cluster. Will be used to retrieve the relevant
        cluster data file.

    standardize_name : bool, optional
        Whether to "standardize" the given `cluster` name to match the exact
        format of "core" cluster files or not. Searches against core clusters
        will always be standardized, no matter this parameter. Defaults to True.

    restrict_to : {None, 'local', 'core'}
        Where to search for the cluster data file, see
        `gcfit.util.get_cluster_path` for more information.

    Attributes
    ----------
    valid_likelihoods

    datasets

    mdata : dict
        Dictionary of all "cluster-level" metadata

    See Also
    --------
    gcfit.util.get_cluster_path : Locating of data file based on `cluster` name
    gcfit.util.data.ClusterFile : Handling of data file creation and editing
    '''

    _valid_likelihoods = None

    def __repr__(self):
        return f'Observations(cluster="{self.cluster}")'

    def __str__(self):
        return f'{self.cluster} Observations'

    def __iter__(self):
        raise TypeError(f"'{self.__class__.__name__}' object is not iterable")

    def __getitem__(self, key):

        try:
            # return a dataset
            return self._dict_datasets[key]
        except KeyError as err:
            try:
                # return a variable within a dataset
                group, name = key.rsplit('/', maxsplit=1)

                try:
                    dset = self._dict_datasets[group]
                except KeyError:
                    mssg = f"Dataset '{group}' does not exist in {self}"
                    raise KeyError(mssg) from err

                try:
                    return dset[name]
                except KeyError as var_err:
                    raise var_err from err

            except ValueError:
                # not in _dict_datasets and no '/' to split on so not a variable
                # TODO if key does exist but has subgroups; mention that in err
                mssg = f"Dataset '{key}' does not exist in {self}"
                raise KeyError(mssg) from err

    @property
    def datasets(self):
        '''Dictionary of all `Dataset`s contained in this class'''
        return self._dict_datasets

    @property
    def valid_likelihoods(self):
        '''List of likelihoods valid for the contained datasets. Returns a list
        of lists, with each list representing a different likelihood function,
        of the form of [dataset name, likelihood function, *function params].
        '''

        if self._valid_likelihoods is None:
            self._valid_likelihoods = self._determine_likelihoods()

        return self._valid_likelihoods

    def _find_groups(self, root_group, exclude_initials=True):
        '''lists pathnames to all groups under root_group, excluding initials'''

        def _walker(key, obj):
            if isinstance(obj, h5py.Group):

                if exclude_initials and key == 'initials':
                    return

                # relies on visititems moving top-down
                # this should theoretically remove all parent groups of groups
                try:
                    parent, name = key.rsplit('/', maxsplit=1)
                    groups.remove(parent)

                except ValueError:
                    pass

                groups.append(key)

        groups = []
        root_group.visititems(_walker)

        return groups

    def filter_datasets(self, pattern, valid_only=True):
        '''Return a subset of `Observations.datasets` based on given `pattern`

        Parameters
        ----------
        pattern : str
            A pattern string to filter all dataset names on, using `fnmatch`

        valid_only : bool, optional
            Whether to filter on all datasets or only those considered "valid"
            by `Observations.valid_likelihoods`
        '''

        if valid_only:
            datasets = {key for (key, *_) in self.valid_likelihoods}
        else:
            datasets = self.datasets.keys()

        return {key: self[key] for key in fnmatch.filter(datasets, pattern)}

    def filter_likelihoods(self, patterns, exclude=False, keys_only=False):
        '''Return subset of `Observations.valid_likelihoods` based on `patterns`

        Filters the results of `Observations.valid_likelihoods` based on a
        *list* of `patterns`. The pattern matching (for each pattern in the
        `patterns` list) is applied to both the dataset name and the likelihood
        function name (func.__name__).

        Parameters
        ----------
        patterns : list of str
            List of pattern strings to filter all likelihoods on using `fnmatch`

        exclude : bool, optional
            Whether to return all likelihoods which match the filters (False,
            default) or to exclude them, and return all others (True)

        keys_only : bool, optional
            Whether to return only the filtered dataset names (True) or the
            entire likelihood format as given by
            `Observations.valid_likelihoods` (False, default). Filtering will
            still be done on both dataset and likelihood names, no matter this
            parameter.
        '''

        matches, no_matches = [], []
        for component in self.valid_likelihoods:
            key, func, *_ = component
            func_name = func.__name__

            if keys_only:
                component = key

            if any(fnmatch.fnmatch(key, p) or fnmatch.fnmatch(func_name, p)
                   for p in patterns):
                matches.append(component)
            else:
                no_matches.append(component)

        return matches if not exclude else no_matches

    def get_sources(self, fmt='bibtex'):
        '''Return a dict of formatted citations for each contained dataset'''
        # TODO make this use dataset __citation__'s so it doesnt pull each time

        res = {}

        for key, *_ in self.valid_likelihoods:

            try:
                bibcode = self[key].mdata['source'].split(';')
            except KeyError:
                res[key] = None
                continue

            if fmt == 'bibcode':
                res[key] = bibcode

            elif fmt == 'bibtex' or fmt is None:
                res[key] = util.bibcode2bibtex(bibcode)

            elif fmt == 'citep':
                # TODO allow some formats which we parse the bibtex into
                raise NotImplementedError

        return res

    def __init__(self, cluster, *, standardize_name=True, restrict_to=None):

        self.mdata = {}
        self._dict_datasets = {}
        self.initials = DEFAULT_THETA.copy()

        filename = util.get_cluster_path(cluster, standardize_name, restrict_to)

        self.cluster = filename.stem

        with h5py.File(filename, 'r') as file:

            logging.info(f"Observations read from {filename}")

            for group in self._find_groups(file):
                self._dict_datasets[group] = Dataset(file[group])

            try:
                # This updates defaults with data while keeping default sort
                self.initials = {**self.initials, **file['initials'].attrs}

                if extras := (self.initials.keys() - DEFAULT_THETA.keys()):
                    mssg = (f"Stored initials do not match expected."
                            f"Extra values found: {extras}")
                    raise ValueError(mssg)

            except KeyError:
                logging.info("No initial state stored, using defaults")
                pass

            # TODO need a way to read units for some mdata from file
            self.mdata = dict(file.attrs)

    def _determine_likelihoods(self):
        from .. import probabilities

        comps = []
        for key in self.datasets:

            # --------------------------------------------------------------
            # Parse each key to determine if it matches with one of our
            # likelihood functions.
            # fnmatch is used to properly handle relevant subgroups
            # such as proper_motion/high_mass and etc, where they exist
            #
            # Some datasets could have multiple probabilities, depending on what
            # variables they contain
            #
            # Each component is a tuple where the first two elements are,
            # respectively, the observation key and likelihood function, and all
            # remaining elements are the extra arguments to pass to the function
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # Pulsar probabilities
            # --------------------------------------------------------------

            if fnmatch.fnmatch(key, '*pulsar*'):

                metadata = (
                    self.mdata['μ'],
                    (self.mdata['b'], self.mdata['l']),
                    'DM' in self[key]
                )

                if 'Pdot' in self[key]:

                    func = probabilities.likelihood_pulsar_spin

                    kde = probabilities.pulsars.field_Pdot_KDE()

                    comps.append((key, func, kde, *metadata))

                if 'Pbdot' in self[key]:

                    func = probabilities.likelihood_pulsar_orbital

                    comps.append((key, func, *metadata))

            # --------------------------------------------------------------
            # Line-of-sight velocity dispersion probabilities
            # --------------------------------------------------------------

            elif fnmatch.fnmatch(key, '*velocity_dispersion*'):

                func = probabilities.likelihood_LOS

                comps.append((key, func, ))

            # --------------------------------------------------------------
            # Number density probabilities
            # --------------------------------------------------------------

            elif fnmatch.fnmatch(key, '*number_density*'):

                func = probabilities.likelihood_number_density

                comps.append((key, func, ))

            # --------------------------------------------------------------
            # Proper motion dispersion probabilities
            # --------------------------------------------------------------

            elif fnmatch.fnmatch(key, '*proper_motion*'):

                if 'PM_tot' in self[key]:

                    func = probabilities.likelihood_pm_tot

                    comps.append((key, func, ))

                if 'PM_ratio' in self[key]:

                    func = probabilities.likelihood_pm_ratio

                    comps.append((key, func, ))

                if 'PM_R' in self[key]:

                    func = probabilities.likelihood_pm_R

                    comps.append((key, func, ))

                if 'PM_T' in self[key]:

                    func = probabilities.likelihood_pm_T

                    comps.append((key, func, ))

            # --------------------------------------------------------------
            # Stellar mass function probabilities
            # --------------------------------------------------------------

            elif fnmatch.fnmatch(key, '*mass_function*'):

                func = probabilities.likelihood_mass_func

                # Field
                cen = (self.mdata['RA'], self.mdata['DEC'])

                field = probabilities.mass.Field.from_dataset(self[key], cen)

                comps.append((key, func, field))

        return comps


# --------------------------------------------------------------------------
# Cluster modelled data
# --------------------------------------------------------------------------

# TODO The units are *quite* incomplete in Model (10)
# TODO what attributes should be documented?

# --------------------------------------------------------------------------
# Base model
# --------------------------------------------------------------------------


class Model(lp.limepy):
    r'''Wrapper class around a LIMEPY model, including mass function evolution

    Multimass globular cluster model implemented as a subclass around a
    `limepy.limepy` model, as defined by the input parameters,
    while including support for initial mass function evolution (through
    `ssptools`), units (through `astropy.Quantity`), tracer masses and enhanced
    metadata and mass results.

    The cluster mass function is evolved from it's initial mass function using
    fixed integration settings alongside the cluster's age, metallicity, escape
    velocity and mass-loss rates and fractions, either as given during
    initilization or as defined in the metadata of a given `Observations`.
    The resulting mass bins are arranged correctly, and have any possibly
    required (by the `Observations`) tracer masses added, before being used
    to solve the Limepy distribution function.

    The first 13 arguments are the main model parameters, which are used by
    the `FittableModel` as well. Most other arguments provide more fine control
    over the mass evolution algorithm and DF ODE solver.

    Note carefully the units required in some of the parameters here, which may
    not be the same as in other classes (such as `FittableModel`). `Quantity`
    objects, from astropy, can also be given in most cases to ensure the units
    are handled correctly.

    Parameters
    ----------

    W0 : float or astropy.Quantity
        The (dimensionless) central potential. Used as a boundary condition for
        solving Poisson’s equation and defines how centrally concentrated the
        model is.

    M : float or astropy.Quantity
        The total mass of the system, in all mass components, in Msun.

    rh : float or astropy.Quantity
        The system half-mass radius, in parsecs.

    g : float, optional
        The truncation parameter, which controls the sharpness of the outer
        density truncation of the model. No finite models exist outside
        0 <= g < 3.5. Defaults to 1.5.

    delta : float, optional
        Sets the mass dependance of the velocity scale for each mass component.
        Increased value of delta (usually up to ~0.5) indicate an increased
        degree of mass segregation present in the system.
        Defaults to 0.45.

    ra : float or astropy.Quantity, optional
        The (dimensionless) anisotropy-radius, which determines the amount of
        anisotropy in the system, with higher ra values indicating more
        isotropy. This quantity is scaled based on the given `rh` in physical
        units.

    a1 : float, optional
        The low-mass IMF exponent (representing masses between `m_breaks[0:2]`).
        Defaults to 1.3, matching Kroupa (2001).

    a2 : float, optional
        The intermediate-mass IMF exponent (representing masses between
        `m_breaks[1:3]`). Defaults to 2.3, matching Kroupa (2001).

    a3 : float, optional
        The high-mass IMF exponent (representing masses between
        `m_breaks[2:4]`). Defaults to 2.3, matching Kroupa (2001).

    BHret : float, optional
        The black hole retention fraction, representing the percentage (between
        0 and 100) of black holes retained after dynamical ejections and natal
        kicks.

    d : float or astropy.Quantity, optional
        Distance to the cluster, from Earth, in kiloparsecs. Mainly used for any
        conversions between observational (angular) and model (linear) units,
        and thus mostly only required for comparing with observations.
        Defaults to an arbitrary distance of 5 kpc.

    s2 : float, optional
        Nuisance parameter applied as an additional unknown uncertainty to all
        number density profiles, allowing for small deviations between the
        outer parts of the model and observations.
        Only used for comparing with observations, not required otherwise
        Defaults to 0.

    F : float, optional
        Nuisance parameter applied as an additional scaling (F >= 1) factor on
        the uncertainty in all mass function profiles, encapsulating possible
        additional sources of error.
        Only used for comparing with observations, not required otherwise.
        Defaults to 1.

    observations : Observations, optional
        The `Observations` instance corresponding to this cluster. While not
        necessary for solving a model which is not meant for comparison to the
        data, a number of optional parameters can be read from the metadata of
        a given `Observations` instance, such as age and metallicity, and thus
        would not need to be provided here.

    age : float or astropy.Quantity, optional
        The current age of the system, in Gyrs. This age is used by the
        mass evolution algorithm to determine which of the initial stars, as
        defined by the IMF, will have evolved into remnants by the present day.
        If no `observations` are given, this quantity *must* be supplied,
        otherwise an attempt will be made to read the age from
        the `observations`.

    FeH : float, optional
        The cluster metallicity, in solar fraction [Fe/H]. This is used by
        the mass evolution algorithm to determine the final masses of the
        remnants formed over the evolution of the cluster.
        If no `observations` are given, this quantity *must* be supplied,
        otherwise an attempt will be made to read the metallicity from
        the `observations`.

    m_breaks : (4,) numpy.ndarray or astropy.Quantity, optional
        The IMF break-masses (including outer bounds) in Msun, defining the
        mass ranges of each IMF exponent. Defaults to [0.1, 0.5, 1.0, 100].

    nbins : (3,) numpy.ndarray of int, optional
        Number of mass bins in each regime of the IMF, as defined by `m_breaks`.
        This number of bins will be log-spaced between each of the break masses.
        Defaults to [5, 5, 20].

    NS_ret : float, optional
        Neutron star retention fraction (0 to 1). Defaults to 0.1 (10%).

    BH_ret_int : float, optional
        Initial black hole retention fraction (0 to 1). Defaults to 1 (100%).

    natal_kicks : bool, optional
        Whether to account for natal kicks in the BH dynamical retention.
        Defaults to True.

    Ndot : float, optional
        Represents rate of change of the number of stars N over time, in stars
        per Myr. Regulates low-mass object depletion (ejection) due to dynamical
        evolution. Do not use unless you know what you're doing.
        Defaults to 0.

    vesc : float or astropy.Quantity, optional
        Initial cluster escape velocity, in km/s, for use in the computation of
        the effects of BH natal kick. Defaults to 90 km/s.

    meanmassdef : {'global', 'central'}, optional
        Definition of the mean mass :math:`\bar{m}` used to define the
        dimensionless mass of each component (:math:`\mu_j = m_j/\bar{m}`).
        See Eqn. 26 of Gieles & Zocchi (2015) for more details.
        Defaults to 'global', i.e. the unweighted mean mass of all stars over
        the entire system.

    ode_maxstep : float, optional
        Maximum step size for the `limepy` ODE integrator. Defaults to 1e10.

    ode_rtol : float, optional
        Relative tolerance parameter for the `limepy` ODE integrator.
        Defaults to 1e-7.

    Attributes
    ----------

    mj : astropy.Quantity
        Array containing the individual mass in all mass bins, including
        both stars and remnants, at the given age.

    Mj : astropy.Quantity
        Array containing the total mass in all mass bins, including
        both stars and remnants, at the given age.

    Nj : astropy.Quantity
        Array containing the total number in all mass bins, including
        both stars and remnants, at the given age.

    mbin_widths : astropy.Quantity
        The width of each mass bin, in Msun.

    nms : int
        The number of mass bins containing main sequence stars.
        The first `nms` mass bins will be the star bins (i.e. mj[:nms-1])

    nmr : int
        The number of mass bins containing remnants.
        The proceeding `nmr` mass bins will be the remnant bins
        (i.e. mj[nms-1:nms + nmr])

    star_types : numpy.ndarray
        Array of 2-character strings representing the type of object in each
        mass bin (MS, WD, NS, BH).
        Note that this is not 100% accurate, as some remnants of different
        types may have final masses which fall into the same mass bin.

    {BH,NS,WD}_mj : astropy.Quantity
        Black hole, neutron star or white dwarf component mass bin size

    {BH,NS,WD}_Mj : astropy.Quantity
        Black hole, neutron star or white dwarf total bin mass

    {BH,NS,WD}_Nj : astropy.Quantity
        Black hole, neutron star or white dwarf total bin number (Mj / mj)

    r : astropy.Quantity
        The projected radial distances, in pc, from the centre of the cluster,
        defining the domain used in all other model profiles.

    phi, phij : astropy.Quantity
        System (total, per mass bin) potential as a function of distance from
        the centre of the cluster.

    rho, rhoj : astropy.Quantity
        System (total, per mass bin) density as a function of distance from
        the centre of the cluster.

    v2, v2j : astropy.Quantity
        System (total, per mass bin) mean-square velocity as a function of
        distance from the centre of the cluster.

    v2r, v2rj, v2t, v2tj : astropy.Quantity
        Radial and Tangential components of the (total, per mass bin)
        mean-square velocity, in the plane of the sky, as a function of
        distance from the centre of the cluster.

    r0, rh, rv, rt, ra, rhp : astropy.Quantity
        The total (King, half-mass, virial, truncation, anisotropy, projected
        half-mass) radius of the cluster.

    r0j, rhj, raj, rhpj : astropy.Quantity
        The per mass bin (King, half-mass, anisotropy, projected half-mass)
        radius of the cluster.

    Notes
    -----
    The IMF of the model is defined using a three-component broken power law,
    of the familiar form:

    .. math::
        \xi(m) \propto \begin{cases}
            m^{-a_1} & 0.1 M_{\odot} < m \leq 0.5 M_{\odot}, \\
            m^{-a_2} & 0.5 M_{\odot} < m \leq 1.0 M_{\odot}, \\
            m^{-a_3} & 1.0 M_{\odot} < m \leq 100 M_{\odot}, \\
        \end{cases}

    where the `a` exponents are given as input parameters.

    See Also
    --------
    limepy : Distribution-function model base of this class
    '''

    def _assign_units(self):
        '''Convert most values to `astropy.Quantity` with correct units'''

        # TODO this needs to be much more general
        #   Right now it is only applied to those params we use in likelihoods?
        #   Also the actualy units used are being set manually

        if not self.scale:
            return

        G_units = u.Unit('(pc km2) / (s2 Msun)')
        R_units = u.pc
        M_units = u.Msun
        V2_units = G_units * M_units / R_units

        self.G <<= G_units

        self.M <<= M_units
        self.mj <<= M_units
        self.Mj <<= M_units
        self.mc <<= M_units
        self.mcj <<= M_units
        self.mmean <<= M_units
        self.mbin_widths <<= M_units

        self.r <<= R_units
        self.r0 <<= R_units
        self.r0j <<= R_units
        self.rh <<= R_units
        self.rhj <<= R_units
        self.rhp <<= R_units
        self.rt <<= R_units
        self.ra <<= R_units
        self.raj <<= R_units
        self.rv <<= R_units
        self.rs <<= R_units

        # TODO this may be wrong (it's "phase-space" volume)
        self.volume <<= R_units**3

        self.v2T <<= V2_units
        self.v2Tj <<= V2_units
        self.v2R <<= V2_units
        self.v2Rj <<= V2_units
        self.v2p <<= V2_units
        self.v2pj <<= V2_units
        self.s2 <<= V2_units
        self.s2j <<= V2_units

        self.rho <<= (M_units / R_units**3)
        self.rhoj <<= (M_units / R_units**3)
        self.Sigma <<= (M_units / R_units**2)
        self.Sigmaj <<= (M_units / R_units**2)

    def __init__(self, W0, M, rh, g=1.5, delta=0.45, ra=1e8,
                 a1=1.3, a2=2.3, a3=2.3, BHret=1.0, d=5,
                 s2=0., F=1., *, observations=None, age=None, FeH=None,
                 m_breaks=[0.1, 0.5, 1.0, 100], nbins=[5, 5, 20],
                 tcc=0.0, NS_ret=0.1, BH_ret_int=1.0,
                 natal_kicks=True, Ndot=0.0, vesc=90.,
                 meanmassdef='global', ode_maxstep=1e10, ode_rtol=1e-7):

        # ------------------------------------------------------------------
        # Add/convert units of some quantities. Supports quantities as inputs
        # ------------------------------------------------------------------

        W0 <<= u.dimensionless_unscaled
        M <<= u.Msun
        rh <<= u.pc
        ra <<= u.dimensionless_unscaled
        d <<= u.kpc
        age <<= u.Gyr

        m_breaks <<= u.Msun
        vesc <<= (u.km / u.s)

        # ------------------------------------------------------------------
        # Pack theta
        # ------------------------------------------------------------------

        self.theta = dict(W0=W0, M=M.to_value('1e6 Msun'), rh=rh.value,
                          ra=np.log10(ra), g=g, delta=delta,
                          a1=a1, a2=a2, a3=a3, BHret=BHret,
                          s2=s2, F=F, d=d.value)

        self.d = d

        # ------------------------------------------------------------------
        # Unpack observations, cluster specific metadata
        # ------------------------------------------------------------------

        if observations is not None:
            if not isinstance(observations, Observations):
                observations = Observations(observations)

            # These are required, and will default to None (must be supplied)
            if age is None:
                age = observations.mdata['age'] << u.Gyr

            if FeH is None:
                FeH = observations.mdata['FeH']

            # These are maybe required, but have actual defaults from the start;
            # if you explicitly set None and its not in obs, then that's on you
            if Ndot is None:
                Ndot = observations.mdata['Ndot']

            if vesc is None:
                vesc = observations.mdata['vesc'] << u.km / u.s

        else:
            if age is None or FeH is None:
                # Error here if age, FeH can't be found
                # for Ndot, vesc, let them be, if necessary they'll fail later
                mssg = {"Must supply either `age` and `FeH` or "
                        "an `observations`, to read them from"}
                raise ValueError(mssg)

        self.observations = observations

        # ------------------------------------------------------------------
        # Get mass function
        # ------------------------------------------------------------------

        self._mf = EvolvedMF(
            m_breaks=m_breaks.value,
            a_slopes=[-a1, -a2, -a3],
            nbins=nbins,
            FeH=FeH,
            tout=np.array([age.to_value('Myr')]),
            Ndot=Ndot,
            N0=5e5,
            tcc=tcc,
            NS_ret=NS_ret,
            BH_ret_int=BH_ret_int,
            BH_ret_dyn=BHret / 100.,
            natal_kicks=natal_kicks,
            vesc=vesc.value
        )

        mj, Mj = self._mf.m, self._mf.M

        self.mbin_widths = self._mf.bin_widths

        self.nms = self._mf.nms
        self.nmr = self._mf.nmr

        self.star_types = self._mf.types

        # append tracer mass bins (must be appended to end to not affect nms)
        if observations is not None:

            # TODO should only append tracer masses for valid likelihood dsets?
            tracer_mj = np.unique([
                dataset.mdata['m'] for dataset in observations.datasets.values()
                if 'm' in dataset.mdata
            ])

            mj = np.concatenate((mj, tracer_mj))
            Mj = np.concatenate((Mj, 0.1 * np.ones_like(tracer_mj)))

            self.star_types = np.concatenate((self.star_types,
                                              ['TR'] * tracer_mj.size))

            self._tracer_bins = slice(self.nms + self.nmr, None)

        else:
            logging.warning("No `Observations` given, no tracer masses added")

        # ------------------------------------------------------------------
        # Create the limepy model base
        # ------------------------------------------------------------------

        self._limepy_kwargs = dict(
            phi0=W0,
            g=g,
            M=M.value,
            rh=rh.value,
            ra=ra,
            delta=delta,
            mj=mj,
            Mj=Mj,
            project=True,
            verbose=False,
            meanmassdef=meanmassdef,
            max_step=ode_maxstep,
            ode_rtol=ode_rtol
        )

        try:
            super().__init__(**self._limepy_kwargs)
        except ValueError as err:
            cause = err.args[0]

            if ("rmax reached in mf iteration" in cause
                    or "maximum number of iterations reached" in cause):

                mssg = ("Model solver failed to converge in time. "
                        "Model parameters must be adjusted")
                raise ValueError(mssg) from err

            else:
                raise err

        if not self.converged:
            mssg = "Model solver failed to converge to a finite extent"
            raise ValueError(mssg)

        # ------------------------------------------------------------------
        # Assign units to model values
        # ------------------------------------------------------------------

        self._assign_units()

        # TODO this is only equal to 10^θ['ra'] if < model.ramax (1e8)
        self.unscaled_ra = self.ra / self.rs

        # ------------------------------------------------------------------
        # Split apart the stellar classes of the mass bins
        # ------------------------------------------------------------------

        # Recompute to account for physical scaling applied by limepy
        self.Nj = self.Mj / self.mj

        # TODO these kind of slices would prob be more useful than nms elsewhere
        self._star_bins = slice(0, self.nms)
        self._remnant_bins = slice(self.nms, self.nms + self.nmr)

        # Masked only within `remnant bins` regime to avoid issues with tracers
        self._BH_bins = self._mf.types[self.nms:] == 'BH'
        self._NS_bins = self._mf.types[self.nms:] == 'NS'
        self._WD_bins = self._mf.types[self.nms:] == 'WD'

        # ------------------------------------------------------------------
        # Get Black Holes
        # ------------------------------------------------------------------

        self.BH_mj = self.mj[self._remnant_bins][self._BH_bins]
        self.BH_Mj = self.Mj[self._remnant_bins][self._BH_bins]
        self.BH_Nj = self.Nj[self._remnant_bins][self._BH_bins]

        self.BH_rhoj = self.rhoj[self._remnant_bins][self._BH_bins]
        self.BH_Sigmaj = self.Sigmaj[self._remnant_bins][self._BH_bins]

        # ------------------------------------------------------------------
        # Get White Dwarfs
        # ------------------------------------------------------------------

        self.WD_mj = self.mj[self._remnant_bins][self._WD_bins]
        self.WD_Mj = self.Mj[self._remnant_bins][self._WD_bins]
        self.WD_Nj = self.Nj[self._remnant_bins][self._WD_bins]

        self.WD_rhoj = self.rhoj[self._remnant_bins][self._WD_bins]
        self.WD_Sigmaj = self.Sigmaj[self._remnant_bins][self._WD_bins]

        # ------------------------------------------------------------------
        # Get Neutron Stars
        # ------------------------------------------------------------------

        self.NS_mj = self.mj[self._remnant_bins][self._NS_bins]
        self.NS_Mj = self.Mj[self._remnant_bins][self._NS_bins]
        self.NS_Nj = self.Nj[self._remnant_bins][self._NS_bins]

        self.NS_rhoj = self.rhoj[self._remnant_bins][self._NS_bins]
        self.NS_Sigmaj = self.Sigmaj[self._remnant_bins][self._NS_bins]

    # ----------------------------------------------------------------------
    # Alternative generators
    # ----------------------------------------------------------------------

    @classmethod
    def isotropic(cls, W0, M, rh, **kw):
        '''Initialize the model with max `ra`, leading to an isotropic model'''
        ra = 1e8
        return cls(W0, M, rh, ra=ra, **kw)

    @classmethod
    def canonical(cls, W0, M, rh, imf='kroupa', **kw):
        '''Initialize with an IMF defined by a canonical IMF formulation'''

        if imf.lower() == 'kroupa':
            a1, a2, a3 = 1.3, 2.3, 2.3
            m_breaks = [0.08, 0.5, 1.0, 100]  # only upper two kroupa exponents

        elif imf.lower() == 'salpeter':
            # TODO once evolve_mf supports any number of exponents, use 1 here
            a1 = a2 = a3 = 2.35

        elif imf.lower() == 'baumgardt':
            a1, a2, a3 = 0.3, 1.65, 2.35
            m_breaks = [0.1, 0.4, 1.0, 100]

        else:
            avail = ('kroupa', 'salpeter', 'baumgardt')
            mssg = f"Unknown IMF: {imf}. Must be one of {avail}"
            raise ValueError(mssg)

        return cls(W0, M, rh, a1=a1, a2=a2, a3=a3, m_breaks=m_breaks, **kw)

    @classmethod
    def woolley(cls, W0, M, rh, **kw):
        '''Initialize a Woolley (1954) Model (g=0 and isotropic)'''
        g = 0
        return cls.isotropic(W0, M, rh, g=g, **kw)

    @classmethod
    def king(cls, W0, M, rh, **kw):
        '''Initialize a King (1966) Model (g=1 and isotropic)'''
        g = 1
        return cls.isotropic(W0, M, rh, g=g, **kw)

    @classmethod
    def wilson(cls, W0, M, rh, **kw):
        '''Initialize a Wilson (1975) Model (g=2 and isotropic)'''
        g = 2
        return cls.isotropic(W0, M, rh, g=g, **kw)

    @classmethod
    def michieking(cls, W0, M, rh, **kw):
        '''Initialize a Michie-King (1963) Model (g=1 and anisotropic)'''
        g = 1
        return cls(W0, M, rh, g=g, **kw)

    # ----------------------------------------------------------------------
    # Model sampling
    # ----------------------------------------------------------------------

    def sample(self, *args, **kwargs):
        '''Return a `SampledModel` instance based on this model'''
        return SampledModel(self, *args, **kwargs)

    # ----------------------------------------------------------------------
    # Model visualizers
    # ----------------------------------------------------------------------

    def get_visualizer(self):
        '''Return a `analysis.ModelVisualizer` instance based on this model'''
        from ..analysis import ModelVisualizer
        return ModelVisualizer(self, observations=self.observations)

# --------------------------------------------------------------------------
# Single-mass version of base model
# --------------------------------------------------------------------------


class SingleMassModel(lp.limepy):
    '''like model, but without all the nice mass stuff. please, dont use
    these aren't physical, think about what you're doing man.
    '''

    def _assign_units(self):
        '''Convert most values to `astropy.Quantity` with correct units'''

        # TODO this needs to be much more general
        #   Right now it is only applied to those params we use in likelihoods?
        #   Also the actualy units used are being set manually

        if not self.scale:
            return

        G_units = u.Unit('(pc km2) / (s2 Msun)')
        R_units = u.pc
        M_units = u.Msun
        V2_units = G_units * M_units / R_units

        self.G <<= G_units

        self.M <<= M_units
        self.mc <<= M_units

        self.r <<= R_units
        self.r0 <<= R_units
        self.rh <<= R_units
        self.rhp <<= R_units
        self.rt <<= R_units
        self.ra <<= R_units
        self.rv <<= R_units
        self.rs <<= R_units

        # TODO this may be wrong (it's "phase-space" volume)
        self.volume <<= R_units**3

        self.v2T <<= V2_units
        self.v2R <<= V2_units
        self.v2p <<= V2_units
        self.s2 <<= V2_units

        self.rho <<= (M_units / R_units**3)
        self.Sigma <<= (M_units / R_units**2)

    def __init__(self, W0, M, rh, ra=1e8, g=1.5, d=5, *,
                 ode_maxstep=1e10, ode_rtol=1e-7):

        # ------------------------------------------------------------------
        # Add/convert units of some quantities. Supports quantities as inputs
        # ------------------------------------------------------------------

        M <<= u.Msun
        rh <<= u.pc
        d <<= u.kpc

        # ------------------------------------------------------------------
        # Pack theta
        # ------------------------------------------------------------------

        self.theta = dict(W0=W0, M=M / 1e6, rh=rh, ra=np.log10(ra), g=g,
                          delta=None, a1=None, a2=None, a3=None, BHret=None,
                          s2=None, F=None, d=d)

        self.d = d

        # ------------------------------------------------------------------
        # Create the limepy model base
        # ------------------------------------------------------------------

        self._limepy_kwargs = dict(
            phi0=W0,
            g=g,
            M=M.value,
            rh=rh.value,
            ra=ra,
            project=True,
            verbose=False,
            max_step=ode_maxstep,
            ode_rtol=ode_rtol
        )

        try:
            super().__init__(**self._limepy_kwargs)
        except ValueError as err:
            cause = err.args[0]

            if ("rmax reached in mf iteration" in cause
                    or "maximum number of iterations reached" in cause):

                mssg = ("Model solver failed to converge in time. "
                        "Model parameters must be adjusted")
                raise ValueError(mssg) from err

            else:
                raise err

        if not self.converged:
            mssg = "Model solver failed to converge to a finite extent"
            raise ValueError(mssg)

        # ------------------------------------------------------------------
        # Assign units to model values
        # ------------------------------------------------------------------

        self._assign_units()

        self.unscaled_ra = self.ra / self.rs

    # ----------------------------------------------------------------------
    # Alternative generators
    # ----------------------------------------------------------------------

    @classmethod
    def isotropic(cls, W0, M, rh, **kw):
        '''initialize with no anisotropy'''
        ra = 1e8
        return cls(W0, M, rh, ra=ra, **kw)

    @classmethod
    def woolley(cls, W0, M, rh, **kw):
        '''g=0, isotropic'''
        g = 0
        return cls.isotropic(W0, M, rh, g=g, **kw)

    @classmethod
    def king(cls, W0, M, rh, **kw):
        '''g=1, isotropic'''
        g = 1
        return cls.isotropic(W0, M, rh, g=g, **kw)

    @classmethod
    def wilson(cls, W0, M, rh, **kw):
        '''g=2, isotropic'''
        g = 2
        return cls.isotropic(W0, M, rh, g=g, **kw)

    @classmethod
    def michieking(cls, W0, M, rh, **kw):
        '''g=1, anisotropic'''
        g = 1
        return cls(W0, M, rh, g=g, **kw)


# --------------------------------------------------------------------------
# Model to be used in fitting to observations
# --------------------------------------------------------------------------


class FittableModel(Model):
    '''Model subclass for use in all fitting functions

    A subclass of the base `Model`, with a simplified and specific
    initilization signature based on a single `theta` input containing the main
    13 model parameters, in a specific order, and `observations` which the
    model should be compared to.

    Unless you have a set of paramters `theta` taken directly from the fitting
    results, you most likely do not want to use this class directly.

    Parameters
    ----------
    theta : dict or list
        The model input parameters. Must either be a dict, or a full list of
        all parameters, in the exact same order as `DEFAULT_THETA`.
        The 13 free parameters used here (W0, M, rh, ra, g, delta, a1, a2, a3,
        BHret, s2, F and d) are key for defining the model structure, mass
        evolution algorithm and fitting parameters.
        See `Model` for further explanation of all possible input parameters.

    observations : Observations, optional
        The `Observations` instance corresponding to this cluster. Required at
        initilization so that the models can be compared to these observations
        in the most consistent way possible.

    Attributes
    ----------
    theta : dict
        Dictionary of input parameters.
        Some parameters may technically also be accessible directly as
        attributes, but that interface should not be considered stable.
        This dictionary should be used as the only direct access to any input
        parameters that make up theta.

    Notes
    -----
    The units of the inputs in `theta` here do not match those in `Model`
    directly. `M` should be in units of [1e6 Msun] and ra should actually be
    log10(ra).

    All cluster metadata parameters (such as age, vesc, etc.) will be read from
    the observations, and should not be provided as arguments here.

    '''

    def __init__(self, theta, observations, **kwargs):

        self.observations = observations

        # ------------------------------------------------------------------
        # Unpack theta
        # ------------------------------------------------------------------

        if not isinstance(theta, dict):
            theta = dict(zip(DEFAULT_THETA, theta))

        else:
            theta = theta.copy()

        if missing_params := (DEFAULT_THETA.keys() - theta.keys()):
            mssg = f"Missing required params: {missing_params}"
            raise KeyError(mssg)

        self.theta = theta

        # ------------------------------------------------------------------
        # Convert a few quantities
        # ------------------------------------------------------------------

        theta['M'] = theta['M'] * 1e6

        theta['ra'] = 10**theta['ra']

        # ------------------------------------------------------------------
        # Create the base model
        # ------------------------------------------------------------------

        kwargs = kwargs.copy()
        kwargs.setdefault('vesc', None)
        kwargs.setdefault('Ndot', None)

        super().__init__(observations=observations, **theta, **kwargs)


# --------------------------------------------------------------------------
# Sampled model
# --------------------------------------------------------------------------


# Some helpful namespaces for SampledModel
_position = namedtuple('position', ['x', 'y', 'z', 'r', 'theta', 'phi'],
                       defaults=[None, ] * 6)
_direction = namedtuple('direction', ['x', 'y', 'z', 'r', 't', 'theta', 'phi'],
                        defaults=[None, ] * 7)
_projection = namedtuple('projection', ['lat', 'lon', 'distance',
                                        'pm_l_cosb', 'pm_b', 'v_los'],
                         defaults=[None, ] * 6)


class SampledModel:
    '''representation of a cluster based on sampling a Model'''

    # TODO get initial masses as well, so can add stuff like photometry

    centre = None
    galactic = None

    # ----------------------------------------------------------------------
    # Initial sampling of a given model
    # ----------------------------------------------------------------------

    def _sample_rkvφ(self, model, pool=None):
        from scipy.special import dawsn, gammainc

        # ------------------------------------------------------------------
        # Helper PDF functions for velocity sampling
        # ------------------------------------------------------------------

        def _Eg(x):
            expx = np.exp(x)
            return expx * gammainc(self.g, x) if self.g > 0 else expx

        def _pdf_k32(x, r, phihat, p, sig2fac):
            '''
            PDF of k^3/2

            r=sampled self.r
            f = sampled phihat
            x = x
            j = mass bin j
            '''

            x13, x23 = x**(1. / 3), x**(2. / 3)

            E = (phihat - x23) * sig2fac

            c = (x > 0)

            prob = np.full_like(x, np.nan)

            # If anisotropic, use full PDF
            if self.ani:

                # Where x > 0, compute full PDF
                F = dawsn(x13 * p * np.sqrt(sig2fac)) / p
                prob[c] = F[c] / (x13[c] * np.sqrt(sig2fac[c])) * _Eg(E[c])

                # Where x=0, only need to compute Eg term (x should never be <0)
                prob[~c] = _Eg(E[~c])

            # if isotropic, only need to compute E term, same for all x
            else:
                prob = _Eg(E)

            return prob

        # ------------------------------------------------------------------
        # Sampling of radial distance from centre (r)
        # ------------------------------------------------------------------

        mcj_norm = model.mcj / model.mcj[:, -1, np.newaxis]
        r = np.concatenate([
            np.interp(self.rng.uniform(size=N), mcj_norm[j], model.r)
            for j, N in enumerate(self.Nj)
        ])

        # ------------------------------------------------------------------
        # Sampling of total stellar velocity (v)
        # ------------------------------------------------------------------

        # Setup of 10 PDF segment edges for constructing the ziggurat

        phihat = model.interp_phi(r) / model.s2.value
        segmax = phihat**1.5

        nseg = 10
        xseg = np.outer(np.linspace(0, 1, nseg + 1), segmax)

        # Initialize final output arrays

        k = np.full_like(r.value, np.nan)
        v = np.full_like(r.value, np.nan)

        # Loop over and sample velocities for each mass bin individually

        # TODO add multiprocessing over this loop
        for j in range(model.nmbin):

            # Grab relevant quantities only in this mass bin

            mask = (self.mbins == j)

            xj = xseg[:, mask]
            rj = r[mask]
            phij = phihat[mask]

            kj = k[mask]
            vj = v[mask]

            # Helper normalization quantities to avoid recomputing each time

            pj = rj / self.raj[mask]
            s2fj = self.σ2_factor[mask]

            # Compute the PDF at all segment edges

            # TODO why are these vstack needed? the last ind isn't even used?
            yj = _pdf_k32(xj[:-1], rj, phij, pj,
                          np.repeat(s2fj[None, ...], nseg, axis=0).value)
            yj = np.vstack((yj, np.zeros(yj.shape[1]), xj[-1]))

            # Compute the CDF at all segment edges

            ycumj = np.cumsum(yj[:-2] * np.diff(xj, axis=0), axis=0)
            ycumj = np.vstack((np.zeros(yj.shape[1]), ycumj))

            # Sample k iteratively until all stars sampled once

            to_sample = np.full(yj.shape[-1], True)

            while to_sample.any():

                # Unpack quantities only for stars left to be sampled

                rj_samp = rj[to_sample]
                phitmp = phij[to_sample]

                yj_samp = yj[:, to_sample]
                xj_samp = xj[:, to_sample]
                ycumj_samp = ycumj[:, to_sample]

                kj_samp = kj[to_sample]
                vj_samp = vj[to_sample]

                pj_samp = pj[to_sample]
                s2fj_samp = s2fj[to_sample].value

                # Generate the first random value

                R = self.rng.uniform(0, ycumj_samp[-1])

                # Setup arrays to hold sample values and probabilities

                xsampled = np.zeros(yj_samp.shape[1])
                P = np.zeros(yj_samp.shape[1])

                # Check value of random R against all segment bounds

                for i in range(nseg):

                    # If sample is between the segs (in CDF), get prob and save

                    btwn = (ycumj_samp[i] <= R) & (R < ycumj_samp[i + 1])

                    P[btwn] = yj_samp[i, btwn]

                    prop = (R[btwn] - ycumj_samp[i, btwn]) / yj_samp[i, btwn]
                    xsampled[btwn] = xj_samp[i, btwn] + prop

                # Generate second random value between 0 and PDF at highest btwn

                R2 = self.rng.uniform(0, P)

                # Compute pdf at sampled x's

                # TODO could maybe mask out all xsampled==not set above??
                f = _pdf_k32(xsampled, rj_samp, phitmp, pj_samp, s2fj_samp)

                # If PDF(sampled x's) > R2, keep this sample

                chk = (f > R2)

                # Compute and store the accepted k and velocity values

                # TODO painful I cant simply combine many masks here at once
                kj_samp[chk] = xsampled[chk]**(2 / 3.)
                vj_samp[chk] = np.sqrt(2 * xsampled[chk]**(2 / 3.) * model.s2)

                kj[to_sample] = kj_samp
                vj[to_sample] = vj_samp

                # Continue until all stars sampled once

                to_sample[to_sample] = ~chk

            k[mask] = kj
            v[mask] = vj

            # Shuffle all the vels/rad computed in this mass bin
            #   supposedly to avoid correlations

            shuffle_key = self.rng.permutation(r[mask].size)
            k[mask] = k[mask][shuffle_key]
            v[mask] = v[mask][shuffle_key]
            r[mask] = r[mask][shuffle_key]

            phihat[mask] = phihat[mask][shuffle_key]

        # Add units

        # TODO I assume vel units are km/s, but should check
        v <<= (u.km / u.s)

        return r, k, v, phihat

    def _sample_coordinates(self, model, pool=None):
        from scipy import optimize

        # ------------------------------------------------------------------
        # Helper PDF functions for velocity angle sampling
        # ------------------------------------------------------------------

        def _pdf_angle(q, a, R):
            # Sample random values for: q = cos(theta)
            # P(q) = erfi(sqrt(k)*p*q)/erfi(sqrt(k)*p)
            from scipy.special import erfi
            return R - erfi(a * q) / erfi(a)

        def solver(a, R):
            return optimize.brentq(_pdf_angle, 0, 1, args=(a, R))

        # ------------------------------------------------------------------
        # Sampling of angles for radial/tangential velocities
        # ------------------------------------------------------------------

        R = self.rng.uniform(size=self.N)

        # Anisotropic systems sample angles:
        #   cdf(q) = erfi(a*q)/erfi(a)
        #   where q = cos(theta), a = sqrt(k)*p

        if self.ani:
            p = self.r / self.raj
            a = (p * np.sqrt(self.k) * np.sqrt(self.σ2_factor)).value

            # Get map, in case we want to use multiprocessing

            _map = pool.starmap if pool else itertools.starmap

            # TODO multiprocessing doesn't seem to increase speed at all?
            q = np.fromiter(_map(solver, np.transpose([a, R])), dtype=float)

            # TODO must be better way; vectorized opt.root takes 15Tb of mem
            # q = optimize.root(_pdf_angle, np.zeros(self.N), args=(a, R))
            #
            # q = np.zeros(self.N)
            # for i in range(self.N):
            #     q[i] = optimize.brentq(_pdf_angle, 0, 1, args=(a[i], R[i]))

        # Isotropic: cdf(q) = q

        else:
            q = R

        # Compute radial and tangential velocities from sampled angles

        # TODO should q (cos(θ), but not that θ) also be saved? Interesting?
        vr = self.v * q * self.rng.choice((-1, 1), size=self.N)
        vt = self.v * np.sqrt(1 - q**2)

        # ------------------------------------------------------------------
        # Sampling of 3D positions in both cartesian and spherical coordinates
        # ------------------------------------------------------------------

        # Generate random angles for 3D positions

        R1 = self.rng.uniform(size=self.N)
        R2 = self.rng.uniform(0, 2 * np.pi, size=self.N)

        # Compute x, y, z coordinates based on the generated angles

        x = (1 - 2 * R1) * self.r
        y = np.sqrt(self.r**2 - x**2) * np.cos(R2)
        z = np.sqrt(self.r**2 - x**2) * np.sin(R2)

        # Compute angles θ, φ for the spherical coordinate system, from x, y, z

        theta = np.arccos(z / self.r)
        phi = np.arctan2(y, x)

        # ------------------------------------------------------------------
        # Sampling of velocity in 3D directions, in spherical and cartesian
        # ------------------------------------------------------------------

        # Compute velocities in θ, φ for the spherical coordinate system

        R1 = self.rng.uniform(size=self.N)

        # TODO limepy only computed these if anisotropic, why? not interesting?
        vphi = vt * np.cos(2 * np.pi * R1)
        vtheta = vt * np.sin(2 * np.pi * R1)

        # Compute velocities in cartesian coordinate systems

        if self.ani:

            # If anisotropic, use sampled angles (vr/vt) and spherical vels

            vx = (vr * np.sin(theta) * np.cos(phi)
                  + vtheta * np.cos(theta) * np.cos(phi)
                  - vphi * np.sin(phi))
            vy = (vr * np.sin(theta) * np.sin(phi)
                  + vtheta * np.cos(theta) * np.sin(phi)
                  + vphi * np.cos(phi))
            vz = vr * np.cos(theta) - vtheta * np.sin(theta)

        else:

            # If isotropic, generate completely random angles in both directions
            # TODO why not just use same stuff for ani/iso? vr=vt, but it exists

            R1 = self.rng.uniform(size=self.N)
            R2 = self.rng.uniform(size=self.N)

            vx = (1 - 2 * R1) * self.v
            vy = np.sqrt(self.v**2 - vx**2) * np.cos(2 * np.pi * R2)
            vz = np.sqrt(self.v**2 - vx**2) * np.sin(2 * np.pi * R2)

        # ------------------------------------------------------------------
        # Place the various positions/velocities into convenient namespaces
        # ------------------------------------------------------------------

        p = _position(x=x, y=y, z=z, r=self.r, theta=theta, phi=phi)
        v = _direction(x=vx, y=vy, z=vz, r=vr, t=vt, phi=vphi, theta=vtheta)

        return p, v

    def _project(self, cen):
        '''return projected-on-sky positions and velocities, given the centre
        skycoord, into the galactic frame (spherical)
        # TODO only gives galactic, is that fine?
        '''
        import erfa

        cen = cen.galactic

        x, y, z = cen.u - self.pos.x, cen.v - self.pos.y, cen.w - self.pos.z
        vx, vy, vz = cen.U - self.vel.x, cen.V - self.vel.y, cen.W - self.vel.z

        pv = util.q2pv(np.c_[x, y, z], np.c_[vx, vy, vz])
        l, b, d, dl, db, dd = erfa.pv2s(pv)

        return _projection(lon=l.to('deg'), lat=b.to('deg'), distance=d,
                           pm_l_cosb=dl.to('mas/yr') * np.cos(b),
                           pm_b=db.to('mas/yr'), v_los=dd)

    def __init__(self, model, centre=None, *, distribute_masses=True,
                 use_model_distance=True, seed=None, pool=None, verbose=False):
        '''Sample N stars with positions and velocities from this model

        if centre is given, will determine projected/observed
            coordinates/velocities for each star based on this centre.
            Centre should be a SkyCoord, and also contain velocities
            The model "distance" `d` will set the distance of the exact centre
            of the cluster, overriding the distance in the given centre
            skycoord, unless `use_model_distance` is False.
        if use_model_distance is False and the distance is not very close to
            the model distance, this will obvisouly give very weird results

        if distribute_masses, masses will be uniformally sampled between all
            the mass bins, otherwise will all have the mean value (mj)
        '''

        # store ref to base model, just in case
        self._basemodel = model

        # ------------------------------------------------------------------
        # Set random seed for all sampling
        # ------------------------------------------------------------------

        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Set and store various important quantities from the base model
        # ------------------------------------------------------------------

        # TODO allow other desired N's, would need to match Nj proportions?
        self.Nj = model.Nj.astype(int)

        # Determine stellar types for each star (MS, WD, NS, BH)
        self.star_types = np.repeat(model.star_types, self.Nj)
        self.star_mask = (self.star_types == 'MS')

        self.WD_mask = (self.star_types == 'WD')
        self.NS_mask = (self.star_types == 'NS')
        self.BH_mask = (self.star_types == 'BH')

        self.N = self.Nj.sum()
        self.Nstars = self.Nj[model.star_types == 'MS'].sum()
        self.Nrems = self.N - self.Nstars

        self.ani = True if min(model.raj) / model.rt < 3 else False

        self.g = model.g
        self.d = model.d

        self.rh = model.rh
        self.rhj = model.rhj
        self.rhp = model.rhp
        self.rt = model.rt

        self.raj = np.repeat(model.raj, self.Nj)
        self.s2j = np.repeat(model.s2j, self.Nj)
        self.σ2_factor = model.s2 / self.s2j

        try:
            self.age = model.observations.mdata['age'] << u.Gyr
            self.feh = model.observations.mdata['FeH']
        except (AttributeError, KeyError):
            pass

        # ------------------------------------------------------------------
        # "Sample" the masses, assuming all stars are exactly mean bin mass
        # ------------------------------------------------------------------

        self.m = np.repeat(model.mj, self.Nj)

        if distribute_masses:
            halfwidth = np.repeat(model.mbin_widths / 2., self.Nj)
            low, high = self.m - halfwidth, self.m + halfwidth
            self.m = self.rng.uniform(low, high)

        self.mbins = np.repeat(range(model.nmbin), self.Nj)

        # Recompute actual total mass
        self.M = self.m.sum()

        # ------------------------------------------------------------------
        # Sample the (unprojected) positions and velocities
        # ------------------------------------------------------------------

        self.r, self.k, self.v, self.phihat = self._sample_rkvφ(model)

        self.phi = -self.phihat * model.s2 - model.G * model.M / model.rt

        # ------------------------------------------------------------------
        # Project the positions and velocities by randomly sampling angles
        # and computing in both cartesian and spherical coordinate systems
        # ------------------------------------------------------------------

        self.pos, self.vel = self._sample_coordinates(model)

        # ------------------------------------------------------------------
        # If centre coordinate is given, project also into the galactic frame
        # ------------------------------------------------------------------

        if centre is not None:
            import astropy.coordinates as coord

            # --------------------------------------------------------------
            # Unpack centre coordinate into a 3d SkyCoord
            # TODO this is not very robust, must be better way to inject dist
            # --------------------------------------------------------------

            if isinstance(centre, coord.SkyCoord):

                centre = centre.copy().galactic

                # Convert to spherical first, so can implant our distance

                if use_model_distance:
                    centre.representation_type = 'spherical'
                    centre.differential_type = 'sphericalcoslat'

                    centre = coord.SkyCoord(
                        l=centre.l, b=centre.b, distance=model.d,
                        pm_l_cosb=centre.pm_l_cosb, pm_b=centre.pm_b,
                        radial_velocity=centre.radial_velocity,
                        frame='galactic'
                    )

                # convert to cartesian to actually use

                centre.representation_type = 'cartesian'

            else:
                mssg = f"'centre' must be a 'SkyCoord', not {type(centre)}"
                raise ValueError(mssg)

            # --------------------------------------------------------------
            # Project all the cartesian positions into this frame
            # --------------------------------------------------------------

            # TODO store this in the best possible frame/representation
            self.centre = centre

            self.galactic = self._project(centre)
