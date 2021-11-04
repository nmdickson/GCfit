from .. import util

import h5py
import numpy as np
import limepy as lp
from astropy import units as u
from ssptools import evolve_mf_3 as emf3

import fnmatch
import logging


__all__ = ['DEFAULT_INITIALS', 'Model', 'Observations']


# The order of this is important!
DEFAULT_INITIALS = {
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
                self._citation = util.bibcode2cite(bibcodes)
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

    The main interface for reading and interacting with (_not writing_) all
    observational data for the specified globular cluster.

    Defined based on a given cluster datafile, handles creation and access to
    all contained `Dataset`s, as well as the setup and arguments for all
    relevant likelihoods.

    The relevant cluster data files will be found using the
    `fitter.util.get_cluster_path` function, and can likewise be retricted to
    "core" or "local" files. The data file used *must* be considered valid
    (i.e. pass all tests within `fitter.utils.data.ClusterFile.test`).

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
        `fitter.util.get_cluster_path` for more information.

    Attributes
    ----------
    valid_likelihoods

    datasets

    mdata : dict
        Dictionary of all "cluster-level" metadata

    See Also
    --------
    fitter.util.get_cluster_path : Locating of data file based on `cluster` name
    fitter.utils.data.ClusterFile : Handling of data file creation and editing
    '''
    # TODO interesting errors occur when trying to iterate over Observ

    _valid_likelihoods = None

    def __repr__(self):
        return f'Observations(cluster="{self.cluster}")'

    def __str__(self):
        return f'{self.cluster} Observations'

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

    # TODO a filter method for finding all datasets matching a pattern
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

        # TODO maybe `datasets` and this should only return ds list not dict?
        #   if thats the case, make `datasets._name` public

        if valid_only:
            datasets = {key for (key, *_) in self.valid_likelihoods}
        else:
            datasets = self.datasets.keys

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
        self.initials = DEFAULT_INITIALS.copy()

        filename = util.get_cluster_path(cluster, standardize_name, restrict_to)

        self.cluster = filename.stem

        with h5py.File(filename, 'r') as file:

            logging.info(f"Observations read from {filename}")

            for group in self._find_groups(file):
                self._dict_datasets[group] = Dataset(file[group])

            try:
                # This updates defaults with data while keeping default sort
                self.initials = {**self.initials, **file['initials'].attrs}
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
# Cluster Modelled data
# --------------------------------------------------------------------------

# TODO The units are *quite* incomplete in Model (10)
# TODO would be cool to get this to work with limepy's `sampling`

class Model(lp.limepy):

    def _init_mf(self):

        m123 = [0.1, 0.5, 1.0, 100]  # Slope breakpoints for imf
        nbin12 = [5, 5, 20]

        a12 = [-self.a1, -self.a2, -self.a3]  # Slopes for imf

        # TODO figure out which of these are cluster dependant, store in hdfs

        # Integration settings
        N0 = 5e5  # Normalization of stars
        tcc = 0  # Core collapse time
        NS_ret = 0.1  # Initial neutron star retention
        BH_ret_int = 1  # Initial Black Hole retention
        BH_ret_dyn = self.BHret / 100  # Dynamical Black Hole retention

        natal_kicks = True

        # Age
        try:
            age = self.observations.mdata['age'] * 1000
        except (AttributeError, KeyError):
            logging.debug("No cluster age stored, defaulting to 12 Gyr")
            age = 12 * 1000

        # Metallicity
        try:
            FeHe = self.observations.mdata['FeHe']
        except (AttributeError, KeyError):
            logging.debug("No cluster FeHe stored, defaulting to -1.0")
            FeHe = -1.0

        # Regulates low mass objects depletion
        try:
            Ndot = self.observations.mdata['Ndot']
        except (AttributeError, KeyError):
            logging.debug("No cluster Ndot stored, defaulting to 0 /Myr")
            Ndot = 0

        # Generate the mass function
        return emf3.evolve_mf(
            m123=m123,
            a12=a12,
            nbin12=nbin12,
            tout=np.array([age]),
            N0=N0,
            Ndot=Ndot,
            tcc=tcc,
            NS_ret=NS_ret,
            BH_ret_int=BH_ret_int,
            BH_ret_dyn=BH_ret_dyn,
            FeHe=FeHe,
            natal_kicks=natal_kicks
        )

    # def _get_scale(self):
    #     TODO I have no idea how the scaling is supposed to work in limepy
    #     G_scale, M_scale, R_scale = self._GS, self._MS, self._RS

    def _assign_units(self):
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
        self.mes_widths <<= M_units

        self.r <<= R_units
        self.rh <<= R_units
        self.rt <<= R_units
        self.ra <<= R_units

        self.v2Tj <<= V2_units
        self.v2Rj <<= V2_units
        self.v2pj <<= V2_units

        self.rhoj <<= (M_units / R_units**3)
        self.Sigmaj <<= (M_units / R_units**2)

        self.d <<= u.kpc

    def __init__(self, theta, observations=None, *, verbose=False):

        self.observations = observations

        if self.observations is None:
            logging.warning("No `Observations` given, if `theta` was computed "
                            "using observations, this model will not match")

        # ------------------------------------------------------------------
        # Unpack theta
        # ------------------------------------------------------------------

        if not isinstance(theta, dict):
            theta = dict(zip(DEFAULT_INITIALS, theta))

        if missing_params := (DEFAULT_INITIALS.keys() - theta.keys()):
            # TODO this error message is wrong if theta is given as an array
            mssg = f"Missing required params: {missing_params}"
            raise KeyError(mssg)

        self._theta = theta

        for key, val in self._theta.items():
            setattr(self, key, val)

        # ------------------------------------------------------------------
        # Get mass function
        # ------------------------------------------------------------------

        # TODO I think how we are handling everything with mj and bins could
        #   be done much more nicely (maybe with some sweet masked arrays)

        self._mf = self._init_mf()

        # Set bins that should be empty to empty
        cs = self._mf.Ns[-1] > 10 * self._mf.Nmin
        ms, Ms = self._mf.ms[-1][cs], self._mf.Ms[-1][cs]

        cr = self._mf.Nr[-1] > 10 * self._mf.Nmin
        mr, Mr = self._mf.mr[-1][cr], self._mf.Mr[-1][cr]

        # Collect mean mass and total mass bins
        mj = np.r_[ms, mr]
        Mj = np.r_[Ms, Mr]

        # store some necessary mass function info in the model
        self.nms = ms.size
        self.nmr = mr.size

        # TODO these kind of slices would prob be more useful than nms elsewhere
        self._star_bins = slice(0, self.nms)
        self._remnant_bins = slice(self.nms, self.nms + self.nmr)

        # TODO still don't entriely understand when this is to be used
        # mj is middle of mass bins, mes are edges, widths are sizes of bins
        # self.mbin_widths = np.diff(self._mf.mes[-1]) ??
        # Whats the differences with `mes` and `me`?
        # TODO is this supposed to habe units? I think so
        self.mes_widths = np.diff(self._mf.mes[-1])

        # append tracer mass bins (must be appended to end to not affect nms)
        if observations is not None:

            # TODO should only append tracer masses for valid likelihood dsets?
            tracer_mj = np.unique([
                dataset.mdata['m'] for dataset in observations.datasets.values()
                if 'm' in dataset.mdata
            ])

            mj = np.concatenate((mj, tracer_mj))
            Mj = np.concatenate((Mj, 0.1 * np.ones_like(tracer_mj)))

            self._tracer_bins = slice(self.nms + self.nmr, None)

        else:
            logging.warning("No `Observations` given, no tracer masses added")

        # ------------------------------------------------------------------
        # Create the limepy model base
        # ------------------------------------------------------------------

        super().__init__(
            phi0=self.W0,
            g=self.g,
            M=self.M * 1e6,
            rh=self.rh,
            ra=10**self.ra,
            delta=self.delta,
            mj=mj,
            Mj=Mj,
            project=True,
            verbose=verbose,
        )

        # fix a couple of conflicted attributes
        self.s2 = self._theta['s2']
        self.Nj = self.Mj / self.mj

        # ------------------------------------------------------------------
        # Assign units to model values
        # ------------------------------------------------------------------

        self._assign_units()

        # ------------------------------------------------------------------
        # Split apart the stellar classes of the mass bins
        # ------------------------------------------------------------------

        # TODO slight difference in mf.IFMR.mBH_min and mf.mBH_min?
        self._mBH_min = self._mf.IFMR.mBH_min << u.Msun
        self._BH_bins = self.mj[self._remnant_bins] > self._mBH_min

        self._mWD_max = self._mf.IFMR.predict(self._mf.IFMR.wd_m_max) << u.Msun
        self._WD_bins = self.mj[self._remnant_bins] < self._mWD_max

        self._NS_bins = ((self._mWD_max < self.mj[self._remnant_bins])
                         & (self.mj[self._remnant_bins] < self._mBH_min))

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
