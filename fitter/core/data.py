import h5py
import numpy as np
import limepy as lp
from astropy import units as u
from ssptools import evolve_mf_3 as emf3

import fnmatch
import logging
from importlib import resources


__all__ = ['DEFAULT_INITIALS', 'DEFAULT_PRIORS', 'Model', 'Observations']


# The order of this is important!
DEFAULT_INITIALS = {
    'W0': 6.0,
    'M': 0.69,
    'rh': 2.88,
    'ra': 1.23,
    'g': 0.75,
    'delta': 0.45,
    's2': 0.1,
    'F': 0.45,
    'a1': 0.5,
    'a2': 1.3,
    'a3': 2.5,
    'BHret': 0.5,
    'd': 6.405,
}

DEFAULT_PRIORS = {
    'W0': (3, 20),
    'M': (0.01, 10),
    'rh': (0.5, 15),
    'ra': (0, 5),
    'g': (0, 2.3),
    'delta': (0.3, 0.5),
    's2': (0, 15),
    'F': (0, 0.5),
    'a1': (-2, 6),
    'a2': (-2, 6),
    'a3': (1.6, 6),
    'BHret': (0, 100),
    'd': (2, 8),
}


# --------------------------------------------------------------------------
# Cluster Observational Data
# --------------------------------------------------------------------------
# TODO maybe define a new excepton for when a req'd thing is not in an obs


class Variable(u.Quantity):
    '''simple readonly Quantity subclass to allow metadata on the variable
    '''
    # TODO better way to handle string arrays, and with nicer method failures
    # TODO the "readonly" part of Variable is currently not functional
    def __new__(cls, value, unit=None, mdata=None, *args, **kwargs):

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

        # quant.flags.writeable = False

        return quant

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.mdata = getattr(obj, 'mdata', dict(defaulted=True))

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
    '''each group of observations, like mass_function, proper_motions, etc
    init from a h5py group
    not to be confused with h5py datasets, this is more analogous to a group

    h5py attributes are called metadata here cause that is more descriptive
    '''

    def __repr__(self):
        return f'Dataset("{self._name}")'

    def __str__(self):
        return f'{self._name} Dataset'

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
            self._dict_variables[name] = Variable(var[:], mdata=mdata)

    def __init__(self, group):

        self._dict_variables = {}
        group.visititems(self._init_variables)

        self.mdata = dict(group.attrs)

        self._name = group.name

    @property
    def variables(self):
        return self._dict_variables

    def build_err(self, varname, model_r, model_val):
        '''
        varname is the variable we want to get the error for
        quantity is the actual model data we will be comparing this with

        model_r, _val must be in equivalent units to the var already.
        conversion will be attempted, but with no equivalencies
        '''

        quantity = self[varname]

        # ------------------------------------------------------------------
        # Attempt to convert model values
        # ------------------------------------------------------------------

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
    '''Collection of Datasets, read from a corresponding hdf5 file (READONLY)'''
    # TODO interesting errors occur when trying to iterate over Observ

    def __repr__(self):
        return f'Observations(cluster="{self.cluster}")'

    def __str__(self):
        return f'{self.cluster} Observations'

    @property
    def datasets(self):
        return self._dict_datasets

    @property
    def valid_likelihoods(self):
        # TODO Set this up so they're only generated once since shouldnt change
        return self._determine_likelihoods()

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
                mssg = f"Dataset '{key}' does not exist in {self}"
                raise KeyError(mssg) from err

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

    def __init__(self, cluster):

        # TODO add a common names sort of thing for cluster names
        self.cluster = cluster

        self.mdata = {}
        self._dict_datasets = {}
        self.initials = DEFAULT_INITIALS.copy()

        with resources.path('fitter', 'resources') as datadir:
            with h5py.File(f'{datadir}/{cluster}.hdf5', 'r') as file:

                logging.info(f"Observations read from {datadir}/{cluster}.hdf5")

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
        '''from observations, determine which likelihood functions will be
        computed and return a dict of the relevant obs dataset keys, and tuples
        of the functions and any other required args
        '''
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
            # Each component is a tuple of where the first two elements are,
            # respectively, the observation key and likelihood function, and all
            # remaining elements are the extra arguments to pass to the function
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # Pulsar probabilities
            # --------------------------------------------------------------

            if fnmatch.fnmatch(key, '*pulsar*'):

                metadata = self.mdata['μ'], (self.mdata['b'], self.mdata['l'])

                if 'Pdot_meas' in self[key]:

                    func = probabilities.likelihood_pulsar_spin

                    kde = probabilities.pulsars.field_Pdot_KDE()

                    comps.append((key, func, kde, *metadata))

                if 'Pbdot_meas' in self[key]:

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

                comps.append((key, func, ))

        return comps

# --------------------------------------------------------------------------
# Cluster Modelled data
# --------------------------------------------------------------------------

# TODO The units are *quite* incomplete in Model (10)

class Model(lp.limepy):

    def __getattr__(self, key):
        '''If `key` is not defined in the limepy model, try to get it from θ'''
        try:
            return self._theta[key]
        except KeyError as err:
            msg = f"'{self.__class__.__name__}' object has no attribute '{key}'"
            raise AttributeError(msg) from err

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

        # Regulates low mass objects depletion, default -20, 0 for 47 Tuc
        try:
            Ndot = self.observations.mdata['Ndot']
        except (AttributeError, KeyError):
            logging.debug("No cluster Ndot stored, defaulting to -20")
            Ndot = -20

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
        self.mes_widths = np.diff(self._mf.mes[-1])

        # append tracer mass bins (must be appended to end to not affect nms)
        if observations is not None:

            # TODO should only append tracer masses for valid likelihood dsets
            tracer_mj = np.unique([
                dataset.mdata['m'] for dataset in observations.datasets.values()
                if 'm' in dataset.mdata
            ])

            # TODO shouldn't append multiple of same tracer mass
            mj = np.concatenate((mj, tracer_mj))
            Mj = np.concatenate((Mj, 0.1 * np.ones_like(tracer_mj)))

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
