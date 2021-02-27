'''observational and modelled data'''

from . import util

import h5py
import numpy as np
import limepy as lp
from astropy import units as u
from ssptools import evolve_mf_3 as emf3

import fnmatch
import logging
from importlib import resources

# TODO better exception handling

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


# --------------------------------------------------------------------------
# Cluster Observational Data
# --------------------------------------------------------------------------


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

    def __contains__(self, key):
        return key in self._dict_variables

    def __getitem__(self, key):
        return self._dict_variables[key]

    def _init_variables(self, name, var):
        '''used by group.visit'''

        if isinstance(var, h5py.Dataset):
            mdata = dict(var.attrs)
            self._dict_variables[name] = Variable(var[:], mdata=mdata)

    def __init__(self, group):

        self._dict_variables = {}
        group.visititems(self._init_variables)

        self.mdata = dict(group.attrs)

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
                mssg = f"There are no error(Δ) values associated with {varname}"
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
        except KeyError:
            try:
                # return a variable within a dataset
                group, name = key.rsplit('/', maxsplit=1)
                return self._dict_datasets[group][name]

            except ValueError:
                # not in _dict_datasets and no '/' to split on so not a variable
                mssg = f"Dataset '{key}' does not exist"
                raise KeyError(mssg)

            except KeyError:
                # looks like a "dataset/variable" but that variable don't exist
                mssg = f"Dataset or variable '{key}' does not exist"
                raise KeyError(mssg)

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
        from . import probabilities

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

                    kde = util.pulsar_Pdot_KDE()

                    comps.append((key, func, kde, *metadata))

                if 'Pb' in self[key]:

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

# TODO The units are *very* incomplete in Model (10)

class Model(lp.limepy):

    def __getattr__(self, key):
        '''If `key` is not defined in the limepy model, try to get it from θ'''
        return self._theta[key]

    def _init_mf(self):

        m123 = [0.1, 0.5, 1.0, 100]  # Slope breakpoints for imf
        nbin12 = [5, 5, 20]

        a12 = [-self.a1, -self.a2, -self.a3]  # Slopes for imf

        # Output times for the evolution (age)
        tout = np.array([11000])

        # TODO figure out which of these are cluster dependant, store in hdfs

        # Integration settings
        N0 = 5e5  # Normalization of stars
        tcc = 0  # Core collapse time
        NS_ret = 0.1  # Initial neutron star retention
        BH_ret_int = 1  # Initial Black Hole retention
        BH_ret_dyn = self.BHret / 100  # Dynamical Black Hole retention

        # Metallicity
        try:
            FeHe = self.observations.mdata['FeHe']
        except (AttributeError, KeyError):
            logging.debug("No cluster FeHe stored, defaulting to -1.02")
            FeHe = -1.02

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
            tout=tout,
            N0=N0,
            Ndot=Ndot,
            tcc=tcc,
            NS_ret=NS_ret,
            BH_ret_int=BH_ret_int,
            BH_ret_dyn=BH_ret_dyn,
            FeHe=FeHe,
            natal_kicks=False
        )

    # def _get_scale(self):
    #     G_scale, M_scale, R_scale = self._GS, self._MS, self._RS

    def _assign_units(self):
        # TODO this needs to be much more general
        #   Right now it is only applied to those params we use in likelihoods?
        #   Also the actualy units used are being set manually

        # TODO I have no idea how the scaling is supposed to work in limepy

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
            mssg = f"Missing required params: {missing_params}"
            raise KeyError(mssg)

        self._theta = theta

        # ------------------------------------------------------------------
        # Get mass function
        # ------------------------------------------------------------------

        self._mf = self._init_mf()

        # Set bins that should be empty to empty
        cs = self._mf.Ns[-1] > 10 * self._mf.Nmin
        cr = self._mf.Nr[-1] > 10 * self._mf.Nmin

        # Collect mean mass and total mass bins
        mj = np.r_[self._mf.ms[-1][cs], self._mf.mr[-1][cr]]
        Mj = np.r_[self._mf.Ms[-1][cs], self._mf.Mr[-1][cr]]

        # append tracer mass bins (must be appended to end to not affect nms)
        if observations is not None:

            tracer_mj = np.unique([
                dataset.mdata['m'] for dataset in observations.datasets.values()
                if 'm' in dataset.mdata
            ])

            # TODO shouldn't append multiple of same tracer mass
            mj = np.concatenate((mj, tracer_mj))
            Mj = np.concatenate((Mj, 0.1 * np.ones_like(tracer_mj)))

        else:
            logging.warning("No `Observations` given, no tracer masses added")

        # store some necessary mass function info in the model
        self.nms = len(self._mf.ms[-1][cs])
        self.mes_widths = self._mf.mes[-1][1:] - self._mf.mes[-1][:-1]

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

        # ------------------------------------------------------------------
        # Assign units to model values
        # ------------------------------------------------------------------

        self._assign_units()

        # ------------------------------------------------------------------
        # Get Black Holes
        # ------------------------------------------------------------------

        self._BH_bins = self.mj > (self._mf.IFMR.mBH_min << u.Msun)

        self.BH_mj = self.mj[self._BH_bins]
        self.BH_Mj = self.Mj[self._BH_bins]

        self.BH_rhoj = self.rhoj[self._BH_bins]
        self.BH_Sigmaj = self.Sigmaj[self._BH_bins]

        # self.BH_Nj =
