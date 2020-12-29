'''READONLY access to data in cluster resources files'''
import numpy as np
import h5py

import os
import glob
import logging
from importlib import resources

# TODO better exception handling
# TODO __str__ methods, for better error messages

# Acceleration space for which we generate a probability distribution.
# TODO generate this based on the data (and do it in likelihoods probably)
# A_SPACE = np.linspace(-15e-9, 15e-9, 300)
A_SPACE = np.linspace(-5e-8, 5e-8, 300)

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


class Dataset:
    '''each group of observations, like mass_function, proper_motions, etc
    init from a h5py group
    not to be confused with h5py datasets, this is more analogous to a group

    h5py attributes are called metadata here cause that is more descriptive
    '''
    class _Variable(np.ndarray):
        '''simple readonly subclass to allow metadata dict on the variable'''
        def __new__(cls, input_array, mdata=None):
            # TODO mdata doesn't carry over into views, might need _finalize_

            obj = np.asarray(input_array).view(cls)

            if isinstance(mdata, dict):
                obj.mdata = mdata
            elif mdata is None:
                obj.mdata = dict()
            else:
                raise TypeError('`mdata` must be a dict or None')

            obj.flags.writeable = False

            return obj

    def __contains__(self, key):
        return key in self._dict_variables

    def __getitem__(self, key):
        return self._dict_variables[key]

    def _init_variables(self, name, var):
        '''used by group.visit'''

        if isinstance(var, h5py.Dataset):
            mdata = dict(var.attrs)
            self._dict_variables[name] = self._Variable(var[:], mdata=mdata)

    def __init__(self, group):

        self._dict_variables = {}
        group.visititems(self._init_variables)

        self.mdata = dict(group.attrs)

    @property
    def variables(self):
        return self._dict_variables

    def compute_async_error(self, varname, quantity):
        '''TODO this is where the async error stuff should go'''
        pass

    def convert_units(self):
        # TODO auto unit conversions based on attributes
        pass


class Observations:
    '''Collection of Datasets, read from a corresponding hdf5 file'''

    @property
    def datasets(self):
        return self._dict_datasets

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

                self.mdata = dict(file.attrs)


def get_dataset(cluster, key):
    '''get a dataset corresponding to a key

    indices is either a slice or a tuple for making a slice or a nparray mask
        mathcing the size of the data. None will return everything (:)
    '''

    with resources.path('fitter', 'resources') as datadir:
        with h5py.File(f'{datadir}/{cluster}.hdf5', 'r') as file:

            return file[key][:]


def cluster_list():
    with resources.path('fitter', 'resources') as datadir:

        return [os.path.splitext(os.path.basename(fn))[0]
                for fn in glob.iglob(f'{datadir}/[!TEST]*.hdf5')]
