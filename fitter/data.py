import numpy as np
import h5py

import logging
from importlib import resources

# TODO put all the units into the h5 file attributes
# TODO auto unit conversions based on ^
# TODO better error handling
# TODO change all asym errors from _up _down to ,up ,down (so they all match)

# Acceleration space for which we generate a probability distribution.
# A_SPACE = np.linspace(-15e-9, 15e-9, 300)
A_SPACE = np.linspace(-5e-8, 5e-8, 300)

# The order of this is important!
DEFAULT_PRIORS = {
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
    # TODO get attributes as well
    # TODO add a 'keys' style method to list the available stuff
    '''

    def __contains__(self, key):
        return key in self._variables

    def __getitem__(self, key):
        return self._variables[key]

    def _init_variables(self, name, var):
        '''used by group.visit'''

        if isinstance(var, h5py.Dataset):
            self._variables[name] = var[:]

    def __init__(self, group):

        self._variables = {}
        group.visititems(self._init_variables)

    def compute_async_error(self, varname, quantity):
        '''TODO this is where the async error stuff should go'''
        pass

    def convert_units(self):
        pass


class Observations:
    '''Collection of Datasets, read from a corresponding hdf5 file'''

    def __getitem__(self, key):
        if '/' in key:
            group, key = key.split('/', maxsplit=1)
            return self._datasets[group][key]

        else:
            return self._datasets[key]

    def __init__(self, cluster):

        self._datasets = {}
        self.priors = DEFAULT_PRIORS.copy()

        with resources.path('fitter', 'resources') as datadir:
            with h5py.File(f'{datadir}/{cluster}.hdf5', 'r') as file:

                logging.info(f"Observations read from {datadir}/{cluster}.hdf5")

                for group in (file.keys() - {'priors'}):
                    self._datasets[group] = Dataset(file[group])

                try:
                    # This updates defaults with data while keeping default sort
                    self.priors = {**self.priors, **file['priors'].attrs}

                except KeyError:
                    logging.info("No priors stored in datafile, using defaults")
                    pass


def get_dataset(cluster, key):
    '''get a dataset corresponding to a key

    indices is either a slice or a tuple for making a slice or a nparray mask
        mathcing the size of the data. None will return everything (:)
    '''

    with resources.path('fitter', 'resources') as datadir:
        with h5py.File(f'{datadir}/{cluster}.hdf5', 'r') as file:

            return file[key][:]
