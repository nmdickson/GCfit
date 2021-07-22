import io
import os
import shutil
import pathlib
import logging
import fnmatch
import warnings
from importlib import resources

import h5py
import fitter
import astropy.units as u


__all__ = ['cluster_list', 'hdf_view', 'get_std_cluster_name',
           'bibcode2bibtex', 'doi2bibtex']


GCFIT_DIR = pathlib.Path(os.getenv('GCFIT_DIR', '~/.GCfit')).expanduser()


# --------------------------------------------------------------------------
# Data file utilities
# --------------------------------------------------------------------------


def cluster_list():
    '''Return a list of cluster names, useable by `fitter.Observations`'''
    with resources.path('fitter', 'resources') as datadir:
        return [f.stem for f in pathlib.Path(datadir).glob('[!TEST]*.hdf')]


# TODO could switch this up to use ClusterFile maybe?
def hdf_view(cluster, attrs=False, spacing='normal', *, outfile="stdout"):
    '''Write out a clean listing of a clusters contents

    For a given cluster, crawl the corresponding hdf data file and write (or
    return) a pretty-printed string listing of the files contents. In the
    least, the file's groups and datasets, but optionally attributes and dataset
    metadata.

    parameters
    ----------
    cluster : string
        Cluster common name, as used in cluster's hdf data file

    attrs : bool, optional
        If False (default) write only base dataset names, else include cluster
        and dataset attributes, as well as dataset shape and datatypes, and
        the 'initials' root dataset.

    spacing : {'normal', 'tight', 'loose'}
        Adjust amount of spacing between each data grouping. Default to 'normal'

    outfile : {'stdout', 'return', file-like}
        Output location of listing. Either written directly to stdout (default),
        returned as string (return) or written to supplied IO object.

    Returns
    -------
    None or string
        if `outfile` is 'return', the full output as string, else None

    '''
    import sys
    import h5py

    # ----------------------------------------------------------------------
    # Crawler to generate each line of output
    # ----------------------------------------------------------------------

    def _crawler(root_group):

        def _writer(key, obj):

            tabs = '    ' * key.count('/')

            key = key.split('/')[-1]

            if isinstance(obj, h5py.Group):
                newline = '' if spacing == 'tight' else '\n'
            else:
                newline = '\n' if spacing == 'loose' else ''

            front = f'{newline}{tabs}'

            outstr = f"{front}{type(obj).__name__}: {key}\n"

            if attrs:
                # TODO on mass functions this shouldn't print the field coords
                for k, v in obj.attrs.items():
                    outstr += f"{tabs}    |- {k}: {v}\n"

                if isinstance(obj, h5py.Dataset) and key != 'initials':
                    outstr += f"{tabs}    |- shape: {obj.shape}\n"
                    outstr += f"{tabs}    |- dtype: {obj.dtype}\n"

            res.append(outstr)

        res = []
        root_group.visititems(_writer)

        return ''.join(res)

    # ----------------------------------------------------------------------
    # Open the file and run the crawler over all its items
    # ----------------------------------------------------------------------

    # TODO use get_std_cluster_name here
    with resources.path('fitter', 'resources') as datadir:
        with h5py.File(f'{datadir}/{cluster}.hdf', 'r') as file:

            out = f"{f' {cluster} ':=^40}\n\n"

            if attrs:
                out += '\n'.join(f"|- {k}: {v}" for k, v in file.attrs.items())
                out += '\n\n'

            out += _crawler(file)

    # ----------------------------------------------------------------------
    # Write the ouput
    # ----------------------------------------------------------------------

    if outfile == 'return':
        return out

    elif outfile == 'stdout':
        sys.stdout.write(out)

    else:
        with open(outfile, 'a') as output:
            output.write(out)


COMMON_NAMES = {
    '47tuc': 'ngc0104',
    '47tucanae': 'ngc0104',
    'm62': 'ngc6266',
}


def get_std_cluster_name(name):
    '''From a given cluster name, convert it to a standard name, which
    can be used to find the cluster data file.
    i.e. 47Tuc, 47_Tuc, NGC104,NGC_104, NGC 104 to NGC0104

    need a naming standard for our files:
        - if in New General Catalogue: NGC####
            NGC, no space, 4 numbers, left padded by 0
        - if palomar, PAL##
            PAL, no space, 2 numbers, left padded by 0

        I don't think we'll be using anything else, I think it'll be mostly
        NGC, so this is safe to go with for now.
    '''
    import re

    if name == 'TEST':
        return name

    # remove whitespace, dashes and underscores
    name = re.sub(r'[\s\-\_]+', '', name)

    # lowercase
    name = name.lower()

    # common names to NGC, Pal
    if name in COMMON_NAMES:
        name = COMMON_NAMES[name]

    # pad zeroes
    if name[:3] == 'ngc':
        name = f'NGC{int(name[3:]):04}'

    elif name[:3] == 'pal':
        name = f'Pal{int(name[3:]):02}'

    else:
        mssg = f"Cluster Catalogue {name[:3]} not recognized, leaving untouched"
        warnings.warn(mssg)

    return name


# --------------------------------------------------------------------------
# Data file creation and editing
# --------------------------------------------------------------------------


class ClusterFile:

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------

    def _new(self, path):
        '''if a completely new file, create and fill in some basic structure'''
        hdf = h5py.File(path, 'w')

        hdf.create_dataset('initials', data=h5py.Empty("f"))

        return hdf

    def __init__(self, name):
        # TODO should make some sort of check that it matches std_cluster_names

        local_dir = pathlib.Path(GCFIT_DIR, 'clusters')
        local_dir.mkdir(parents=True, exist_ok=True)

        local_file = pathlib.Path(local_dir, name).with_suffix('.hdf')

        # check if its an already created local file
        if local_file.exists():
            logging.info(f'{name} is already a local cluster, opening to edit')

            self.file = h5py.File(local_file, 'r+')

        # else Check if this is a "core" file and make a copy locally
        elif name in fitter.util.cluster_list():
            logging.info(f'{name} is a core cluster, making a new local copy')

            # TODO Add a flag that this is a local file? or only n Observations?
            with resources.path('fitter', 'resources') as core_dir:
                core_file = pathlib.Path(core_dir, name).with_suffix('.hdf')
                shutil.copyfile(core_file, local_file)

                self.file = h5py.File(local_file, 'r+')

        # else make a new local file
        else:
            logging.info(f'{name} does not yet exist, making new local cluster')

            self.file = self._new(local_file)

        self.live_datasets = {}
        self.live_metadata = {}

    # ----------------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------------

    # TODO delete functionality, from live and from file

    def get_dataset(self, key, reset=False):

        # Check if this dataset is already live
        if key in self.live_datasets():

            # Delete the live one and continue on to reread from the file
            if reset:
                del self.live_datasets[key]
                pass

            # return the live one
            else:
                return self.live_datasets[key]

        # Read this dataset from the file
        dset = Dataset(key)

        for varname, variable in self.file[key].items():

            unit = variable.attrs['unit']

            mdata_keys = variable.attrs.keys() - {'unit'}
            metadata = {k: variable.attrs[k] for k in mdata_keys}

            dset.add_variable(varname, variable[:], unit, metadata)

        for key, val in self.file[key].attrs.items():
            dset.add_metadata(key, val)

        return dset

    def _write_datasets(self):
        '''actually write it out to file, after we've tested all changes
        '''

        for name, dset in self.live_datasets.items():

            grp = self.file.create_group(name=dset.name)

            for key, val in dset.metadata.items():
                grp.attrs[key] = val

            # TODO if editing this will have to account for (del) existing dsets
            for varname, variable in dset.variables.items():
                var = grp.create_dataset(varname, data=variable['data'])
                var.attrs['unit'] = variable['unit']

                for k, v in variable['metadata'].items():
                    var.attrs[k] = v

        # Reset live datasets
        self.live_datasets = {}

    def add_dataset(self, dataset):
        '''get a new or edited dataset and store it in this object until
        tested and written
        '''
        self.live_datasets[dataset.name] = dataset

    # ----------------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------------

    def get_metadata(self, key, reset=False):
        if key in self.live_metadata:
            if reset:
                del self.live_metadata[key]
                pass
            else:
                return self.live_metadata[key]

        return self.file.attrs[key]

    def _write_metadata(self):
        for key, value in self.live_metadata.items():
            self.file.attrs[key] = value

        # reset live metadata
        self.live_metadata = {}

    def add_metadata(self, key, value):
        '''cluster-level metadata'''
        # TODO still need to figure out to store metadata units
        self.live_metadata[key] = value

    # ----------------------------------------------------------------------
    # Finalization
    # ----------------------------------------------------------------------

    # TODO these checks check for having the right data, but not that it's valid
    #   i.e. matching shapes, correct dtype, correct unit class, etc

    def _check_contains(self, dataset, key):

        if key in dataset.variables:
            return True
        else:
            self._inv_mssg.append(f'Required variable {key} not in {dataset}')

    def _check_contains_choice(self, dataset, key_choices):

        if any([key in dataset for key in key_choices]):
            return True
        else:
            self._inv_mssg.append(f'Not one of required variable choices '
                                  f' ({key_choices}) in {dataset}')
            return False

    def _check_for_error(self, dataset, key):
        variables = dataset.variables

        if (f'Δ{key},up' in variables) and (f'Δ{key},down' in variables):
            return True
        elif f'Δ{key}' in variables:
            return True
        else:
            self._inv_mssg.append(f'Required uncertainties on variable {key} '
                                  f'not in {dataset}')
            return False

    def _check_for_units(self, dataset, key, none_ok=False):
        variable = dataset.variables[key]
        try:

            unit = u.Unit(variable['unit'])

            if unit is None and not none_ok:
                self._inv_mssg.append(f"Variable {key}'s unit cannot be None")
                return False
            else:
                return True

        except KeyError:
            self._inv_mssg.append(f"Variable {key} has no attached unit")
            return False

        except ValueError:
            self._inv_mssg.append(f"Variable {key}'s unit is invalid")
            return False

    def _check_for_field(self, dataset):
        import string
        # TODO this one is definitely not right right now, mf hasnt been tried
        # TODO Should also require the field is valid?

        self._check_contains(dataset, 'fields')

        fields = dataset.variables['fields'].attrs

        if (fields.keys() & set(string.ascii_letters)):
            return True

        else:
            self._inv_mssg.append(f"Mass function {dataset} has no fields")
            return False

    def _test_dataset(self, key, dataset):
        '''Strongly inspired by the compliance unittests, could maybe be a way
        to reduce repetition between the two in the future? (If that unittest
        is still even necessary of course)'''

        valid = True

        # Pulsars

        if fnmatch.fnmatch(key, '*pulsar*'):
            valid &= self._check_contains(dataset, 'r')
            valid &= self._check_for_units(dataset, 'r')

            pulsar_fields = ('P', 'Pb')
            valid &= self._check_contains_choice(dataset, pulsar_fields)

            if 'P' in dataset:

                valid &= self._check_contains(dataset, 'Pdot')
                valid &= self._check_for_error(dataset, 'Pdot')
                valid &= self._check_for_units(dataset, 'Pdot')

            elif 'Pb' in dataset:

                valid &= self._check_contains(dataset, 'Pbdot')
                valid &= self._check_for_error(dataset, 'Pbdot')
                valid &= self._check_for_units(dataset, 'Pbdot')

        # LOS Velocity Dispersion

        elif fnmatch.fnmatch(key, '*velocity_dispersion*'):
            valid &= self._check_contains(dataset, 'r')
            valid &= self._check_for_units(dataset, 'r')

            valid &= self._check_contains(dataset, 'σ')
            valid &= self._check_for_error(dataset, 'σ')
            valid &= self._check_for_units(dataset, 'σ')

        # Number Density

        elif fnmatch.fnmatch(key, '*number_density*'):
            valid &= self._check_contains(dataset, 'r')
            valid &= self._check_for_units(dataset, 'r')

            valid &= self._check_contains(dataset, 'Σ')
            valid &= self._check_for_error(dataset, 'Σ')
            valid &= self._check_for_units(dataset, 'Σ')

        # Proper Motion Dispersion

        elif fnmatch.fnmatch(key, '*proper_motion*'):
            valid &= self._check_contains(dataset, 'r')
            valid &= self._check_for_units(dataset, 'r')

            # make sure that atleast one usable PM is there
            pm_fields = ('PM_tot', 'PM_ratio', 'PM_R', 'PM_T')
            valid &= self._check_contains_choice(dataset, pm_fields)

            # Check for corresponding errors/units
            if 'PM_tot' in dataset:
                valid &= self._check_for_error(dataset, 'PM_tot')
                valid &= self._check_for_units(dataset, 'PM_tot')

            if 'PM_ratio' in dataset:
                valid &= self._check_for_error(dataset, 'PM_ratio')
                valid &= self._check_for_units(dataset, 'PM_R', none_ok=True)

            if 'PM_R' in dataset:
                valid &= self._check_for_error(dataset, 'PM_R')
                valid &= self._check_for_units(dataset, 'PM_R')

            if 'PM_T' in dataset:
                valid &= self._check_for_error(dataset, 'PM_T')
                valid &= self._check_for_units(dataset, 'PM_T')

        # Mass Function

        elif fnmatch.fnmatch(key, '*mass_function*'):
            valid &= self._check_contains(dataset, 'N')
            valid &= self._check_contains(dataset, 'ΔN')

            valid &= self._check_contains(dataset, 'r1')
            valid &= self._check_for_units(dataset, 'r1')
            valid &= self._check_contains(dataset, 'r2')
            valid &= self._check_for_units(dataset, 'r2')

            valid &= self._check_contains(dataset, 'm1')
            valid &= self._check_for_units(dataset, 'm1')
            valid &= self._check_contains(dataset, 'm2')
            valid &= self._check_for_units(dataset, 'm2')

            valid &= self._check_for_field(dataset)
            valid &= self._check_contains_mdata(dataset, 'field_unit')

        return valid

    def test(self):
        '''Something along lines of the Observation complicance test in "tests"
        test that all the "live" data is correct and valid before we write it
        '''

        # TODO how will users handle initials?, feels separate from the rest
        # test initials (make sure there is no extra initial parameters)
        # extra = obs.initials.keys() - data.DEFAULT_INITIALS.keys()
        # self.assertEqual(len(extra), 0)

        # test datasets for required variables

        self._inv_mssg = []

        valid = True

        for key, dataset in self.live_datasets.items():
            valid &= self._test_dataset(key, dataset)

    def save(self, force=False):
        '''test all the new stuff and then actually write it out, if it passes
        '''

        valid = self.test()

        if not valid:
            logging.warning("Live data is not entirely valid: "
                            + "; ".join(self._inv_mssg))

            if force:
                logging.warning("Forcing write, despite invalid data")
                pass
            else:
                logging.warning("Abandoning save without writing")
                return

        logging.info("Writing live data to file")

        self._write_datasets()
        self._write_metadata()


# TODO *really* don't like the potential conflict with this and `fitter.Dataset`
class Dataset:

    # Methods for constructing a Dataset, to be placed into the clusterfile

    # Read from a ClusterFile (for editing)
    # Created brand new (to be filled with variables and metadata)
    # Created based on a raw data file
    #   This has to be somewhat robust, as we will want to use these raw files
    #   for most data additions, and might need multiple files and specific
    #   columns / rows for it.
    #
    #   Might actually be best to move that logic into some functions and just
    #   init a new Dataset here, without existing variables, and add
    #   them using the `add_variable` in those functions, or in ClusterFile if
    #   has existing variables/metadata

    def __repr__(self):
        return f"Dataset('{self.name}')"

    def __str__(self):
        return str(self.name)

    def __init__(self, key):
        # both new and from the ClusterFile?
        # maybe only give name, and rely on ClusterFile to populate everything,
        # using add_* methods

        self.name = key
        self.metadata = {}
        self.variables = {}

    def add_variable(self, varname, data, unit, metadata):

        self.variables[varname] = {
            "data": data,
            "unit": unit,
            "metadata": metadata
        }

    def add_metadata(self, key, value):
        self.metadata[key] = value

    def read_data(self, src, **kwargs):
        '''
        based on src, send this off to different smaller more specific functions

        get raw data from src, put it into this dataset
        '''
        import pandas as pd

        def _from_dict(src, keys=None):
            '''
            src: dict of multiple "varname: variable" entries
            keys: which of the keys in src to use, defaults to all of them
            '''
            keys = keys or src.keys()

            for varname in keys:
                variable = src[varname]

                if varname == 'metadata':
                    for mdata_key, mdata in variable.items():
                        self.add_metadata(varname, variable)

                else:
                    self.add_variable(varname, **variable)

        def _from_hdffile(src, keys=None, grp='/', get_metadata=True):
            '''
            src : path to an hdf5 file
            key : name of hdf-dataset in (if None, does all datasets in)
            grp : name of hdf-group, defaults to root group
            '''

            with h5py.File(src, 'r') as hdf:
                root = hdf[grp]

                keys = keys or root.keys()

                for varname in keys:
                    dset = root[varname]

                    unit = dset.attrs['unit']
                    self.add_variable(varname, dset[:], unit, dict(dset.attrs))

                if get_metadata:
                    for k, v in root.attrs.items():
                        self.add_metadata(k, v)

        def _from_dataframe(df, keys=None, filter_=None,
                            units=None, metadata=None, **kwargs):
            '''
            called by other methods which split up or read files and stuff and
            then pass it here as a pandas dataframe

            df: dataframe
            keys : name of columns to store (if None, uses all)
            filter_, filter: constraints we should filter the dataframe on, like
                "where column x is equal to RV" and stuff like that
                list of valid queries, for use in df.query
            units : dict of unit strings for each `key`. If None, all are None
            metadata : is just a dict of metadata to pass on to the self, idk

            '''

            if units is None:
                units = {}

            if metadata is None:
                metadata = {}

            filter_ = filter_ or kwargs.get('filter', None)

            try:
                expr = " & ".join(filter_)
                df = df.query(expr)
            except TypeError:
                pass

            keys = keys or df.columns

            for varname in keys:
                data = df[varname]

                # TODO still don't know how best to get units from the data file
                unit = units.get(varname, None)

                self.add_variable(varname, data, unit, metadata)

        def _from_delimfile(src, delim=None, names=None, comment='#', **kwargs):

            # read file into dataframe
            df = pd.read_table(src, sep=delim, names=names, comment=comment)

            # pass to _from_dataframe
            _from_dataframe(df, **kwargs)

        # If not given a self, assume `self` is the name for a new one

        if not isinstance(self, self):
            self = self(self)

        # Parse src, sent to specific function

        if isinstance(src, dict):
            _from_dict(src, **kwargs)

        elif isinstance(src, pd.DataFrame):
            _from_dataframe(src, **kwargs)

        elif isinstance(src, pathlib.Path) or isinstance(src, (str, bytes)):

            # TODO some of the errors here might not be nice, ie missing files:

            # Check if this seems like a path to a file which exists
            if (path := pathlib.Path(src).expanduser()).exists():

                if path.suffix.lower() in ('hdf', 'hdf5'):
                    _from_hdffile(path, **kwargs)

                else:
                    _from_delimfile(path, **kwargs)

            # else assume its a str of data, put into IO and pass to delimfile
            else:
                _from_delimfile(io.StringIO(src), **kwargs)

        else:
            raise ValueError("Invalid src")


# --------------------------------------------------------------------------
# Data source retrieval
# --------------------------------------------------------------------------


def doi2bibtex(doi):
    '''Request the bibtex entry of this `doi` from crossref'''
    import requests

    headers = {'accept': 'application/x-bibtex'}
    url = f'http://dx.doi.org/{doi}'

    return requests.get(url=url, headers=headers)


def bibcode2bibtex(bibcode):
    '''Request the bibtex entry of this `bibcode` from the ADS

    Requires the `ads` package and a NASA ADS API-key saved to a file called
    `~/.ads/dev_key` or as an environment variable named `ADS_DEV_KEY`
    '''
    import ads

    query = ads.ExportQuery(bibcode, format='bibtex')

    return query.execute()
