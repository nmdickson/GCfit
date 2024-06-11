import io
import os
import sys
import json
import shutil
import pathlib
import logging
import fnmatch

import h5py
import numpy as np
import astropy.units as u

from .units import angular_width


__all__ = ['core_cluster_list', 'hdf_view',
           'get_std_cluster_name', 'get_cluster_path',
           'bibcode2bibtex', 'doi2bibtex', 'bibcode2cite']


GCFIT_DIR = pathlib.Path(os.getenv('GCFIT_DIR', '~/.GCfit')).expanduser()


# --------------------------------------------------------------------------
# Data source retrieval
# --------------------------------------------------------------------------


def _validate_bibcodes(bibcodes):
    '''Simple check that a list of bibcodes at least *looks* valid.'''
    import re
    bibcode_pattern = r"\b\d{4}[a-zA-Z][0-9a-zA-Z&.]{14}\b"
    return all(re.fullmatch(bibcode_pattern, bib) for bib in bibcodes)


def doi2bibtex(doi):
    '''Request the bibtex entry of this `doi` from crossref.'''
    import requests

    headers = {'accept': 'application/x-bibtex'}
    url = f'http://dx.doi.org/{doi}'

    return requests.get(url=url, headers=headers)


def bibcode2bibtex(bibcode):
    '''Request the bibtex entry of this `bibcode` from the ADS.

    Requires the `ads` package and a NASA ADS API-key saved to a file called
    `~/.ads/dev_key` or as an environment variable named `ADS_DEV_KEY`.
    '''
    import ads

    if isinstance(bibcode, (str, bytes)):
        bibcode = [bibcode]

    if not _validate_bibcodes(bibcode):
        mssg = f"{bibcode} is not a valid bibcode"
        raise ValueError(mssg)

    query = ads.ExportQuery(bibcode, format='bibtex')

    try:
        return query.execute().strip().split('\n\n')

    except ads.exceptions.APIResponseError as err:
        mssg = "Failed to retrieve citation from ads"
        raise RuntimeError(mssg) from err


def bibcode2cite(bibcode, strict=True):
    r'''Request this `bibcode` from ADS and attempt to parse a \cite from it.

    Requires the `ads` package and a NASA ADS API-key saved to a file called
    `~/.ads/dev_key` or as an environment variable named `ADS_DEV_KEY`.

    For lack of a more sophisticated parser, simply grabs the start of a full
    format which begins with the "\cite" style (aastex) and uses that.

    If the given bibcode is not valid, won't even attempt to query ads, and
    will simply error or return the bibcode (depending on `strict`).
    '''
    import ads

    if isinstance(bibcode, (str, bytes)):
        bibcode = [bibcode]

    if not _validate_bibcodes(bibcode):
        if strict:
            mssg = f"{bibcode} contains an invalid bibcode"
            raise ValueError(mssg)
        else:
            return '; '.join(bibcode)

    query = ads.ExportQuery(bibcode, format='aastex')

    cites = []

    try:
        for entry in query.execute().strip().split('\n'):
            entry = entry.replace('\\', '')

            # Grab citation from initial square brackets of aastex format
            entry = entry[entry.index('[') + 1: entry.index(']')]

            # Add a space between the authors and the year
            entry = f"{entry[:entry.index('(')]} {entry[entry.index('('):]}"

            cites.append(entry)

    except ads.exceptions.APIResponseError as err:
        mssg = "Failed to retrieve citation from ads"
        raise RuntimeError(mssg) from err

    return '; '.join(cites)


# --------------------------------------------------------------------------
# Data file utilities
# --------------------------------------------------------------------------

def _open_resources():
    '''Return the path to the `resources` directory for this package'''
    from importlib import resources
    return resources.files('gcfit') / 'resources'


def core_cluster_list():
    '''Return a list of cluster names, useable by `gcfit.Observations`'''

    with _open_resources() as datadir:
        return [f.stem for f in pathlib.Path(datadir).glob('[!TEST]*.hdf')]


# TODO could switch this up to use ClusterFile maybe?
def hdf_view(cluster, attrs=False, spacing='normal', *, outfile="stdout"):
    '''Write out a clean listing of a clusters contents.

    For a given cluster, crawl the corresponding hdf data file and write (or
    return) a pretty-printed string listing of the files contents. In the
    least, the file's groups and datasets, but optionally attributes and dataset
    metadata.

    Parameters
    ----------
    cluster : str
        Cluster common name, as used in cluster's hdf data file.

    attrs : bool, optional
        If False (default) write only base dataset names, else include cluster
        and dataset attributes, as well as dataset shape and datatypes, and
        the 'initials' root dataset.

    spacing : {'normal', 'tight', 'loose'}
        Adjust amount of spacing between each data grouping.
        Defaults to 'normal'.

    outfile : {'stdout', 'return', file-like}
        Output location of listing. Either written directly to stdout (default),
        returned as string (return) or written to supplied IO object.

    Returns
    -------
    None or string
        If `outfile` is 'return', the full output as string, else None.
    '''
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
    with _open_resources() as datadir:
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
    'm79': 'ngc1904',
    'm68': 'ngc4590',
    'm53': 'ngc5024',
    'm3': 'ngc5272',
    'm5': 'ngc5904',
    'm80': 'ngc6093',
    'm4': 'ngc6121',
    'm107': 'ngc6171',
    'm13': 'ngc6205',
    'm12': 'ngc6218',
    'm10': 'ngc6254',
    'm62': 'ngc6266',
    'm19': 'ngc6273',
    'm92': 'ngc6341',
    'm9': 'ngc6333',
    'm14': 'ngc6402',
    'm28': 'ngc6626',
    'm69': 'ngc6637',
    'm22': 'ngc6656',
    'm70': 'ngc6681',
    'm54': 'ngc6715',
    'm56': 'ngc6779',
    'm55': 'ngc6809',
    'm71': 'ngc6838',
    'm75': 'ngc6864',
    'm72': 'ngc6981',
    'm15': 'ngc7078',
    'm2': 'ngc7089',
    'm30': 'ngc7099',
    'ter11': 'ter05',
    'hp3': 'ter02',
    'hp4': 'ter04',
    'ton1': 'ngc6380',
    'hp5': 'ter06',
    'djo3': 'ngc6540',
    'djorg3': 'ngc6540',
    'ic1276': 'pal7',
    'pal9': 'ngc6717',
    'omegacen': 'ngc5139',
    'ωcen': 'ngc5139',
    'avdb': 'pal14'
}


def get_std_cluster_name(name):
    '''Convert a given cluster name to a standardized version.

    Convert a given cluster name, in a variety of formats, to a standardized
    format, which can be used to find and access cluster data files.

    A standardized format exists for three different common cluster catalogues:

    - New General Catalogue: NGC####
        NGC, no spaces, 4 numbers, left padded by 0
    - Palomar, PAL##
        PAL, no spaces, 2 numbers, left padded by 0
    - Terzan, TER##
        TER, no spaces, 2 numbers, left padded by 0

    Before being standardized, all whitespace, dashes and underscores are
    stripped from the name, and all letters are lowercased.
    A check for common cluster aliases is also made against `COMMON_NAMES`.

    If not a part of these three catalogues, no standardization will be
    effected, and the exact same `name` will be returned.

    Parameters
    ----------
    name : str
        Input cluster name to be standardized.

    Returns
    -------
    str
        Standardized cluster name.
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

    # extract all numbers in name
    digits = ''.join(filter(str.isdigit, name))

    # pad zeroes
    if name[:3] == 'ngc':
        name = f'NGC{int(digits):04}'

    elif name[:3] == 'pal':
        name = f'PAL{int(digits):02}'

    elif name[:3] == 'ter':
        name = f'TER{int(digits):02}'

    else:
        mssg = f"Cluster Catalogue {name[:3]} not recognized, leaving untouched"
        logging.warning(mssg)

    return name


def get_cluster_path(name, standardize_name=True, restrict_to=None):
    '''Based on a cluster name, return path to the corresponding cluster file.

    Given a cluster name, search the relevant directories for the corresponding
    cluster data file and return the path to said file.

    Cluster files can exist as 'core' clusters, standardized and constant within
    the package itself, or as 'local' clusters, created by the user and stored
    within the `GCFIT_DIR` directory. Searches for cluster files will begin
    with local files, and then core files if no matching local files are found.
    If a specific location is desired, the search can be restricted to one
    or the other.

    Parameters
    ----------
    name : str
        Cluster name.

    standardize_name : bool, optional
        If True (default) pass the cluster name through `get_std_cluster_name`
        before searching for the matching file. Searches for core clusters will
        always be standardized, no matter this parameter.

    restrict_to : {None, 'local', 'core'}
        Where to restrict file searches to. By default (None) searches local
        first, then core if no matching local files are found.

    Returns
    -------
    pathlib.Path
        Path to corresponding cluster HDF data file.
    '''

    if restrict_to not in (None, 'local', 'core'):
        mssg = "Invalid `restrict_to`, must be one of (None, 'local', 'core') "
        raise ValueError(mssg)

    # ----------------------------------------------------------------------
    # Get file paths of prospective local and core cluster files
    # ----------------------------------------------------------------------

    # TODO maybe this shouldn't be made if it doesnt exists, its just clutter
    local_dir = pathlib.Path(GCFIT_DIR, 'clusters')
    local_dir.mkdir(parents=True, exist_ok=True)

    # handle if name was cluster name or filename (with/without suffix)
    filename = pathlib.Path(name).with_suffix('.hdf')

    # get the standardized name, and if desired use it primarily
    std_name = get_std_cluster_name(filename.stem)
    std_filename = pathlib.Path(std_name).with_suffix('.hdf')

    if standardize_name:
        filename = std_filename

    # Get full paths to each file
    local_file = pathlib.Path(local_dir, filename)

    with _open_resources() as core_dir:
        core_file = pathlib.Path(core_dir, std_filename)

    # ----------------------------------------------------------------------
    # Check which files exists and return based on restrict_to
    # ----------------------------------------------------------------------

    if restrict_to == 'local':
        if local_file.exists():
            return local_file
        else:
            mssg = f"No local cluster file matching {filename} in {local_dir}"
            raise FileNotFoundError(mssg)

    elif restrict_to == 'core':
        if core_file.exists():
            return core_file
        else:
            mssg = f"No core cluster file matching {std_filename}"
            raise FileNotFoundError(mssg)

    elif restrict_to is None:

        if local_file.exists():
            return local_file
        elif core_file.exists():
            return core_file
        else:
            mssg = f"No local or core cluster file matching {std_filename}"
            raise FileNotFoundError(mssg)


# --------------------------------------------------------------------------
# Data file creation and editing
# --------------------------------------------------------------------------


# TODO add a close functionality or something because hangs file in write mode
class ClusterFile:
    '''Create, edit and manage hdf cluster data files.

    Contains all necessary methods for interacting with cluster data files,
    the backend of all `gcfit.Observations` classes. Includes functions for
    creating, reading, writing, deleting and testing all data and metadata
    associated with a cluster.

    This class works solely in the "local" cluster regime, and creates or
    searches for all files within the `GCFIT_DIR` directory.

    The observational data must be stored in a strict format in order to be
    read by `gcfit.Observations` and used in all relevant likelihood functions.
    As such, an extensive "testing" regime exists here, which acts to
    ensure all data stored in these cluster files follows the standards defined
    in the `specification.json` resource file.

    This class is intimately tied to the `Dataset` class, which defines each
    individual set of data, which must be created seperately and added to this
    class using one of the corresponding methods.

    Parameters
    ----------
    name : str
        Cluster name, used in file creation or search.

    standardize_name : bool, optional
        If True (default) pass the cluster name through `get_std_cluster_name`
        before searching for or creating the matching file.

    force_new : bool, optional
        If True, will force the creation of a new blank file, potentially
        overwriting any existing file with the same name. Defaults to False.

    Attributes
    ----------
    live_datasets : dict
        Dictionary of current `Dataset`s set to be written to file.

    live_metadata : dict
        Dictionary of current metadata key-values set to be written to file.

    live_initials : dict
        Dictionary of current "initials" key-values set to be written to file.

    '''

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------

    # TODO
    def _new(self, path):
        '''if a completely new file, create and fill in some basic structure'''
        hdf = h5py.File(path, 'w')

        hdf.create_dataset('initials', data=h5py.Empty("f"))

        return hdf

    def __init__(self, name, standardize_name=True, force_new=False):

        local_dir = pathlib.Path(GCFIT_DIR, 'clusters')
        local_dir.mkdir(parents=True, exist_ok=True)

        # handle if name was cluster name or filename (with/without suffix)
        filename = pathlib.Path(name).with_suffix('.hdf')

        # get the standardized name, and if desired use it primarily
        std_name = get_std_cluster_name(filename.stem)

        if standardize_name:
            filename = pathlib.Path(std_name).with_suffix('.hdf')

        local_file = pathlib.Path(local_dir, filename)

        # If desired, force the creation of a new blank local file
        if force_new:
            logging.info('Forcing creation of new local cluster')

            self.file = self._new(local_file)

        # check if its an already created local file
        elif local_file.exists():
            logging.info(f'{name} is already a local cluster, opening to edit')

            # TODO maybe we should only open as r until _write_* is called
            self.file = h5py.File(local_file, 'r+')

        # else Check if this is a "core" file and make a copy locally
        elif std_name in core_cluster_list():
            logging.info(f'{name} is a core cluster, making a new local copy')

            # TODO Add a flag that this is a local file? or only n Observations?

            with _open_resources() as core_dir:
                core_file = pathlib.Path(core_dir, std_name).with_suffix('.hdf')
                shutil.copyfile(core_file, local_file)

                self.file = h5py.File(local_file, 'r+')

        # else make a new local file
        else:
            logging.info(f'{name} does not yet exist, making new local cluster')

            self.file = self._new(local_file)

        self.live_datasets = {}
        self.live_metadata = {}
        self.live_initials = {}

    # ----------------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------------

    def get_dataset(self, key, reset=False):
        '''Return the `Dataset` corresponding to this `key` in the file.'''

        # Check if this dataset is already live
        if key in self.live_datasets:

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

            unit = variable.attrs.get('unit', None)

            mdata_keys = variable.attrs.keys() - {'unit'}
            metadata = {k: variable.attrs[k] for k in mdata_keys}

            data = variable[:] if variable.shape is not None else np.array([])

            dset.add_variable(varname, data, unit, metadata)

        for key, val in self.file[key].attrs.items():
            dset.add_metadata(key, val)

        return dset

    def delete_dataset(self, key):
        '''Set the dataset corresponding to `key` to be deleted upon save.'''

        if key in self.file:
            self.live_datasets[key] = 'DELETE'

        else:
            mssg = f"Can't delete {key}, does not exist in file"
            raise KeyError(mssg)

    def _write_datasets(self, confirm=False):
        '''actually write it out to file, after we've tested all changes.'''

        # TODO writing empty datasets will actually be tricky here I think
        #   cause we don't want size-0 arrays (like we read them as) but `Empty`

        check = True

        for name, dset in self.live_datasets.items():

            if confirm:

                if dset == 'DELETE':
                    mssg = f'Delete Dataset({name})? [y]/n/a/q/? '

                else:
                    var, Nvar = set(dset.variables), len(dset.variables)
                    mssg = f'Save {dset} ({Nvar} variables {var})? [y]/n/a/q/? '

                while True:

                    inp = input(mssg).lower().strip()

                    if inp in ('', 'y'):
                        check = True

                    elif inp == 'n':
                        check = False

                    elif inp == 'a':
                        confirm = False
                        check = True

                    elif inp == 'q':
                        check = False
                        confirm = False

                    elif inp == '?':
                        sys.stdout.write(f'\x1b[2K\r')
                        sys.stdout.write(
                            'y - write this dataset\n'
                            'n - do not write this dataset\n'
                            'a - write this and all remaining datasets\n'
                            'q - quit; do not write this '
                            'or any remaining datasets\n'
                        )
                        continue

                    else:
                        sys.stdout.write(f'\x1b[2K\rUnrecognized input {inp}')
                        continue

                    break

            if check:

                # Try to delete existing dset, in case this is an edited dset
                try:
                    del self.file[name]
                except KeyError:
                    if dset == 'DELETE':
                        logging.warning(f"Can't delete {name}, does not exist")

                # if this isn't just a delete operation, write the new stuff
                if dset != 'DELETE':

                    grp = self.file.create_group(name=name)

                    for key, val in dset.metadata.items():
                        grp.attrs[key] = val

                    for varname, variable in dset.variables.items():
                        var = grp.create_dataset(varname, data=variable['data'])

                        if variable['unit'] is not None:
                            var.attrs['unit'] = variable['unit']

                        for k, v in variable['metadata'].items():
                            var.attrs[k] = v

        # Reset live datasets
        self.live_datasets = {}

    def add_dataset(self, dataset):
        '''Add `Dataset` to the live datasets, to be written to file on save.'''
        self.live_datasets[dataset.name] = dataset

    def unadd_dataset(self, key, pop=True):
        '''Remove the dataset corresponding to `key` from the live datasets.'''
        try:
            dset = self.live_datasets[key]
            del self.live_datasets[key]
        except KeyError:
            raise KeyError(f"Can't unadd {key}, does not exist")

        if pop:
            return dset

    # ----------------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------------

    def get_metadata(self, key, reset=False):
        '''Return the root metadata corresponding to this `key` in the file.'''

        if key in self.live_metadata:
            if reset:
                del self.live_metadata[key]
                pass
            else:
                return self.live_metadata[key]

        return self.file.attrs[key]

    def delete_metadata(self, key):
        '''Set the metadata corresponding to `key` to be deleted upon save.'''

        if key in self.file.attrs:
            self.live_metadata[key] = 'DELETE'

        else:
            mssg = f"Can't delete {key}, does not exist in file"
            raise KeyError(mssg)

    def _write_metadata(self, confirm=False):
        '''actually write it out to file, after we've tested all changes.'''

        check = True

        for key, value in self.live_metadata.items():

            if confirm:

                if value == 'DELETE':
                    mssg = f'Delete metadata {key}? [y]/n/a/q/? '

                else:
                    mssg = f'Save metadata ({key}: {value})? [y]/n/a/q/? '

                while True:
                    inp = input(mssg).lower().strip()

                    if inp in ('', 'y'):
                        check = True

                    elif inp == 'n':
                        check = False

                    elif inp == 'a':
                        confirm = False
                        check = True

                    elif inp == 'q':
                        check = False
                        confirm = False

                    elif inp == '?':
                        sys.stdout.write(f'\x1b[2K\r')
                        sys.stdout.write(
                            'y - write this metadata\n'
                            'n - do not write this metadata\n'
                            'a - write this and all remaining metadata\n'
                            'q - quit; do not write this '
                            'or any remaining metadata\n'
                        )
                        continue

                    else:
                        sys.stdout.write(f'\x1b[2K\rUnrecognized input {inp}')
                        continue

                    break

            if check:

                if value == 'DELETE':
                    try:
                        del self.file.attrs[key]
                    except KeyError:
                        logging.warning(f"Can't delete {key}, does not exist")

                else:
                    self.file.attrs[key] = value

        # reset live metadata
        self.live_metadata = {}

    def add_metadata(self, key, value):
        '''Add this entry to the live metadata to be written to file on save.'''

        # TODO still need to figure out to store metadata units
        self.live_metadata[key] = value

    def unadd_metadata(self, key, pop=True):
        '''Remove the entry corresponding to `key` from the live metadata.'''

        try:
            value = self.live_metadata[key]
            del self.live_metadata[key]
        except KeyError:
            raise KeyError(f"Can't unadd {key}, does not exist")

        if pop:
            return value

    # ----------------------------------------------------------------------
    # Initials
    # ----------------------------------------------------------------------

    def get_initials(self, key, reset=False):
        '''Return the initial value corresponding to this `key` in the file.'''
        if key in self.live_initials:
            if reset:
                del self.live_initials[key]
                pass
            else:
                return self.live_initials[key]

        return self.file['intials'].attrs[key]

    # TODO forgot to add the delete_initials method? (18)

    def _write_initials(self, confirm=False):
        '''actually write it out to file, after we've tested all changes.'''

        check = True

        for key, value in self.live_initials.items():

            if confirm:

                if value == 'DELETE':
                    mssg = f'Delete initial value for {key}? [y]/n/a/q/? '

                else:
                    mssg = f'Save intial value ({key}: {value})? [y]/n/a/q/? '

                while True:
                    inp = input(mssg).lower().strip()

                    if inp in ('', 'y'):
                        check = True

                    elif inp == 'n':
                        check = False

                    elif inp == 'a':
                        confirm = False
                        check = True

                    elif inp == 'q':
                        check = False
                        confirm = False

                    elif inp == '?':
                        sys.stdout.write(f'\x1b[2K\r')
                        sys.stdout.write(
                            'y - write this initial value\n'
                            'n - do not write this initial value\n'
                            'a - write this and all remaining initial values\n'
                            'q - quit; do not write this '
                            'or any remaining initial values\n'
                        )
                        continue

                    else:
                        sys.stdout.write(f'\x1b[2K\rUnrecognized input {inp}')
                        continue

                    break

            if check:

                if value == 'DELETE':
                    try:
                        del self.file['initials'].attrs[key]
                    except KeyError:
                        logging.warning(f"Can't delete {key}, does not exist")

                else:
                    self.file['initials'].attrs[key] = value

        # reset live initials
        self.live_initials = {}

    def add_initials(self, key, value):
        '''Add this entry to the live initials to be written to file on save.'''
        self.live_initials[key] = value

    def unadd_initials(self, key, pop=True):
        '''Remove the entry corresponding to `key` from the live initials.'''
        try:
            value = self.live_initials[key]
            del self.live_initials[key]
        except KeyError:
            raise KeyError(f"Can't unadd {key}, does not exist")

        if pop:
            return value

    # ----------------------------------------------------------------------
    # Finalization
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Base checks
    # ----------------------------------------------------------------------

    def _check_contains(self, dataset, key):
        '''Check that `key` exists within `dataset`'''

        if key in dataset.variables:
            return True
        else:
            self._inv_mssg.append(f'Required variable {key} not in {dataset}')
            return False

    def _check_contains_any(self, dataset, key_choices):
        '''Check that atleast one of `key_choices` exists within `dataset`.'''

        if not key_choices:
            raise ValueError("key_choices must have at least one element")

        if any([key in dataset.variables for key in key_choices]):
            return True
        else:
            self._inv_mssg.append(f'Not one of required variable choices '
                                  f'({key_choices}) in {dataset}')
            return False

    def _check_for_error(self, dataset, key):
        '''Check that "Δ{key}" or "Δ{key},up;down" exists within `dataset`.'''

        variables = dataset.variables

        if (f'Δ{key},up' in variables) and (f'Δ{key},down' in variables):
            return True
        elif f'Δ{key}' in variables:
            return True
        else:
            self._inv_mssg.append(f'Required uncertainties on variable {key} '
                                  f'not in {dataset}')
            return False

    def _check_for_units(self, dataset, key, kind=None, *, none_ok=False):
        '''Check that `key` within `dataset` has valid units.'''

        variable = dataset.variables[key]

        try:

            # TODO I don't think this is a good way to handle none logic
            if variable['unit'] is None and not none_ok:
                self._inv_mssg.append(f"Variable {key}'s unit cannot be None")
                return False

            unit = u.Unit(variable['unit'])

        except KeyError:
            self._inv_mssg.append(f"Variable {key} has no attached unit")
            return False

        except ValueError:
            self._inv_mssg.append(f"Variable {key}'s unit is invalid")
            return False

        if kind is not None:

            with u.set_enabled_equivalencies(angular_width(1 * u.kpc)):

                # if kind is a physical type, convert it to a default unit
                try:
                    match_unit = u.get_physical_type(kind)._unit
                except ValueError:
                    match_unit = u.Unit(kind)

                if not unit.is_equivalent(match_unit):
                    self._inv_mssg.append(f"Variable {key}'s unit does not "
                                          f"match required type {kind}")
                    return False

        return True

    def _check_for_size(self, dataset, key, match):
        '''Check that `key` within `dataset` is same size as `match`.'''

        variable = dataset.variables[key]

        try:
            size = int(match)
            is_num = True

        except ValueError:
            is_num = False

            try:
                sibling = dataset.variables[match]
                size = sibling['data'].size
            except KeyError:
                self._inv_mssg.append(f'Required variable {match} not in '
                                      f'{dataset}, cannot check size of {key}')
                return False

        if variable['data'].size != size:
            self._inv_mssg.append(f'Variable {key} does not match size of '
                                  f'{size}' if is_num else f'{match} ({size})')
            return False

        return True

    def _check_for_field(self, dataset, key):
        '''Check that a valid mass function field variable exists in dataset.'''

        import string
        # TODO Should also require the field is valid (i.e. Field can work)?

        fields = dataset.variables[key]['metadata'].keys() - {'unit'}

        if (fields & set(string.ascii_letters)):
            return True

        else:
            self._inv_mssg.append(f"Mass function {dataset} has no fields")
            return False

    def _check_for_all(self, dataset, varname, requirements):
        '''Parse this variable's requirements and pass it out to the checks.'''

        valid = True

        for req in requirements:

            if isinstance(req, (str, bytes)):
                req = [req]

            req_type, *req_args = req

            if req_type == "unit":
                valid &= self._check_for_units(dataset, varname, *req_args)

            elif req_type == "error":
                valid &= self._check_for_error(dataset, varname, *req_args)

            elif req_type == "fields":
                valid &= self._check_for_field(dataset, varname, *req_args)

            elif req_type == "size":
                valid &= self._check_for_size(dataset, varname, *req_args)

            else:
                valid &= self._check_contains(dataset, req_type)

        return valid

    # ----------------------------------------------------------------------
    # Specification checks
    # ----------------------------------------------------------------------

    def _check_required(self, dataset, varname, requirements):
        '''Check the requirements of `varname`.'''

        valid = True

        if exists := self._check_contains(dataset, varname):

            valid &= self._check_for_all(dataset, varname, requirements)

        valid &= exists

        return valid

    def _check_optional(self, dataset, varname, requirements):
        '''Check the optional requirements of `varname`.'''

        valid = True

        # If this var isn't in the dataset, that's fine & dont check it's spec
        if varname in dataset.variables:

            valid &= self._check_for_all(dataset, varname, requirements)

        return valid

    def _check_choice(self, dataset, choices):
        '''Check the choice requirements of `varname`.'''

        valid = True

        if exists := self._check_contains_any(dataset, choices):

            for varname, requirements in choices.items():

                # If this choice is present, it *has* to pass requirements
                if varname in dataset.variables:

                    valid &= self._check_for_all(dataset, varname, requirements)

        valid &= exists

        return valid

    # ----------------------------------------------------------------------
    # Component tests
    # ----------------------------------------------------------------------

    def _test_dataset(self, key, dataset):
        '''Make all checks of this dataset.'''

        with _open_resources() as datadir:
            with open(f'{datadir}/specification.json') as ofile:
                fullspec = json.load(ofile)

        for spec_pattern in fullspec.keys() - {'INITIALS', 'METADATA'}:
            if fnmatch.fnmatch(key, spec_pattern):
                spec = fullspec[spec_pattern]
                break

        else:
            self._inv_mssg.append(f"'{key}' does not match any specification")
            return False

        valid = True

        if reqd := spec.get('requires'):

            for varname, varspec in reqd.items():

                valid &= self._check_required(dataset, varname, varspec)

        if opti := spec.get('optional'):

            for varname, varspec in opti.items():

                valid &= self._check_optional(dataset, varname, varspec)

        if chce := spec.get('choice'):

            valid &= self._check_choice(dataset, chce)

        return valid

    def _test_metadata(self, metadata):
        '''Make all checks of this metadata.'''

        with _open_resources() as datadir:
            with open(f'{datadir}/specification.json') as ofile:
                mdata_spec = json.load(ofile)['METADATA']

        valid = True

        if reqd := mdata_spec.get('requires'):

            for item in reqd:

                if item not in metadata:

                    mssg = f"Required metadata item {item} not found"
                    self._inv_mssg.append(mssg)

                    valid &= False

        # if opti := mdata_spec.get('optional'):

        return valid

    def _test_initials(self, initials):
        '''Make all checks of these initials.'''

        with _open_resources() as datadir:
            with open(f'{datadir}/specification.json') as ofile:
                init_spec = json.load(ofile)['INITIALS']

        valid = True

        # if reqd := init_spec.get('requires'):

        if opti := init_spec.get('optional'):

            if extra := initials.keys() - set(opti):

                mssg = f"Extraneous initial values found {extra}"
                self._inv_mssg.append(mssg)

                valid &= False

        # check all are valid numbers
        for key, value in initials.items():
            try:
                float(value)

            except (TypeError, ValueError):
                dt = type(value)
                self._inv_mssg.append(f"Invalid dtype for initial {key} ({dt})")

                valid &= False

        return valid

    # ----------------------------------------------------------------------
    # Full tests and file writing
    # ----------------------------------------------------------------------

    def test(self):
        '''Test all live datasets, metadata and initials for compliance.

        Based on the `specification.json` standards file, passes all live
        datasets, metadata and initials through a bevy of tests and checks, to
        ensure they meet all requirements.

        If *any* tests fail, the reasons for failure are logged in the
        `_inv_mssg` list, and this function returns `False`.
        '''
        # TODO test should also include non-live objects (maybe optional?)
        # TODO testing seems brittle/broken for any "DELETE"

        # test datasets for required variables

        self._inv_mssg = []

        valid = True

        for key, dataset in self.live_datasets.items():
            valid &= self._test_dataset(key, dataset)

        # TODO only testing *live* metadata means it will fail on editted files
        valid &= self._test_metadata(self.live_metadata)

        valid &= self._test_initials(self.live_initials)

        return valid

    def save(self, force=False, confirm=False):
        '''Test all live data and, if valid, write all live data to file.

        Passes the live data through the `test` method, and if valid, writes
        and saves all data to the cluster data hdf file. If some tests are
        invalid, will log a warning containing the specific reasons for failure.

        Parameters
        ----------
        force : bool, optional
            If True, force the save even if some tests fail. Defaults to False.

        confirm : bool, optional
            If True, prompts and waits for confirmation from user for each
            write operation. Defaults to False.
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

        # TODO quitting the confirm prompts seems to remove live stuff?
        self._write_datasets(confirm=confirm)
        self._write_metadata(confirm=confirm)
        self._write_initials(confirm=confirm)


# TODO *really* don't like the potential conflict with this and `gcfit.Dataset`
class Dataset:
    '''Read and manage representation of a complete dataset.

    Representing the actual data to be stored in a `ClusterFile`, this class
    contains all relevant methods for reading data (variables) from a variety
    of sources, in a robust manner. Intimately tied to the `ClusterFile` class,
    all datasets should be added to a cluster file using the
    `ClusterFile.add_dataset` method.

    Dataset's must be initialized with a key/name, and have all data added in
    the form of "variables", either manually through the `add_variable` method,
    or from a raw data source using the `read_data` method.

    Parameters
    ----------
    key : str
        Dataset name. Will be used as the group path in the cluster hdf file.

    Attributes
    ----------
    variables : dict
        Dictionary of all currently added variables. All key's represent
        variable names, while each entry must be a dictionary of "data", "unit"
        and "metadata".

    metadata : dict
        Dictionary of any dataset-level metadata key-value pairs.
    '''

    def __repr__(self):
        return f"Dataset('{self.name}')"

    def __str__(self):
        return str(self.name)

    def __init__(self, key):

        self.name = key
        self.metadata = {}
        self.variables = {}

    def add_variable(self, varname, data, unit, metadata, error_base=None):
        '''Manually add a variable to this dataset.

        Populate an entry in this datasets `variables` dictionary with the
        given information.

        Parameters
        ----------
        varname : str
            Name of variable. To be used as key in the `variables` dictionary.

        data : numpy.ndarray
            Array of data representing this variable.

        unit : str or astropy.Unit
            Unit of this variable. Either astropy unit or corresponding valid
            unit name.

        metadata : dict
            Dictionary of all variable-level metadata.

        error_base : str or None
            If this variable represents the error or uncertainties on another
            variable, providing the name of said variable will attempt to
            format this data into the correct representation of an error.
        '''

        # TODO it's probably fine if we don't require units here
        #   some actually don't require them, and we test later anyways

        # If this is an error (given error_base), try to parse into a valid name
        if error_base:

            if varname.lower().endswith(('up', '+', 'plus', 'high')):
                varname = f'Δ{error_base},up'

            elif varname.lower().endswith(('down', '-', 'minus', 'low')):
                varname = f'Δ{error_base},down'

            else:
                varname = f'Δ{error_base}'

        # Store all data in the variables dict
        self.variables[varname] = {
            "data": data,
            "unit": str(unit) if unit is not None else None,
            "metadata": metadata
        }

    def add_metadata(self, key, value):
        '''Manually add a dataset-level metadata entry.'''
        self.metadata[key] = value

    def read_data(self, src, **kwargs):
        '''Based on a raw data source, parse any variables into this dataset.

        Given an arbitrary raw `src`, parse the input and send it to any
        number of submethods which will attempt to extract any desired variables
        from the input, and add them to this dataset.

        Currently this method supports reading data from dictionaries,
        hdf files, dataframes, delimited text files and raw strings.
        '''
        import pandas as pd

        def _from_dict(src, keys=None):
            '''Read data from a dictionary of variables

            Parameters
            ----------
            src: dict
                Dictionary of multiple "name: variable" entries

            keys: list of str, optional
                List of which keys within `src` to use. Defaults to using all
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
            '''Read data from a `h5py.File` root group

            Parameters
            ----------
            src : pathlib.Path
                Path to a valid hdf5 file

            key : list of str, optional
                List of all hdf datasets to read from this group into variables.
                Defaults to all children of this group

            grp : str, optional
                Name of the desired hdf-group within the file. Defaults to the
                root group, at "/"
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

        def _from_dataframe(df, keys=None, filter_=None, empty_ok=False,
                            units=None, metadata=None, names=None, errors=None,
                            **kwargs):
            '''Read data from a `pandas.Dataframe`

            Parameters
            ----------
            df : pandas.Dataframe
                Dataframe to read data from

            keys : list of str, optional
                List of columns to read from this dataframe. Dataframe must have
                named columns. By default, uses all

            filter_, filter : str or list of str, optional
                Filtering constraints to filter the given dataframe on. Must be
                valid string queries for the dataframe `query` method.

            units : dict, optional
                Dictionary of unit strings for each `key`. Defaults to None for
                all variables

            names : dict, optional
                A mapping of `keys` to custom variable names. Allows for
                changing of column names to match required standard variable
                names

            errors : dict, optional
                A mapping of any `key` to a different `key`, indicating that the
                first key is an error on the second. This changes what is passed
                to the `error_base` arg of `add_variable`
            '''
            # TODO this is often the final method, but some of these kwargs
            #   conflict with kwargs that would be passed to, say, pd.read_table

            if units is None:
                units = {}

            if metadata is None:
                metadata = {}

            if names is None:
                names = {}

            if errors is None:
                errors = {}

            filter_ = filter_ or kwargs.get('filter', None)

            try:
                expr = " & ".join(filter_)
                df = df.query(expr)
            except TypeError:
                pass

            if df.empty:
                if not empty_ok:
                    mssg = f'Filtered ({expr})' if filter_ else ''
                    raise ValueError(mssg + " Dataframe is empty")

            keys = keys or df.columns

            for colname in keys:
                data = df[colname].to_numpy()

                if data.dtype.kind == 'O' and isinstance(data[0], (str, bytes)):
                    # Need to use bytes array for strings, hdf5 can't handle "U"
                    data = data.astype('S')

                varname = names.get(colname, colname)

                # TODO still don't know how best to get units from the data file
                unit = units.get(colname, None)

                # TODO how does this mesh with a given `names`
                err = errors.get(colname, None)

                self.add_variable(varname, data, unit, metadata, err)

        def _from_delimfile(src, delim=None, comment='#', **kwargs):
            '''Read data from a delimited plain-text file

            Parameters
            ----------
            src : str, pathlib.Path or file-like object
                File, either as path to or IO-object, from which to read data.

            delim : str, optional
                Column delimiter to use.

            comment : str, optional
                Character used to indicate the beginning of a comment line.
                Defaults to '#'.

            **kwargs
                All extra keyword arguments are pass to `_from_dataframe`.
            '''

            # read file into dataframe
            df = pd.read_table(src, sep=delim, comment=comment)

            # pass to _from_dataframe
            _from_dataframe(df, **kwargs)

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
