import warnings
import pathlib
from importlib import resources


__all__ = ['cluster_list', 'hdf_view', 'get_std_cluster_name',
           'bibcode2bibtex', 'doi2bibtex']


def cluster_list():
    '''Return a list of cluster names, useable by `fitter.Observations`'''
    with resources.path('fitter', 'resources') as datadir:
        return [f.stem for f in pathlib.Path(datadir).glob('[!TEST]*.hdf')]


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
