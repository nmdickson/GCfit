import pathlib
from importlib import resources


__all__ = ['cluster_list', 'hdf_view']


def cluster_list():
    with resources.path('fitter', 'resources') as datadir:
        return [f.stem for f in pathlib.Path(datadir).glob('[!TEST]*.hdf5')]


def hdf_view(cluster, attrs=False, spacing='normal', *, outfile="stdout"):
    '''print out the contents of a given cluster hdf5 file in a pretty way'''
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

    with resources.path('fitter', 'resources') as datadir:
        with h5py.File(f'{datadir}/{cluster}.hdf5', 'r') as file:

            out = f"{f' {cluster} ':=^40}\n\n"

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
