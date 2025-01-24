from .. import Observations
from ..probabilities import priors
from ..core.main import MCMCFittingState, NestedFittingState
from .models import CIModelVisualizer, ModelVisualizer
from .models import EvolvedVisualizer, CIEvolvedVisualizer
from .models import ModelCollection

import sys
import pathlib
import logging
import warnings
import itertools
import contextlib

import h5py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr
import matplotlib.offsetbox as mpl_obx

from mpl_toolkits.axes_grid1 import make_axes_locatable


__all__ = ['RunCollection', 'MCMCRun', 'NestedRun']


_label_math_mapping = {
    # Model Base Parameters
    'W0': r'\hat{\phi}_0',
    'M': r'M',
    'rh': r'r_{\mathrm{h}}',
    'ra': r'\log\left(\hat{r}_\mathrm{a}\right)',
    'g': r'g',
    'delta': r'\delta',
    's2': r's^2',
    'F': r'F',
    'a1': r'\alpha_1',
    'a2': r'\alpha_2',
    'a3': r'\alpha_3',
    'BHret': r'\mathrm{BH}_{ret}',
    'd': r'd',
    # Evolved Model Parameters
    'M0': r'M_{0}',
    'rh0': r'r_{\mathrm{h},0}',
    # Flexible BH Parameters  # TODO need symbols for these
    'kick_slope': r'\mathrm{kick\ slope}',
    'kick_scale': r'\mathrm{kick\ scale}',
    'IFMR_slope': r'\mathrm{IFMR\ slope}',
    'IFMR_slope1': r'\mathrm{IFMR\ slope 1}',
    'IFMR_slope2': r'\mathrm{IFMR\ slope 2}',
    'IFMR_slope3': r'\mathrm{IFMR\ slope 3}',
    'IFMR_scale': r'\mathrm{IFMR\ scale}',
    'IFMR_scale1': r'\mathrm{IFMR\ scale 1}',
    'IFMR_scale2': r'\mathrm{IFMR\ scale 2}',
    'IFMR_scale3': r'\mathrm{IFMR\ scale 3}',
    # Cluster Metadata
    'FeH': r'[\mathrm{Fe}/\mathrm{H}]',
    'Ndot': r'\dot{N}',
    'RA': r'\mathrm{RA}',
    'DEC': r'\mathrm{DEC}',
    'chi2': r'\chi^{2}',
    # Derived Model Quantities
    'BH_mass': r'\mathrm{M}_{\mathrm{BH}}',
    'BH_num': r'\mathrm{N}_{\mathrm{BH}}',
    'f_rem': r'f_{\mathrm{remn}}',
    'f_BH': r'f_{\mathrm{BH}}',
    'spitzer_chi': r'\chi_{\mathrm{Spitzer}}',
    'trh': r't_{\mathrm{r_h}}',
    'N_relax': r'N_{\mathrm{relax}}',
    'r0': r'r_{0}',
    'ra_model': r'r_{\mathrm{a}}',
    'rt': r'r_{\mathrm{t}}',
    'rv': r'r_{\mathrm{v}}',
    'rhp': r'r_{\mathrm{hp}}',
    'mmean': r'\bar{m}',
}


_label_unit_mapping = {
    'M': r'10^6\ M_\odot',
    'M0': r'10^6\ M_\odot',
    'rh': r'\mathrm{pc}',
    'rh0': r'\mathrm{pc}',
    's2': r'\mathrm{arcmin^{-4}}',
    'BHret': r'\%',
    'd': r'\mathrm{kpc}',
    'Ndot': r'\dot{N}',
    'RA': r'\deg',
    'DEC': r'\deg',
    'BH_mass': r'M_\odot',
    'f_rem': r'\%',
    'f_BH': r'\%',
    'trh': r'\mathrm{Gyr}',
    'r0': r'\mathrm{pc}',
    'ra_model': r'\mathrm{pc}',
    'rt': r'\mathrm{pc}',
    'rv': r'\mathrm{pc}',
    'rhp': r'\mathrm{pc}',
    'mmean': r'M_\odot',
}


def _get_latex_label(param, with_units=True):
    '''Return param name in latex form, optionally with units.'''

    name = _label_math_mapping.get(param, param)
    unit = _label_unit_mapping.get(param, None)

    if with_units and unit is not None:
        label = rf'${name.strip("$")}\ \left[{unit}\right]$'
    else:
        label = rf'${name.strip("$")}$' if name else name

    return label

# --------------------------------------------------------------------------
# Individual Run Analysis
# --------------------------------------------------------------------------


class _RunAnalysis:
    '''Base class for all visualizers of all run types.'''

    _cmap = plt.rcParams['image.cmap']

    @property
    def cmap(self):
        return plt.colormaps.get_cmap(self._cmap)

    @cmap.setter
    def cmap(self, cm):
        if isinstance(cm, mpl_clr.Colormap) or (cm in plt.colormaps()):
            self._cmap = cm
        elif cm is None:
            self._cmap = plt.rcParams['image.cmap']
        else:
            mssg = f"{cm} is not a registered colormap, see `plt.colormaps`"
            raise ValueError(mssg)

    def _setup_artist(self, fig, ax, *, use_name=True, **sub_kw):
        '''setup a plot (figure and ax) with one single ax'''

        if ax is None:
            if fig is None:
                # no figure or ax provided, make one here
                fig, ax = plt.subplots(**sub_kw)

            else:
                # Figure provided, no ax provided. Try to grab it from the fig
                # if that doens't work, create it
                cur_axes = fig.axes

                if len(cur_axes) > 1:
                    raise ValueError(f"figure {fig} already has too many axes")

                elif len(cur_axes) == 1:
                    ax = cur_axes[0]

                else:
                    ax = fig.add_subplot(**sub_kw)

        else:
            if fig is None:
                # ax is provided, but no figure. Grab it's figure from it
                fig = ax.get_figure()

        if hasattr(self, 'name') and use_name:
            fig.suptitle(self.name)

        return fig, ax

    def _setup_multi_artist(self, fig, shape, *, allow_blank=True,
                            use_name=True, constrained_layout=True,
                            subfig_kw=None, **sub_kw):
        '''Setup a figure with multiple axes, using subplots or subfigures.

        Given a desired shape tuple, returns a figure with the correct
        arrangement of axes. Allows for easily specifying slightly more complex
        arrangements of axes while filling out the figure space, without the
        use of functions like `subplot_mosaic`.

        The requested shape will determine the exact configuration of the
        returned figure, with subfigures being used in cases where the number
        of axes in a given row or column is mismatched, and otherwise only
        subplots being used. All created axes are returned in a flat array,
        as given by `fig.axes`.

        If `shape` is a length 1 tuple, will create N rows. If a length 2
        tuple, will create (Nrows, Ncols) subplots. If either Nrows or Ncols
        is itself a 2-tuple, will create the number of subfigures (either rows
        or columns) specified by the other value and fill each with the number
        of axes specified in the corresponding tuple. The axes will fill the
        entire space provided by each row/column, and not necessarily be
        aligned along subfigures.
        Nrows and Ncols cannot both be tuples at once.

        For example:
        (3,) -> Single column with 3 rows
        (3,2) -> 3 row, 2 column subplot
        (2,(1,3)) -> 2 rows of subfigures, with 1 and 3 columns each
        ((1,3,4), 3) -> 3 columns of subfigures, with 1, 3 and 4 rows each
        ((1,3,4), 4) -> if `allow_blank`, same as above with an empty 4th column

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure
            Given starting figure. If None will create a new figure from
            scratch. If given an existing `Figure` with existing axes, will
            simply check that axes shape matches and return the figure
            untouched If `Figure` is empty (no axes), will create new axes
            within the given figure.

        shape : tuple
            Tuple representing the shape of the subplot grid.

        allow_blank : bool, optional
            If shape requires subfigures and the number of subfigures is larger
            than the corresponding tuple, allow the creation of blank (empty)
            subfigures on the end. Defaults to True.

        use_name : bool, optional
            If True (default) add `self.name` to the figure suptitle.

        constrained_layout : bool, optional
            Passed to `Figure` if a new figure must be created. Defaults to
            True.

        subfig_kw : dict, optional
            Passed to `fig.subfigures`, if required.

        **sub_kw : dict, optional
            Extra arguments passed to all calls to `fig.subplots`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.

        axes : array of matplotlib.axes.AxesSubplot
            Flattened numpy array of all axes in `fig`, as given by `fig.axes`
        '''

        if subfig_kw is None:
            subfig_kw = {}

        def create_axes(base, shape):
            '''create the axes of `shape` on this base (fig). shape as dict'''

            # if either of them is also a tuple, means we want columns or rows
            #   of varying sizes, switch to using subfigures

            # TODO what are the chances stuff like `sharex` works correctly?

            if isinstance(shape['nrows'], tuple):

                subfigs = base.subfigures(ncols=shape['ncols'], nrows=1,
                                          squeeze=False, **subfig_kw)

                for ind, sf in enumerate(subfigs.flatten()):

                    try:
                        nr = shape['nrows'][ind]
                    except IndexError:

                        if allow_blank:
                            continue

                        # TODO this still creates the figure...
                        mssg = (f"Number of row entries {shape['nrows']} must "
                                f"match number of columns ({shape['ncols']})")
                        raise ValueError(mssg)

                    sf.subplots(ncols=1, nrows=nr, **sub_kw)

            elif isinstance(shape['ncols'], tuple):

                subfigs = base.subfigures(nrows=shape['nrows'], ncols=1,
                                          squeeze=False, **subfig_kw)

                for ind, sf in enumerate(subfigs.flatten()):

                    try:
                        nc = shape['ncols'][ind]
                    except IndexError:

                        if allow_blank:
                            continue

                        mssg = (f"Number of col entries {shape['ncols']} must "
                                f"match number of rows ({shape['nrows']})")
                        raise ValueError(mssg)

                    sf.subplots(nrows=1, ncols=nc, **sub_kw)

            # otherwise just make a simple subplots and return that
            else:
                base.subplots(**shape, **sub_kw)

            return base, base.axes

        # ------------------------------------------------------------------
        # Create figure, if necessary
        # ------------------------------------------------------------------

        if fig is None:
            fig = plt.figure(constrained_layout=constrained_layout)

        # ------------------------------------------------------------------
        # If no shape is provided, just return the figure, probably empty
        # ------------------------------------------------------------------

        if shape is None:
            axarr = []

        # ------------------------------------------------------------------
        # Otherwise attempt to first grab this figures axes, or create them
        # ------------------------------------------------------------------

        else:

            # make sure shape is a tuple of atleast 1d, at most 2d

            if not isinstance(shape, tuple):
                shape = tuple(shape)

            if len(shape) == 1:
                shape = (shape, 1)

            elif len(shape) > 2:
                mssg = f"Invalid `shape` for subplots {shape}, must be 2D"
                raise ValueError(mssg)

            # split into dict of nrows, ncols

            shape = dict(zip(("nrows", "ncols"), shape))

            # this fig has axes, check that they match shape
            if axarr := fig.axes:

                if isinstance(shape['nrows'], tuple):
                    Naxes = np.sum(shape['nrows'])
                elif isinstance(shape['ncols'], tuple):
                    Naxes = np.sum(shape['ncols'])
                else:
                    Naxes = shape['nrows'] * shape['ncols']

                if len(axarr) != Naxes:
                    mssg = (f"figure {fig} already contains wrong number of "
                            f"axes ({len(axarr)} != {Naxes})")
                    raise ValueError(mssg)

            else:
                fig, axarr = create_axes(fig, shape)

        # ------------------------------------------------------------------
        # If desired, default to titling the figure based on it's "name"
        # ------------------------------------------------------------------

        if hasattr(self, 'name') and use_name:
            fig.suptitle(self.name)

        # ------------------------------------------------------------------
        # Ensure the axes are always returned in an array
        # ------------------------------------------------------------------

        return fig, np.atleast_1d(axarr)

    def add_residuals(self, ax, y1, y2, e1, e2, clrs=None,
                      res_ax=None, loc='bottom', size='15%', pad=0.1):
        '''Append an extra axis to `ax` for plotting residuals.

        Automatically appends a new axis to the the bottom of the given `ax`,
        and plots the residuals between the two given quantities (and their
        errors) on it, as a percentage.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            An axes instance on which to plot this observational data.

        y1, y2 : np.ndarray
            Arrays of data values to plot the residual between. Residuals
            are of (y2 - y1) / y1.

        e1, e2 : np.ndarray
            Arrays of errors on each datapoint.

        clrs : color, optional
            Colour used for all datapoints, passed to `errorbar` and `scatter`.

        res_ax : matplotlib.axes.Axes, optional
            Optionally provide an already created axis to plot residuals on.
            This is useful for overplotting multiple residuals (i.e. for
            multiple datasets).

        loc : {"left", "right", "bottom", "top"}, optional
            Where the new axes is positioned relative to the main axes.

        size : str or float, optional
            The size of the appended residuals axes, with respect to the
            primary axes.
            See `mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`
            for more information. Defaults to "15%".

        pad : float, optional
            Padding between the axes. Defaults to 0.1.

        Returns
        -------
        matplotlib.axes.Axes
            The created axes instance containing the residuals plot.
        '''

        # make a newres ax if needed
        if res_ax is None:
            divider = make_axes_locatable(ax)
            res_ax = divider.append_axes(loc, size=size, pad=pad, sharex=ax)

        res_ax.grid(visible=True)
        res_ax.set_xscale(ax.get_xscale())
        res_ax.set_ylabel(r"% difference")

        # plot residuals (in percent)
        res = 100 * (y2 - y1) / y1
        res_err = 100 * np.sqrt(e1**2 + e2**2) / y1
        res_ax.errorbar(y1, res, yerr=res_err, fmt='none', ecolor=clrs)
        res_ax.scatter(y1, res, color=clrs,)

        return res_ax


# TODO a way to plot our priors, probably for both vizs
class _SingleRunAnalysis(_RunAnalysis):
    '''Base class for all visualizers of single runs, of all types.

    Parameters
    ----------
    filename : pathlib.Path or str
        Path to the run output HDF5 file.

    observations : gcfit.Observations or None
        The `Observations` instance corresponding to this cluster.
        If None, an educated guess will be made on the source location
        (i.e. the `restrict_to` argument) and the observations will be created
        based on the cluster name stored in the output.

    group : str
        Name of the root group in the HDF5 file. Most likely either "nested"
        or "mcmc", depending on the run type.

    name : str, optional
        Custom name for this run.
    '''

    _mask = None
    results = None

    @property
    def mask(self):
        '''Mask out certain samples, removing them from all analysis.'''
        return self._mask

    def reset_mask(self):
        '''Reset the mask.'''
        self._mask = None

    @mask.setter
    def mask(self, value):

        if value is not None:

            _, ch = self._get_chains(include_fixed=False, apply_mask=False)

            # flatten chain if necessary (to work with MCMC, mask must be flat)
            if ch.ndim > 2:
                ch = ch.reshape((-1, ch.shape[-1]))

            if value.ndim > 1 or value.shape[0] != ch.shape[0]:
                mssg = (f'Invalid mask shape {value.shape}; '
                        f'must have shape {ch[:, 0].shape}')
                raise ValueError(mssg)

        self._mask = value

        # If necessary, recompute results with mask applied
        if self.results is not None:
            self.results = self._get_results()

    def slice_on_param(self, param, lower_lim, upper_lim):
        '''Set `mask` based on the value of a certain parameter.
        Already present masks will be combined. If that's not desired, masks
        should be reset (`self.reset_mask()`) first.
        '''

        labels = self._get_labels()

        if param not in labels:
            mssg = f'Invalid param "{param}". Must be one of {labels}'
            raise ValueError(mssg)

        data = self._get_chains(apply_mask=False)[1][..., labels.index(param)]

        if data.ndim > 1:
            data = data.reshape((-1, data.shape[-1]))

        mask = (data < lower_lim) | (data > upper_lim)

        if self.mask is None:
            self.mask = mask
        else:
            self.mask |= mask

    @property
    def state(self):
        '''Return a flag representing where in the fitting this run reached.'''

        scls = MCMCFittingState if self._gname == 'mcmc' else NestedFittingState

        with self._openfile('metadata') as mdata:
            try:
                return scls(mdata.attrs.get('STATE', -1))

            except ValueError as err:
                mssg = "No valid fitting state was stored. Is this an old run?"
                raise RuntimeError(mssg) from err

    def __str__(self):
        try:
            return f'{self._filename} - Run Results'
        except AttributeError:
            return "Run Results"

    def __init__(self, filename, observations, group, name=None):

        self._filename = filename
        self._gname = group

        with h5py.File(filename, 'r') as file:

            # Check that all necessary groups exist in the given file
            reqd_groups = {group, 'metadata'}

            if missing_groups := (reqd_groups - file.keys()):
                mssg = (f"Output file {filename} is invalid: "
                        f"missing {missing_groups} groups. "
                        "Are you sure this was created by GCfit?")

                raise RuntimeError(mssg)

            # Check what the state of the fit was
            fitting_state = file['metadata'].attrs.get('STATE', -1)

            # works for both mcmc & nested
            if fitting_state < MCMCFittingState.START:
                # assume this is just an old fit, we cannot say what state
                pass

            elif fitting_state < MCMCFittingState.SAMPLING:
                mssg = ("This fit did not begin final sampling; results are "
                        "likely incorrect and some functionality may break.")
                warnings.warn(mssg)

            elif fitting_state < MCMCFittingState.FINAL:
                mssg = ("This fit did not reach the full stopping conditions; "
                        "sampling may not be fully converged.")
                warnings.warn(mssg)

            mdata = file['metadata'].attrs

            # Check if this is an evolved modelling fit or not
            self._evolved = mdata.get('evolved', False)

            # Check if this had extra flexible BH parameters
            self._free_kicks = mdata.get('flexible_natal_kicks', False)
            self._free_IFMR = mdata.get('flexible_IFMR', False)

            # Check for backwards compatibility with old, free BHs
            if mdata.get('flexible_BHs', False):
                self._free_kicks = self._free_IFMR = True

            # Check if this run seems to have used a local cluster data file
            restrict_to = mdata.get('restrict_to', None)

        # Determine and init cluster observations if necessary
        if name is not None:
            self.name = name

        if observations is not None:
            self.obs = observations

        else:
            try:
                with h5py.File(filename, 'r') as file:
                    cluster = file['metadata'].attrs['cluster']

                self.obs = Observations(cluster, restrict_to=restrict_to)

            except KeyError as err:
                mssg = "No cluster name in metadata, must supply observations"
                raise ValueError(mssg) from err

        self._parameters = list(self.obs.initials if not self._evolved
                                else self.obs.ev_initials)

        if self._free_kicks:
            self._parameters += list(self.obs._kick_initials)

        if self._free_IFMR:
            self._parameters += list(self.obs._IFMR_initials)

    @contextlib.contextmanager
    def _openfile(self, group=None, mode='r'):
        file = h5py.File(self._filename, mode)

        try:

            if group is not None:
                yield file[group]

            else:
                yield file

        finally:
            file.close()

    def _get_labels(self, label_fixed=True, math_labels=False):
        '''Retrieve labels for all parameters.'''

        labels = self._parameters.copy()

        if math_labels:
            labels = [_get_latex_label(lbl, with_units=True) for lbl in labels]

        if label_fixed:

            with self._openfile('metadata') as mdata:

                fixed = sorted(
                    ((k, labels.index(k)) for k in mdata['fixed_params'].attrs),
                    key=lambda item: labels.index(item[0])
                )

            for k, i in fixed:
                labels[i] += ' (fixed)'

        return labels

    def _get_model_kwargs(self, note_flexible_BHs=False):
        '''Return the `model_kwargs` metadata (backwards compatible)'''

        def _gather_attrs(key, grp):
            try:
                model_kw[key] = grp[:]  # in case this is a dataset, not a group
            except TypeError:
                model_kw[key] = dict(grp.attrs)

        with self._openfile('metadata') as mdata:
            try:
                model_kw = dict(mdata['model_kwargs'].attrs)
                mdata['model_kwargs'].visititems(_gather_attrs)
            except KeyError:
                model_kw = {}

        # this is not for `Model`, but for some preliminary stuff (`_get_model`)
        if note_flexible_BHs:
            if self._free_kicks:
                model_kw['flexible_natal_kicks'] = True
            if self._free_IFMR:
                model_kw['flexible_IFMR'] = True

        return model_kw


class MCMCRun(_SingleRunAnalysis):
    '''Analysis and visualization of an MCMC cluster fitting run.

    Provides a number of flexible plotting, output and summary methods useful
    for the analysis of both the procedure and results of an MCMC run,
    based on the output file generated by the fitting (`emcee.HDFBackend`).

    Parameters
    ----------
    filename : pathlib.Path or str
        Path to the run output HDF5 file.

    observations : gcfit.Observations or None, optional
        The `Observations` instance corresponding to this cluster.
        If None (default), an educated guess will be made on the source location
        (i.e. the `restrict_to` argument) and the observations will be created
        based on the cluster name stored in the output.

    group : str, optional
        Name of the root group in the HDF5 file. Defaults to "mcmc", the most
        likely name used by `gcfit.MCMC_fit`.

    name : str, optional
        Custom name for this run.
    '''

    def __init__(self, filename, observations=None, group='mcmc', name=None):

        super().__init__(filename, observations, group, name)

        # Ensure the dimensions are initialized correctly
        self.iterations = slice(None)
        self.walkers = slice(None)

    # ----------------------------------------------------------------------
    # Dimensions
    # ----------------------------------------------------------------------

    @property
    def chains(self):
        return self._get_chains(flatten=True)[1]

    def _reduce(self, array, *, only_iterations=False):
        '''apply the necesary iterations and walkers slicing to given array.'''

        # Apply iterations cut

        array = array[self.iterations]

        # Apply walkers cut

        if not only_iterations:

            if callable(self.walkers):

                # Call on array, and ensure the dimensions still work out

                dims = array.shape

                try:
                    array = self.walkers(array, axis=1)
                except TypeError:
                    array = self.walkers(array)

                newdim = array.shape

                if not (len(dims) == len(newdim) and dims[::2] == newdim[::2]):
                    mssg = ("Invalid `walkers`, callables must operate along "
                            "only the 1st axis, or accept an `axis` keyword")
                    raise ValueError(mssg)

            else:
                # assume walkers is a slice or 1-d array
                array = array[:, self.walkers, ...]

        return array

    @property
    def walkers(self):
        '''Slice or mask defining the walkers to use in all analysis.'''
        return self._walkers

    @walkers.setter
    def walkers(self, value):
        '''Walkers must be a slice, callable to be applied to walkers axes or
        1-D boolean mask array.
        '''

        if value is None or value is Ellipsis:
            value = slice(None)

        self._walkers = value

    # cut the ending zeroed iterations, if a run was cut short
    cut_incomplete = True

    @property
    def iterations(self):
        '''Slice or mask defining the iterations to use in all analysis.'''
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        '''Iterations must be a slice. if cut_incomplete is True, will default
        to cutting the final empty iterations from everything.
        '''
        if not isinstance(value, slice):
            mssg = f"`iteration` must be a slice, not {type(value)}"
            raise TypeError(mssg)

        if value.stop is None and self.cut_incomplete:

            with self._openfile(self._gname) as file:
                stop = file.attrs['iteration']

            value = slice(value.start, stop, value.step)

        self._iterations = value

    @property
    def _iteration_domain(self):
        '''Helper array defining the iteration domain, for plotting.'''

        if (start := self.iterations.start) is None:
            start = 0

        if (stop := self.iterations.stop) is None:

            with self._openfile(self._gname) as file:
                stop = file['chain'].shape[0]

        step = self.iterations.step

        return np.arange(start + 1, stop + 1, step)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _get_chains(self, flatten=False):
        '''Get the MCMC chains, properly using the iterations and walkers
        slices, and accounting for fixed params'''

        with self._openfile() as file:

            labels = self._parameters.copy()

            chain = self._reduce(file[self._gname]['chain'])

            # Handle fixed parameters

            fixed = sorted(
                ((k, v, labels.index(k)) for k, v in
                 file['metadata']['fixed_params'].attrs.items()),
                key=lambda item: labels.index(item[0])
            )

            for k, v, i in fixed:
                labels[i] += ' (fixed)'
                chain = np.insert(chain, i, v, axis=-1)

        if flatten:
            chain = chain.reshape((-1, chain.shape[-1]))

        return labels, chain

    def _reconstruct_priors(self):
        '''Based on the stored "specified_priors" get a `Priors` object.'''

        # TODO this seems broken with latest priors storage scheme
        with self._openfile('metadata') as mdata:

            stored_priors = mdata['specified_priors']
            fixed = dict(mdata['fixed_params'].attrs)

        prior_params = {}

        for key in self._parameters:
            try:
                type_ = stored_priors[key].attrs['type']
                args = stored_priors[key]['args']

                if args.dtype.kind == 'S':
                    args = args.astype('U')

                prior_params[key] = (type_, *args)
            except KeyError:
                continue

        prior_kwargs = {'fixed_initials': fixed, 'err_on_fail': False}
        return priors.Priors(prior_params, **prior_kwargs)

    # ----------------------------------------------------------------------
    # Model Visualizers
    # ----------------------------------------------------------------------

    def get_model(self, method='median'):
        '''Return a single `ModelVisualizer` instance corresponding to this run.

        The visualizer is initialized through the `ModelVisualizer.from_chain`
        classmethod, with the chain from this run and the method given here.

        Parameters
        ----------
        method : {'median', 'mean', 'final'}, optional
            The method used to compute a single `theta` set from the chain.
            Defaults to 'median'.

        Returns
        -------
        ModelVisualizer
            The created model visualization object.
        '''

        labels, chain = self._get_chains()

        model_cls = ModelVisualizer if not self.evolved else EvolvedVisualizer

        model_kw = self._get_model_kwargs(note_flexible_BHs=True)

        return model_cls.from_chain(chain, self.obs, method, **model_kw)

    def get_CImodel(self, N=100, Nprocesses=1, load=False):
        '''Return a `CIModelVisualizer` instance corresponding to this run.

        The visualizer is initialized through the `CIModelVisualizer.from_chain`
        classmethod, with the chain from this run and using `N` samples, if
        `load` is False, otherwise will attempt to use the
        `CIModelVisualizer.load` classmethod, assuming a CI model has already
        been created and saved to this same file, under the `model` group.

        Parameters
        ----------
        N : int, optional
            The number of samples to use in computing the confidence intervals.

        Nprocesses : int, optional
            The number of processes to use in a `multiprocessing.Pool` passed
            to the CI model initializer. Defaults to only 1 cpu.

        load : bool, optional
            If True, will attempt to load a CI model, rather than creating a
            new one.

        Returns
        -------
        CIModelVisualizer
            The created model visualization (with confidence intervals) object.
        '''
        import multiprocessing

        viz_cls = CIModelVisualizer if not self.evolved else CIEvolvedVisualizer

        if load:
            return viz_cls.load(self._filename, observations=self.obs)

        else:

            labels, chain = self._get_chains()

            model_kw = self._get_model_kwargs(note_flexible_BHs=True)

            with multiprocessing.Pool(processes=Nprocesses) as pool:
                return viz_cls.from_chain(chain, self.obs, N,
                                          pool=pool, **model_kw)

    # ----------------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------------

    def plot_chains(self, fig=None, params=None):
        '''Plot trace plot of the chains of walkers for all parameters.

        Plots an Nparam-panel figure following the evolution of the different
        walkers throughout all iterations.
        All walkers are shown, per parameter, in different colours.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        params : None or list of str, optional
            The parameters to show on this figure. If None (default) all
            parameters (including fixed params) will be shown.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        # ------------------------------------------------------------------
        # Get the sample chains (weighted and unweighted), paring down the
        # chains to only the desired params, if provided
        # ------------------------------------------------------------------

        labels, chain = self._get_chains()

        # params is None or a list of string labels
        if params is not None:
            prm_inds = [labels.index(p) for p in params]

            labels = params
            chain = chain[..., prm_inds]

        # ------------------------------------------------------------------
        # Setup axes
        # ------------------------------------------------------------------

        if (shape := (len(labels), 1))[0] > 5:
            shape = (
                (int(np.ceil(shape[0] / 2)), int(np.floor(shape[0] / 2))),
                2
            )

        # TODO this sharex still doesn't work between subfigures
        fig, axes = self._setup_multi_artist(fig, shape, sharex=True)

        # ------------------------------------------------------------------
        # Plot each parameter
        # ------------------------------------------------------------------

        for ind, ax in enumerate(axes.flatten()):

            try:
                ax.plot(self._iteration_domain, chain[..., ind])
            except IndexError as err:
                mssg = 'reduced parameters, but no explanatory metadata stored'
                raise RuntimeError(mssg) from err

            ax.set_ylabel(labels[ind])

        for sf in (fig.subfigs or [fig]):
            sf.axes[-1].set_xlabel('Iterations')

        return fig

    def plot_params(self, fig=None, params=None, *,
                    posterior_color='tab:blue', posterior_border=True,
                    ylims=None, truths=None, **kw):
        '''Plot a diagnostic figure of the distributions of parameter samples.

        Plots an Nparam-panel figure showcasing the parameter values of all
        samples, over the iteration domain, as well as a KDE-based smoothed
        posterior distribution for each parameter.

        This class is mostly provided to match the nested sampling version,
        where it is much more instructive. This may be very expensive for large
        MCMC runs, and `plot_chains` may be more instructive.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        params : None or list of str, optional
            The parameters to show on this figure. If None (default) all
            parameters (including fixed params) will be shown.

        posterior_color : color, optional
            The colour of the smoothed posterior distributions.

        posterior_border : bool, optional
            If False, will remove the axis frame around the smooth posterior
            distribution to the right of each panel.

        ylims : list[Nparam] of 2-tuples, optional
            Used to set the upper and lower y axis-limits on each parameter.

        truths : np.ndarray[Nparam] or np.ndarray[Nparam, 3], optional
            Optionally indicate the "true" values as horizontal lines on the
            posterior frames. If [Nparam, 3], the values in each row
            will be taken as the median, lower limit and upper limit.

        **kw : dict
            All other arguments are passed to `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        # ------------------------------------------------------------------
        # Setup plotting kwarg defaults
        # ------------------------------------------------------------------

        color = mpl_clr.to_rgb(posterior_color)
        facecolor = color + (0.33, )

        kw.setdefault('marker', '.')

        # ------------------------------------------------------------------
        # Get the sample chains (weighted and unweighted), paring down the
        # chains to only the desired params, if provided
        # ------------------------------------------------------------------

        labels, chain = self._get_chains()

        # params is None or a list of string labels
        if params is not None:
            prm_inds = [labels.index(p) for p in params]

            labels = params
            chain = chain[..., prm_inds]

        # ------------------------------------------------------------------
        # Setup the truth values and confidence intervals
        # ------------------------------------------------------------------

        if truths is not None and truths.ndim == 2:
            # Assume confidence bounds rather than single truth value

            truth_ci = truths[:, 1:]
            truths = truths[:, 0]

        else:
            truth_ci = None

        # ------------------------------------------------------------------
        # Setup axes
        # ------------------------------------------------------------------

        if ylims is None:
            ylims = [(None, None)] * len(labels)

        elif len(ylims) != len(labels):
            mssg = "`ylims` must match number of params"
            raise ValueError(mssg)

        if (shape := (len(labels), 1))[0] > 5:
            shape = (int(np.ceil(shape[0] / 2)), 2)

        fig, axes = self._setup_multi_artist(fig, shape, sharex=True)

        axes = axes.reshape(shape)

        for ax in axes[-1]:
            ax.set_xlabel(r'$-\ln(X)$')

        # ------------------------------------------------------------------
        # Plot each parameter
        # ------------------------------------------------------------------

        domain = np.tile(self._iteration_domain, (1024, 1)).T

        for ind, ax in enumerate(axes.flatten()):

            # --------------------------------------------------------------
            # Get the relevant samples.
            # If necessary, remove any unneeded axes
            # --------------------------------------------------------------

            try:
                lbl, prm = labels[ind], chain[..., ind]
            except IndexError:
                # If theres an odd number of (>5) params need to delete last one
                # TODO preferably this would also resize this column of plots
                ax.remove()
                continue

            # --------------------------------------------------------------
            # Divide the ax to accomodate the posterior plot on the right
            # --------------------------------------------------------------

            divider = make_axes_locatable(ax)
            post_ax = divider.append_axes('right', size="25%", pad=0, sharey=ax)

            post_ax.set_xticks([])

            # --------------------------------------------------------------
            # Plot the samples with respect to ln(X)
            # --------------------------------------------------------------

            # TODO the y tick values have disappeared should be on the last axis
            ax.scatter(domain, prm, cmap=self.cmap, **kw)

            ax.set_ylabel(lbl)
            ax.set_xlim(left=0)

            # --------------------------------------------------------------
            # Plot the posterior distribution (accounting for weights)
            # --------------------------------------------------------------

            post_kw = {
                'chain': prm.flatten(),
                'flipped': True,
                'truth': truths if truths is None else truths[ind],
                'truth_ci': truth_ci if truth_ci is None else truth_ci[ind],
                'color': color,
                'fc': facecolor
            }

            try:
                self.plot_posterior(lbl, fig=fig, ax=post_ax, **post_kw)
            except ValueError:
                post_ax.axhline(np.median(prm), color=color)

            if not posterior_border:
                post_ax.axis('off')

            # TODO maybe put ticks on right side as well?
            for tk in post_ax.get_yticklabels():
                tk.set_visible(False)

            ax.set_ylim(ylims[ind])

        return fig

    def plot_indiv(self, fig=None):
        '''Plot the evolution of each individual likelihood component over time.

        Plots a multipanel figure tracing the value of the log-likelihood of
        each likelihood component (i.e. based on `self.obs.valid_likelihoods`,
        not each dataset type) individually.
        All walkers are shown, per component, in different colours.

        Individual likelihoods must have been stored as "blobs" during the fit.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        with self._openfile(self._gname) as file:
            try:
                probs = file['blobs']
            except KeyError as err:
                mssg = f"No individial likelihoods stored as blobs"
                raise KeyError(mssg) from err

            fig, axes = self._setup_multi_artist(fig, (len(probs.dtype), ),
                                                 sharex=True)

            for ind, ax in enumerate(axes.flatten()):

                label = probs.dtype.names[ind]

                indiv = self._reduce(probs[:][label])

                ax.plot(self._iteration_domain, indiv)

                ax.set_title(label)

        axes[-1].set_xlabel('Iterations')

        return fig

    def plot_marginals(self, fig=None, **corner_kw):
        '''Plot a "corner plot" showcasing the relationships between parameters.

        Plots a Nparam-Nparam lower-triangular "corner" marginal plot showing
        the projections of all sampled parameter values, using the `corner.py`
        package.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        **corner_kw : dict
            All other arguments are passed to `corner.corner`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''
        import corner

        fig, ax = self._setup_multi_artist(fig, shape=None,
                                           constrained_layout=False)

        labels = self._get_labels(math_labels=True, label_fixed=False)
        _, chain = self._get_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        # ugly
        ranges = [1. if 'fixed' not in lbl
                  else (chain[0, i] - 1, chain[0, i] + 1)
                  for i, lbl in enumerate(labels)]

        corner_kw.setdefault('plot_datapoints', False)
        corner_kw.setdefault('labelpad', 0.25)

        fig = corner.corner(chain, labels=labels, fig=fig,
                            range=ranges, **corner_kw)

        fig.subplots_adjust(left=0.05, bottom=0.06)

        return fig

    def plot_posterior(self, param, fig=None, ax=None, chain=None,
                       flipped=True, truth=None, truth_ci=None,
                       *args, **kwargs):
        '''Plot a smoothed posterior distribution of a single parameter.

        Plots a gaussian-KDE smoothed posterior probability distribution of
        a given parameter.
        Designed mainly to be used within the `plot_params` method, but can
        be used on its own.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this posterior. Should be a
            part of the given `fig`.

        chain : np.ndarray, optional
            Optionally supply a (flattened) array of samples to create the
            posterior from. By default, will load the full chain using
            `self._get_chains`.

        flipped : bool, optional
            If True (default) the posterior will be flipped on it's side,
            attached to the left-axis.

        truth : float, optional
            Optionally indicate the "true" value as horizontal lines on the
            posterior.

        truth_ci : 2-tuple of float, optional
            Optionally shade between the lower and upper limits of the "truth"
            values, using `plt.axhspan`.

        *args, **kwargs
            All other arguments are passed to the `ax.fill_between` function.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''
        from scipy.stats import gaussian_kde

        fig, ax = self._setup_artist(fig, ax)

        labels = self._get_labels()

        if param not in labels:
            mssg = f'Invalid param "{param}". Must be one of {labels}'
            raise ValueError(mssg)

        if chain is None:

            prm_ind = labels.index(param)
            chain = self._get_chains(flatten=True)[1][..., prm_ind]

        try:
            kde = gaussian_kde(chain)
        except np.linalg.LinAlgError as err:
            mssg = f"Cannot compute kde of {param}: {err}"
            raise ValueError(mssg)

        domain = np.linspace(chain.min(), chain.max(), 500)

        if flipped:

            ax.fill_betweenx(domain, 0, kde(domain), *args, **kwargs)

            if truth is not None:
                ax.axhline(truth, c='tab:red')

                if truth_ci is not None:
                    ax.axhspan(*truth_ci, color='tab:red', alpha=0.33)

            ax.set_xlim(left=0)

        else:

            ax.fill_between(domain, 0, kde(domain), *args, **kwargs)

            if truth is not None:
                ax.axvline(truth, c='tab:red')

                if truth_ci is not None:
                    ax.axvspan(*truth_ci, color='tab:red', alpha=0.33)

            ax.set_ylim(bottom=0)

        return fig

    def plot_acceptance(self, fig=None, ax=None):
        '''Plot the sampler acceptance rate over time.

        Plots the "acceptance rate" of the MCMC sampler as a function of
        iterations.
        All walkers are shown, in different colours.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this rate. Should be a
            part of the given `fig`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        with self._openfile() as file:
            try:
                acc = self._reduce(file['statistics']['acceptance_rate'])
            except KeyError as err:
                mssg = f"No acceptance rate stored"
                raise KeyError(mssg) from err

        ax.plot(self._iteration_domain, acc)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Acceptance Rate')

        return fig

    def plot_probability(self, fig=None, ax=None):
        '''Plot the posterior probability over time.

        Plots the total (sum of all components) logged posterior probability
        of the MCMC sampler as a function of iterations.
        All walkers are shown, in different colours.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this probability. Should be a
            part of the given `fig`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        with self._openfile(self._gname) as file:
            prob = self._reduce(file['log_prob'])

        ax.plot(self._iteration_domain, prob)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Total Log Posterior Probability')

        return fig

    # ----------------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------------

    def print_summary(self, out=None, content='all'):
        '''Write a short summary of the run results and metadata.

        Write out (to a file or stdout) a short summary of the final
        median and 1σ parameter values, as well as some metadata surrounding
        the fitting run setup, such as fixed parameters.

        Parameters
        ----------
        out : None or file-like object, str, or pathlib.Path, optional
            The file to write out the summary to. If None (default) will be
            printed to stdout.

        content : {'all', 'results', 'metadata'}
            Which parts of the summary to write. If "results", will print only
            the parameter values. If "metadata", will print only the run
            metadata. If "all" (default), prints both.
        '''
        # TODO add more 2nd level results, like comments on BH masses, etc

        if out is None:
            out = sys.stdout

        mssg = f'{self}'
        mssg += f'\n{"=" * len(mssg)}\n'

        # RESULTS

        # organize this more like it is in cum_mass plots
        if content == 'all' or content == 'results':

            # median and 16, 84 percentiles of all params
            labels, chain = self._get_chains()

            chain = chain.reshape((-1, chain.shape[-1]))

            p16, p50, p84 = np.percentile(chain, [16, 50, 84], axis=0)

            uncert_minus, uncert_plus = p50 - p16, p84 - p50

            for ind, param in enumerate(labels):

                if 'fixed' in param:
                    mssg += (f'{param[:-8]:>5} = {p50[ind]:.3f} '
                             f'({"fixed":^14})\n')
                else:
                    mssg += (f'{param:>5} = {p50[ind]:.3f} '
                             f'(+{uncert_plus[ind]:.3f}, '
                             f'-{uncert_minus[ind]:.3f})\n')

        if content == 'all' or content == 'metadata':

            with self._openfile() as file:

                # INFO OF RUN
                mssg += f'\nRun Metadata'
                mssg += f'\n{"=" * 12}\n'

                # number of iterations
                Niter = file[self._gname].attrs['iteration']
                mssg += f'Iterations = {Niter}\n'

                # dimensions ndim, nwalkers
                Ndim = file[self._gname].attrs['ndim']
                Nwalkers = file[self._gname].attrs['nwalkers']
                mssg += f'Dimensions = ({Nwalkers}, {Ndim})\n'

                mdata = file['metadata']

                mssg += 'Fixed parameters:\n'
                fixed = mdata['fixed_params'].attrs
                if fixed:
                    for k, v in fixed.items():
                        mssg += f'    {k} = {v}\n'
                else:
                    mssg += '    None\n'

                mssg += 'Excluded components:\n'
                exc = mdata['excluded_likelihoods']
                if exc.size > 0:
                    for i, v in enumerate(exc):
                        mssg += f'    ({i}) {v}\n'
                else:
                    mssg += '    None\n'

                    # TODO add specified bounds/priors
                    # mssg += 'Specified prior bounds'

        out.write(mssg)


class NestedRun(_SingleRunAnalysis):
    '''Analysis and visualization of an nested sampling cluster fitting run.

    Provides a number of flexible plotting, output and summary methods useful
    for the analysis of both the procedure and results of a nested sampling
    fitting run, based on the output file generated by the fitting.

    Parameters
    ----------
    filename : pathlib.Path or str
        Path to the run output HDF5 file.

    observations : gcfit.Observations or None, optional
        The `Observations` instance corresponding to this cluster.
        If None (default), an educated guess will be made on the source location
        (i.e. the `restrict_to` argument) and the observations will be created
        based on the cluster name stored in the output.

    group : str, optional
        Name of the root group in the HDF5 file. Defaults to "nested", the most
        likely name used by `gcfit.nested_fit`.

    name : str, optional
        Custom name for this run.

    *args, **kwargs : dict
        All other arguments are passed to `_SingleRunAnalysis`.
    '''

    @property
    def weights(self):
        '''Array of weights, based on the default `dynesty.weight_function`.'''

        from dynesty.dynamicsampler import weight_function

        # TODO If maxfrac is added as arg, make sure to add here as well

        with self._openfile('metadata') as mdata:
            try:
                stop_kw = {'pfrac': mdata.attrs['pfrac']}

            except KeyError:
                stop_kw = {}

        return weight_function(self.results, stop_kw, return_weights=True)[1][2]

    @property
    def chains(self):
        return self._get_equal_weight_chains()[1]

    @property
    def ESS(self):
        '''The effective sample size.'''
        from scipy.special import logsumexp
        logwts = self.results.logwt
        logneff = logsumexp(logwts) * 2 - logsumexp(logwts * 2)
        return np.exp(logneff)

    @property
    def AIC(self):
        '''Akaike information criterion.'''

        with self._openfile() as file:

            exc = [L.decode() for L in file['metadata/excluded_likelihoods']]

            N = sum([self.obs[comp[0]].size for comp in
                     self.obs.filter_likelihoods(exc, True)])

            k = len(self._get_chains(include_fixed=False)[1])
            lnL0 = np.max(file[self._gname]['logl'][:])

        AIC = -2 * lnL0 + (2 * k) + ((2 * k * (k + 1)) / (N - k - 1))

        return AIC

    @property
    def BIC(self):
        '''Bayesian information criterion.'''

        with self._openfile() as file:

            exc = [L.decode() for L in file['metadata/excluded_likelihoods']]

            N = sum([self.obs[comp[0]].size for comp in
                     self.obs.filter_likelihoods(exc, True)])

            k = len(self._get_chains(include_fixed=False)[1])
            lnL0 = np.max(file[self._gname]['logl'][:])

        BIC = -2 * lnL0 + (k * np.log(N))

        return BIC

    @property
    def _resampled_weights(self):
        '''Resample `weights`.'''
        from scipy.stats import gaussian_kde
        from dynesty.utils import resample_equal

        # "resample" logvols so they all have equal weights
        eq_logvol = resample_equal(-self.results.logvol, self.weights)

        # Compute the KDE of resampled logvols and evaluate on normal logvols
        return gaussian_kde(eq_logvol)(-self.results.logvol)

    def __init__(self, filename, observations=None, group='nested', name=None,
                 *args, **kwargs):

        super().__init__(filename, observations, group, name, *args, **kwargs)

        self.results = self._get_results()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _get_results(self, finite_only=False, *, apply_mask=True):
        '''Return a `dynesty.Results` class reconstructed from this run file.'''
        from dynesty.results import Results

        with self._openfile() as file:

            if finite_only:
                inds = file[self._gname]['logl'][:] > -1e300
            else:
                inds = slice(None)

            r = {}

            Niter = file[self._gname]['logl'].shape[0]

            for k, d in file[self._gname].items():

                if k in ('current_batch', 'initial_batch', 'bound'):
                    continue

                if d.shape and (d.shape[0] == Niter):
                    d = np.array(d)[inds]

                    if apply_mask and self.mask is not None:
                        d = d[self.mask]

                else:
                    d = np.array(d)

                r[k] = d

            # add in any fixed params, if they exist

            labels = self._get_labels(False, False)

            fixed = sorted(
                ((k, v, labels.index(k)) for k, v in
                 file['metadata']['fixed_params'].attrs.items()),
                key=lambda item: labels.index(item[0])
            )

            for k, v, i in fixed:
                r['samples'] = np.insert(r['samples'], i, v, axis=-1)
                r['samples_u'] = np.insert(r['samples_u'], i, v, axis=-1)

        if finite_only:
            # remove the amount of non-finite values we removed from niter
            r['niter'] -= (r['niter'] - r['logl'].size)

        # TODO this is a temp fix for backwards compat. and should be removed
        try:
            r['bound'] = self._reconstruct_bounds()
        except KeyError:
            r['bound'] = None

        r['blob'] = np.full_like(r['logl'], np.nan)  # should be None, but okay

        return Results(r)

    def _reconstruct_bounds(self):
        '''Reconstruct `dynesty.bounding` objects from stored bounding info.'''

        from dynesty import bounding

        with self._openfile(self._gname) as file:

            bnd_grp = file['bound']

            bnds = []
            for i in range(len(bnd_grp)):

                ds = bnd_grp[str(i)]
                btype = ds.attrs['type']

                if btype == 'UnitCube':
                    bnds.append(bounding.UnitCube(ds.attrs['ndim']))

                elif btype == 'Ellipsoid':
                    ctr = ds['centre'][:]
                    cov = ds['covariance'][:]
                    bnds.append(bounding.Ellipsoid(ctr=ctr, cov=cov))

                elif btype == 'MultiEllipsoid':
                    ctrs = ds['centres'][:]
                    covs = ds['covariances'][:]
                    bnds.append(bounding.MultiEllipsoid(ctrs=ctrs, covs=covs))

                elif btype == 'RadFriends':
                    cov = ds['covariances'][:]
                    ndim = ds.attrs['ndim']
                    bnds.append(bounding.RadFriends(ndim=ndim, cov=cov))

                elif btype == 'SupFriends':
                    cov = ds['covariances'][:]
                    ndim = ds.attrs['ndim']
                    bnds.append(bounding.SupFriends(ndim=ndim, cov=cov))

                else:
                    raise RuntimeError('unrecognized bound type ', btype)

        return bnds

    # TODO some ways of handling and plotting initial_batch only clusters
    def _get_chains(self, include_fixed=True, *, apply_mask=True):
        '''Get the "chains" of all samples from this nested sampling run.'''

        with self._openfile() as file:

            chain = file[self._gname]['samples'][:]

            if apply_mask and self.mask is not None:
                chain = chain[self.mask, :]

            labels = self._parameters.copy()

            fixed = sorted(
                ((k, v, labels.index(k)) for k, v in
                 file['metadata']['fixed_params'].attrs.items()),
                key=lambda item: labels.index(item[0])
            )

            if include_fixed:
                for k, v, i in fixed:
                    labels[i] += ' (fixed)'
                    chain = np.insert(chain, i, v, axis=-1)
            else:
                for *_, i in reversed(fixed):
                    del labels[i]

        return labels, chain

    def _get_equal_weight_chains(self, include_fixed=True, add_errors=False, *,
                                 apply_mask=True):
        '''Get the "chains" of samples resampled to be equally weighted.'''

        from dynesty.utils import resample_equal

        with self._openfile() as file:

            if add_errors is False:
                chain = file[self._gname]['samples'][:]

                if apply_mask and self.mask is not None:
                    chain = chain[self.mask, :]

                eq_chain = resample_equal(chain, self.weights)

            else:
                from dynesty.dynamicsampler import weight_function
                sim_run = self._sim_errors(1)[0]
                sim_wt = weight_function(sim_run, {'pfrac': 1.}, True)[1][2]
                eq_chain = resample_equal(sim_run.samples, sim_wt)

            labels = self._parameters.copy()

            fixed = sorted(
                ((k, v, labels.index(k)) for k, v in
                 file['metadata']['fixed_params'].attrs.items()),
                key=lambda item: labels.index(item[0])
            )

            # TODO allow including fixed without labelling as fixed
            if include_fixed:
                for k, v, i in fixed:
                    labels[i] += ' (fixed)'
                    eq_chain = np.insert(eq_chain, i, v, axis=-1)
            else:
                for *_, i in reversed(fixed):
                    del labels[i]

        return labels, eq_chain

    def _reconstruct_priors(self):
        '''Reconstruct a `PriorTransform` object based on the stored info.'''

        with self._openfile('metadata') as mdata:

            stored_priors = mdata['specified_priors']
            fixed = dict(mdata['fixed_params'].attrs)

        prior_params = {}

        for key in self._parameters:
            try:
                type_ = stored_priors[key].attrs['type']
                args = stored_priors[key]['args']

                if args.dtype.kind == 'S':
                    args = args.astype('U')

                prior_params[key] = (type_, *args)
            except KeyError:
                continue

        prior_kwargs = {'fixed_initials': fixed, 'err_on_fail': False}
        return priors.PriorTransforms(prior_params, **prior_kwargs)

    # ----------------------------------------------------------------------
    # Model Visualizers
    # ----------------------------------------------------------------------

    def get_model(self, method='mean', add_errors=False):
        '''Return a single `ModelVisualizer` instance corresponding to this run.

        The visualizer is initialized through the `ModelVisualizer.from_chain`
        classmethod, with the chain from this run and the method given here.

        Parameters
        ----------
        method : {'median', 'mean', 'final'}, optional
            The method used to compute a single `theta` set from the chain.
            Defaults to 'median'.

        add_errors : bool, optional
            Optionally add the statistical and sampling errors, not normally
            accounted for, to the chain of samples used (using
            `self._sim_errors(1)`).

        Returns
        -------
        ModelVisualizer
            The created model visualization object.
        '''

        model_cls = ModelVisualizer if not self._evolved else EvolvedVisualizer

        model_kw = self._get_model_kwargs(note_flexible_BHs=True)

        if method == 'mean':
            theta = self.parameter_means()[0]
            return model_cls.from_theta(theta, self.obs, **model_kw)

        else:
            labels, chain = self._get_equal_weight_chains(add_errors=add_errors)
            return model_cls.from_chain(chain, self.obs, method, **model_kw)

    def get_CImodel(self, N=100, Nprocesses=1, add_errors=False, shuffle=True,
                    load=False):
        '''Return a `CIModelVisualizer` instance corresponding to this run.

        The visualizer is initialized through the `CIModelVisualizer.from_chain`
        classmethod, with the chain from this run and using `N` samples, if
        `load` is False, otherwise will attempt to use the
        `CIModelVisualizer.load` classmethod, assuming a CI model has already
        been created and saved to this same file, under the `model` group.

        Parameters
        ----------
        N : int, optional
            The number of samples to use in computing the confidence intervals.

        Nprocesses : int, optional
            The number of processes to use in a `multiprocessing.Pool` passed
            to the CI model initializer. Defaults to only 1 cpu.

        add_errors : bool, optional
            Optionally add the statistical and sampling errors, not normally
            accounted for, to the chain of samples used (using
            `self._sim_errors(1)`).

        shuffle : bool, optional
            Optionally shuffle the chains. This may be useful if N is too small
            to be representative of the full (reweighted) posteriors, and the
            final samples in the chain are nearly equal (due to their high
            weights).

        load : bool, optional
            If True, will attempt to load a CI model, rather than creating a
            new one.

        Returns
        -------
        CIModelVisualizer
            The created model visualization (with confidence intervals) object.
        '''
        import multiprocessing

        ci_cls = CIModelVisualizer if not self._evolved else CIEvolvedVisualizer

        if load:
            return ci_cls.load(self._filename, observations=self.obs)

        else:
            labels, chain = self._get_equal_weight_chains(add_errors=add_errors)

            if shuffle:
                np.random.default_rng().shuffle(chain, axis=0)

            model_kw = self._get_model_kwargs(note_flexible_BHs=True)

            with multiprocessing.Pool(processes=Nprocesses) as pool:
                return ci_cls.from_chain(chain, self.obs, N,
                                         pool=pool, **model_kw)

    # ----------------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------------

    def plot_marginals(self, fig=None, full_volume=False, **corner_kw):
        '''Plot a "corner plot" showcasing the relationships between parameters.

        Plots a Nparam-Nparam lower-triangular "corner" marginal plot showing
        the projections of all sampled parameter values, using the `corner.py`
        package.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        full_volume : bool, optional
            Use the entire raw chains, not resampled based on the weights.
            This will not show correct posteriors.

        **corner_kw : dict
            All other arguments are passed to `corner.corner`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''
        import corner

        fig, ax = self._setup_multi_artist(fig, shape=None,
                                           constrained_layout=False)

        labels = self._get_labels(math_labels=True, label_fixed=False)

        if full_volume:
            _, chain = self._get_chains()
        else:
            _, chain = self._get_equal_weight_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        # ugly
        ranges = [1. if 'fixed' not in lbl
                  else (chain[0, i] - 1, chain[0, i] + 1)
                  for i, lbl in enumerate(labels)]

        corner_kw.setdefault('plot_datapoints', False)
        corner_kw.setdefault('labelpad', 0.25)

        fig = corner.corner(chain, labels=labels, fig=fig,
                            range=ranges, **corner_kw)

        fig.subplots_adjust(left=0.05, bottom=0.06)

        return fig

    def plot_bounds(self, iteration, fig=None, show_live=False, **kw):
        '''Plot a "corner plot" approximating the bounds at some iteration.

        Plots a Nparam-Nparam lower-triangular "corner" plot showing the
        approximate extent of the bounding distributions of each parameter
        at a given iteration.
        Uses the `plotting.cornerbound` function built into `dynesty`.

        Parameters
        ----------
        iteration : int or list of int
            The iterations of the nested sampling run to show the bounding
            distributions for. If multiple iterations are given, they will be
            overplotted, in order, in different colours.

            tribution at the specified iteration of the nested sampling run.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        show_live : bool, optional
            Also show the live points at this given iteration.
            Doesn't seem correct currently.

        **kw : dict
            All other arguments are passed to `dynesty.plotting.cornerbound`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.

        See Also
        --------
        dynesty.plotting.cornerbound
            Base `dynesty` function used for plotting of each iteration.
        '''
        from dynesty import plotting as dyplot
        from matplotlib.patches import Patch

        # TODO id rather use contours or polygons showing the bounds,
        #   rather than how dyplot does it by sampling a bunch of random points

        # TODO doesnt work for some bound types (like balls)

        # TODO this doesn't seem to work the same way corner did
        # fig = self._setup_multi_artist(fig, shape=(10,10))
        # TODO real strange bug with failing on 4th ind on second function call

        priors = self._reconstruct_priors()

        clr = kw.pop('color', None)

        labels, _ = self._get_chains(include_fixed=False)

        try:
            N = len(iteration)
        except TypeError:
            N = 1
            iteration = [iteration]

        legends = []

        for ind, it in enumerate(iteration):

            if N > 1:
                clr = self.cmap((ind + 1) / N)

            if show_live:
                kw.setdefault('live_color', clr)
                kw.setdefault('live_kwargs', {'marker': 'x'})

            fig = dyplot.cornerbound(self.results, it, fig=fig, labels=labels,
                                     prior_transform=priors, color=clr,
                                     show_live=show_live, **kw)

            legends.append(Patch(facecolor=clr, label=f'Iteration {it}'))

        fig[0].legend(handles=legends)

        return fig[0]

    def plot_weights(self, fig=None, ax=None, show_bounds=False,
                     resampled=False, filled=False, **kw):
        r'''Plot the sample weights as a function of the prior volume.

        Plots the importance weights :math:`\hat{w}_i` of all samples as a
        function of the (log) prior volume :math:`\ln(X)`.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot the weights. Should be a
            part of the given `fig`.

        show_bounds : bool, optional
            Display the location of the weight-bounds used in the computation
            of the (first) dynamical batch likelihood boundaries, as a
            horizontal line across `max(weights) * maxfrac`.

        resampled : bool, optional
            Plot the weights after (equally-weighted) resamping, effectively
            smoothing the weights.

        filled : bool, optional
            Fill between the plotted weights and the x-axis.

        **kw : dict
            All other arguments are passed to `ax.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        wts = self._resampled_weights if resampled else self.weights

        line, = ax.plot(-self.results.logvol, wts, **kw)

        if filled:
            color = mpl_clr.to_rgb(line.get_color())
            facecolor = color + (0.33, )

            ax.fill_between(-self.results.logvol, 0, wts,
                            color=color, fc=facecolor)

        if show_bounds:

            with self._openfile('metadata') as mdata:

                try:
                    maxfrac = mdata.attrs['maxfrac']

                except KeyError:

                    maxfrac = 0.8

                    mssg = "No maxfrac stored in metadata, defaulting to 80%"
                    warnings.warn(mssg)

            ax.axhline(maxfrac * max(wts), c='g')

        ax.set_ylabel('weights')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_probability(self, fig=None, ax=None, **kw):
        '''Plot the posterior probability.

        Plots the total (sum of all components) logged posterior probability
        of the nested sampler as a function of (log) prior volume.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this probability. Should be a
            part of the given `fig`.

        **kw : dict
            All other arguments are passed to `ax.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        finite = self.results.logl > -1e300

        ax.plot(-self.results.logvol[finite], self.results.logl[finite], **kw)

        ax.set_ylabel('Total Log Likelihood')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_evidence(self, fig=None, ax=None, error=False, **kw):
        r'''Plot the estimated evidence.

        Plots the estimated (log) bayesian evidence as a function of the (log)
        prior volume.

        Nested sampling provides a continuous estimate of the bayesian evidence
        based on the integral over the prior volume contained within a given
        iso-likelihood contour.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this evidence. Should be a
            part of the given `fig`.

        error : bool, optional
            Optionally also show the error on the evidence estimation as
            contours on the plot.

        **kw : dict
            All other arguments are passed to `ax.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.

        Notes
        -----
        In nested sampling the evidence integral can be numerically
        approximated using a given set of dead points as:

        .. math::

            \mathcal{Z} = \int_{0}^{1} \mathcal{L}(X)\,dX
                \approx \sum_{i=1}^{N}\,f(\mathcal{L}_i)\,f(\Delta X_i)
                \equiv \sum_{i=1}^{N}\,\hat{w}_i
        '''

        fig, ax = self._setup_artist(fig, ax)

        finite = self.results.logz > -1e300

        logvol = self.results.logvol[finite]
        logz = self.results.logz[finite]

        line, = ax.plot(-logvol, logz, **kw)

        if error:
            err_up = logz + self.results.logzerr[finite]
            err_down = logz - self.results.logzerr[finite]

            ax.fill_between(-logvol, err_up, err_down,
                            color=line.get_color(), alpha=0.5)

        ax.set_ylabel(r'Estimated Evidence $\log(Z)$')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_H(self, fig=None, ax=None, **kw):
        r'''Plot the information integral H.

        Plots the "information" gain (H) provided by the updating of a given
        prior, as characterized by the Kullback-Leibler divergence, as a
        function of the (log) prior volume.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this information. Should be a
            part of the given `fig`.

        **kw : dict
            All other arguments are passed to `ax.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.

        Notes
        -----

        .. math::

            H \equiv \int_{\Omega_{\boldsymbol{\Theta}}} P(\boldsymbol{\Theta})
                \ln\frac{P(\boldsymbol{\Theta})}{\pi(\boldsymbol{\Theta})}\,
                d\boldsymbol{\Theta}
        '''

        fig, ax = self._setup_artist(fig, ax)

        finite = self.results.information > -1e250

        logvol = self.results.logvol[finite]

        ax.plot(-logvol, self.results.information[finite], **kw)

        ax.set_ylabel(r'Information $H \equiv \int_{\Omega_{\Theta}} '
                      r'P(\Theta)\ln\frac{P(\Theta)}{\pi(\Theta)} \,d\Theta$')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_HN(self, fig=None, ax=None, **kw):
        '''Plot the information H by the number of live points N.

        Plots the "information" gain (H) multiplied by the current number of
        live points, as a function of run iteration.
        Intended to compare against one of the termination conditions
        described by (Skilling, 2006)

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this information. Should be a
            part of the given `fig`.

        **kw : dict
            All other arguments are passed to `ax.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        finite = self.results.information > -1e250

        HN = self.results.information * self.results.samples_n

        ax.plot(HN[finite], **kw)

        x = np.arange(0, HN[finite].size)
        ax.plot(x, c='k', alpha=0.15)

        ax.set_ylabel(r'HN')
        ax.set_xlabel('Iteration')

        return fig

    def plot_nlive(self, fig=None, ax=None, **kw):
        '''Plot the number of live points.

        Plots the current number of live points, as a function of the (log)
        prior volume.
        This should remain constant until dynamic sampling begins, increase
        incrementally, and then decrease smoothly until all live points are
        removed.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this number. Should be a
            part of the given `fig`.

        **kw : dict
            All other arguments are passed to `ax.plot`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        ax.plot(-self.results.logvol, self.results.samples_n, **kw)

        ax.set_ylabel(r'Number of live points')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_ncall(self, fig=None, ax=None, **kw):
        '''Plot the number of likelihood calls.

        Plots the total number of likelihood function calls made at each
        step as a function of the (log) prior volume.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this number. Should be a
            part of the given `fig`.

        **kw : dict
            All other arguments are passed to `ax.step`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        kw.setdefault('where', 'mid')

        ax.step(-self.results.logvol, self.results.ncall, **kw)

        ax.set_ylabel(r'Number of likelihood calls')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_KL_divergence(self, fig=None, ax=None, Nruns=100,
                           kl_kwargs=None, **kw):
        from dynesty.utils import kld_error

        fig, ax = self._setup_artist(fig, ax)

        if kl_kwargs is None:
            kl_kwargs = {}

        kw.setdefault('color', 'b')
        kw.setdefault('alpha', 0.25)

        for _ in range(Nruns):

            KL = kld_error(self.results, **kl_kwargs)

            ax.plot(KL, **kw)

        ax.set_ylabel('KL Divergence')
        ax.set_xlabel('Iterations')

        return fig

    def plot_posterior(self, param, fig=None, ax=None, chain=None,
                       flipped=True, kde=True, truth=None, truth_ci=None,
                       *args, **kwargs):
        '''Plot a smoothed posterior distribution of a single parameter.

        Plots a gaussian-KDE smoothed posterior probability distribution of
        a given parameter.
        Designed mainly to be used within the `plot_params` method, but can
        be used on its own.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this posterior. Should be a
            part of the given `fig`.

        chain : np.ndarray, optional
            Optionally supply a (flattened) array of samples to create the
            posterior from. By default, will load the full chain using
            `self._get_chains`.

        flipped : bool, optional
            If True (default) the posterior will be flipped on it's side,
            attached to the left-axis.

        kde : bool, optional
            Whether to plot a gaussian-KDE smoothed posterior (default), or a
            simple histogram.

        truth : float, optional
            Optionally indicate the "true" value as horizontal lines on the
            posterior.

        truth_ci : 2-tuple of float, optional
            Optionally shade between the lower and upper limits of the "truth"
            values, using `plt.axhspan`.

        **kwargs : dict
            All other arguments are passed to the `ax.fill_between` function.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''
        from scipy.stats import gaussian_kde

        fig, ax = self._setup_artist(fig, ax)

        labels = self._get_labels()

        if param not in labels:
            mssg = f'Invalid param "{param}". Must be one of {labels}'
            raise ValueError(mssg)

        if chain is None:

            prm_ind = labels.index(param)
            chain = self._get_equal_weight_chains()[1][..., prm_ind]

        # ------------------------------------------------------------------
        # Plot posterior
        # ------------------------------------------------------------------

        if kde:
            try:
                gkde = gaussian_kde(chain)
            except np.linalg.LinAlgError as err:
                mssg = f"Cannot compute gkde of {param}: {err}"
                raise ValueError(mssg)

            domain = np.linspace(chain.min(), chain.max(), 500)

            plot_func = ax.fill_betweenx if flipped else ax.fill_between
            plot_func(domain, 0, gkde(domain), *args, **kwargs)

        else:
            orientation = "horizontal" if flipped else "vertical"
            ax.hist(chain, orientation=orientation, *args, **kwargs)

        # ------------------------------------------------------------------
        # Plot truths
        # ------------------------------------------------------------------

        if flipped:

            if truth is not None:
                ax.axhline(truth, c='tab:red')

                if truth_ci is not None:
                    ax.axhspan(*truth_ci, color='tab:red', alpha=0.33)

            ax.set_xlim(left=0)

        else:

            if truth is not None:
                ax.axvline(truth, c='tab:red')

                if truth_ci is not None:
                    ax.axvspan(*truth_ci, color='tab:red', alpha=0.33)

            ax.set_ylim(bottom=0)

        return fig

    def plot_params(self, fig=None, params=None, *,
                    posterior_color='tab:blue', posterior_border=True,
                    show_weight=True, fill_type='weights', ylims=None,
                    truths=None, **kw):
        '''Plot a diagnostic figure of the distributions of parameter samples.

        Plots an Nparam-panel figure showcasing the parameter values of all
        samples, over the iteration domain, as well as a KDE-based smoothed
        posterior distribution for each parameter.

        Provides a diagnostic figure for examining the parameter estimation.
        This is a modified version of the diagnostic plot
        first introduced in `Higson et al.
        (2018) <https://ui.adsabs.harvard.edu/abs/2018BayAn..13..873H>`_.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        params : None or list of str, optional
            The parameters to show on this figure. If None (default) all
            parameters (including fixed params) will be shown.

        posterior_color : color, optional
            The colour of the smoothed posterior distributions.

        posterior_border : bool, optional
            If False, will remove the axis frame around the smooth posterior
            distribution to the right of each panel.

        show_weight : bool, optional
            Plot the (resampled) weights above the parameter samples columns,
            using the `plot_weights` method.

        fill_type : {weights, iters, id, batch, bound}, optional
            The mapping used to colour all points within the samples axes.
            Defaults to 'weights'.

        ylims : list[Nparam] of 2-tuples, optional
            Used to set the upper and lower y axis-limits on each parameter.

        truths : np.ndarray[Nparam] or np.ndarray[Nparam, 3], optional
            Optionally indicate the "true" values as horizontal lines on the
            posterior frames. If [Nparam, 3], the values in each row
            will be taken as the median, lower limit and upper limit.

        **kw : dict
            All other arguments are passed to `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        # ------------------------------------------------------------------
        # Setup plotting kwarg defaults
        # ------------------------------------------------------------------

        color = mpl_clr.to_rgb(posterior_color)
        facecolor = color + (0.33, )

        kw.setdefault('marker', '.')

        # ------------------------------------------------------------------
        # Determine which property will define the color-scale of the samples
        # ------------------------------------------------------------------

        if fill_type in ('weights', 'weight', 'wts', 'wt', 'logwt'):
            c = self._resampled_weights

        elif fill_type in ('iterations', 'iters', 'samples_it'):
            c = self.results.samples_it

        elif fill_type in ('id', 'samples_id'):
            c = self.results.samples_id

        elif fill_type in ('batch', 'samples_batch'):
            # TODO when showing batches, make the initial sample distinguishable
            c = self.results.samples_batch

        elif fill_type in ('bound', 'samples_bound'):
            c = self.results.samples_bound

        else:
            mssg = ('Invalid fill type, must be one of '
                    '{weights, iters, id, batch, bound}')
            raise ValueError(mssg)

        # ------------------------------------------------------------------
        # Get the sample chains (weighted and unweighted), paring down the
        # chains to only the desired params, if provided
        # ------------------------------------------------------------------

        labels, chain = self._get_chains()
        eq_chain = self._get_equal_weight_chains()[1]

        # params is None or a list of string labels
        if params is not None:
            prm_inds = [labels.index(p) for p in params]

            labels = params
            chain, eq_chain = chain[..., prm_inds], eq_chain[..., prm_inds]

        # ------------------------------------------------------------------
        # Setup the truth values and confidence intervals
        # ------------------------------------------------------------------

        if truths is not None and truths.ndim == 2:
            # Assume confidence bounds rather than single truth value

            truth_ci = truths[:, 1:]
            truths = truths[:, 0]

        else:
            truth_ci = None

        # ------------------------------------------------------------------
        # Setup axes
        # ------------------------------------------------------------------

        if ylims is None:
            ylims = [(None, None)] * len(labels)

        elif len(ylims) != len(labels):
            mssg = "`ylims` must match number of params"
            raise ValueError(mssg)

        gs_kw = {}

        if (shape := (len(labels) + show_weight, 1))[0] > 5 + show_weight:
            shape = (int(np.ceil(shape[0] / 2)) + show_weight, 2)

            if show_weight:
                gs_kw = {"height_ratios": [0.5] + [1] * (shape[0] - 1)}

        fig, axes = self._setup_multi_artist(fig, shape, sharex=True,
                                             gridspec_kw=gs_kw)

        axes = axes.reshape(shape)

        for ax in axes[-1]:
            ax.set_xlabel(r'$-\ln(X)$')

        # ------------------------------------------------------------------
        # If showing weights explicitly, format the ax and use the
        # `plot_weights` method
        # ------------------------------------------------------------------

        if show_weight:
            for ax in axes[0]:
                # plot weights above scatter plots
                # TODO figure out what colors to use
                self.plot_weights(fig=fig, ax=ax, resampled=True, filled=True,
                                  color=self.cmap(np.inf))

                ax.set_xticklabels([])
                ax.set_xlabel(None)
                ax.set_yticklabels([])
                ax.set_ylabel(None)

                # Theres probably a cleaner way to do this
                divider = make_axes_locatable(ax)
                spacer = divider.append_axes('right', size="25%", pad=0)
                spacer.set_visible(False)

        # ------------------------------------------------------------------
        # Plot each parameter
        # ------------------------------------------------------------------

        for ind, ax in enumerate(axes[1:].flatten()):

            # --------------------------------------------------------------
            # Get the relevant samples.
            # If necessary, remove any unneeded axes
            # --------------------------------------------------------------

            try:
                prm, eq_prm = chain[:, ind], eq_chain[:, ind]
                lbl = labels[ind]
            except IndexError:
                # If theres an odd number of (>5) params need to delete last one
                # TODO preferably this would also resize this column of plots
                ax.remove()
                continue

            # --------------------------------------------------------------
            # Divide the ax to accomodate the posterior plot on the right
            # --------------------------------------------------------------

            divider = make_axes_locatable(ax)
            post_ax = divider.append_axes('right', size="25%", pad=0, sharey=ax)

            post_ax.set_xticks([])

            # --------------------------------------------------------------
            # Plot the samples with respect to ln(X)
            # --------------------------------------------------------------

            # TODO the y tick values have disappeared should be on the last axis
            ax.scatter(-self.results.logvol, prm, c=c, cmap=self.cmap, **kw)

            ax.set_ylabel(lbl)
            ax.set_xlim(left=0)

            # --------------------------------------------------------------
            # Plot the posterior distribution (accounting for weights)
            # --------------------------------------------------------------

            post_kw = {
                'chain': eq_prm,
                'flipped': True,
                'truth': truths if truths is None else truths[ind],
                'truth_ci': truth_ci if truth_ci is None else truth_ci[ind],
                'color': color,
                'fc': facecolor
            }

            try:
                self.plot_posterior(lbl, fig=fig, ax=post_ax, **post_kw)
            except ValueError:
                post_ax.axhline(np.median(prm), color=color)

            if not posterior_border:
                post_ax.axis('off')

            # TODO maybe put ticks on right side as well?
            for tk in post_ax.get_yticklabels():
                tk.set_visible(False)

            ax.set_ylim(ylims[ind])

        return fig

    def plot_IMF(self, fig=None, ax=None, show_canonical='all', ci=True):
        '''Plot the IMF, based on the alpha exponents.'''
        def salpeter(m):
            return m**-2.35

        def chabrier(m):
            k = 0.158 * np.exp(-(-np.log10(0.08))**2 / (2 * 0.69**2))
            imf = k * m**-2.3
            imf[m <= 1] = (0.158 * (1. / m[m <= 1])
                           * np.exp(-(np.log10(m[m <= 1]) - np.log10(0.08))**2
                                    / (2 * 0.69**2)))
            return imf

        def kroupa(m):
            imf = 0.08**-0.3 * (0.5 / 0.08)**-1.3 * (m / 0.5)**-2.3
            imf[m < 0.5] = 0.08**-0.3 * (m[m < 0.5] / 0.08)**-1.3
            imf[m < 0.08] = m[m < 0.08]**-0.3
            return imf

        def this_imf(m, perc=50):
            '''perc is percentile of alpha chain to use'''

            ch = self._get_equal_weight_chains()[1]
            a1, a2, a3 = np.percentile(ch[:, 8:11], perc, axis=0)

            imf = 0.5**-a1 * (1 / 0.5)**-a2 * (m / 1)**-a3
            imf[m < 1] = 0.5**-a1 * (m[m < 1] / 0.5)**-a2
            imf[m < 0.5] = m[m < 0.5]**-a1
            return imf

        fig, ax = self._setup_artist(fig, ax)

        m0 = np.array([1])
        m_domain = np.logspace(-2, 2, 400)

        if show_canonical is True or show_canonical == 'all':
            show_canonical = {'salpeter', 'chabrier', 'kroupa'}

        if 'salpeter' in show_canonical:
            norm = salpeter(m0)
            ax.loglog(m_domain, salpeter(m_domain) / norm, label='Salpeter')

        if 'chabrier' in show_canonical:
            norm = chabrier(m0)
            ax.loglog(m_domain, chabrier(m_domain) / norm, label='Chabrier')

        if 'kroupa' in show_canonical:
            norm = kroupa(m0)
            ax.loglog(m_domain, kroupa(m_domain) / norm, label='Kroupa')

        # plot median
        med_plot, = ax.loglog(m_domain, this_imf(m_domain) / this_imf(m0))

        # if ci, plot confidence interval
        if ci:
            lower = this_imf(m_domain, perc=15.87) / this_imf(m0, perc=15.87)
            upper = this_imf(m_domain, perc=84.13) / this_imf(m0, perc=84.13)

            # TODO better label?
            ax.fill_between(m_domain, upper, lower,
                            alpha=0.3, color=med_plot.get_color(),
                            label=getattr(self, 'name', None))

        ax.set_xlabel(r'Mass $[M_{\odot}]$')
        ax.set_ylabel(r'Mass Function $\xi(m)\Delta m$')

        ax.legend()

        return fig

    # ----------------------------------------------------------------------
    # Parameter estimation
    # ----------------------------------------------------------------------

    def _sim_errors(self, Nruns=250):
        '''add the statistical and sampling errors not normally accounted for
        by using the built-in `resample_run` function

        returns list `Nruns` results
        '''
        from dynesty.utils import resample_run

        return [resample_run(self.results) for _ in range(Nruns)]

    def parameter_means(self, Nruns=250, sim_runs=None, return_samples=True):
        '''Compute the mean of each parameter with corresponding errors.

        Returns the means of each parameter posterior estimation, with the
        corresponding error on this statistic. The uncertainties come from the
        two main sources of errors in nested sampling; statistical errors
        associated with the uncertainties surrounding the prior volume and
        sampling errors associated with the integral over the parameters
        of interest.

        These errors can be computed using the standard deviation of the mean
        from a number of "simulated" (resampled and jittered) runs based on
        this run.
        See https://dynesty.readthedocs.io/en/latest/errors.html for a more
        thorough description.

        Parameters
        ----------
        Nruns : int, optional
            The number of simulated runs to use to estimate the uncertainties.

        sim_runs : None or list of dynesty.Results
            A list of simulated runs to use. A precomputed list of runs may be
            provided, otherwise they will be computed using `_sim_errors`.

        return_samples : bool, optional
            Optionally also return the full array of parameter means from each
            simulated run.

        Returns
        -------
        mean : np.ndarray[Nparams]
            Mean values of each parameter.

        err : np.ndarray[Nparams]
            Errors on the mean of each parameter.

        means_arr : np.ndarray[Nruns, Nparams]
            The mean values of each parameter for each simulated run.
        '''
        from dynesty.utils import mean_and_cov

        if sim_runs is None:
            sim_runs = self._sim_errors(Nruns)

        # TODO returns a weird nan very rarely?
        means = []
        for res in sim_runs:
            wt = np.exp(res.logwt - res.logz[-1])
            means.append(mean_and_cov(res.samples, wt)[0])

        # TODO I think this assumes symmetrical guassian dist, is that alright?
        mean = np.mean(means, axis=0)
        err = np.std(means, axis=0)

        if return_samples:
            return mean, err, np.array(means)
        else:
            return mean, err

    def parameter_vars(self, Nruns=250, sim_runs=None, return_samples=True):
        '''Compute the variance of each parameter with corresponding errors.

        Returns the covariance array for each parameter posterior estimation,
        with the corresponding error on this statistic.
        The uncertainties come from the two main sources of errors in nested
        sampling; statistical errors associated with the uncertainties
        surrounding the prior volume and sampling errors associated with the
        integral over the parameters of interest.

        These errors can be computed using the standard deviation of the
        variance from a number of "simulated" (resampled and jittered) runs
        based on this run.
        See https://dynesty.readthedocs.io/en/latest/errors.html for a more
        thorough description.

        Parameters
        ----------
        Nruns : int, optional
            The number of simulated runs to use to estimate the uncertainties.

        sim_runs : None or list of dynesty.Results
            A list of simulated runs to use. A precomputed list of runs may be
            provided, otherwise they will be computed using `_sim_errors`.

        return_samples : bool, optional
            Optionally also return the full array of parameter variances from
            each simulated run.

        Returns
        -------
        vars : np.ndarray[Nparams, Nparams]
            Covariance matric for all parameters.

        err : np.ndarray[Nparams, Nparams]
            Errors on the covariance matric for all parameters.

        vars_arr : np.ndarray[Nruns, Nparams, Nparams]
            The covariance matrix for each simulated run.
        '''
        from dynesty.utils import mean_and_cov

        if sim_runs is None:
            sim_runs = self._sim_errors(Nruns)

        vars_ = []
        for res in sim_runs:
            wt = np.exp(res.logwt - res.logz[-1])
            vars_.append(mean_and_cov(res.samples, wt)[1])

        mean = np.mean(vars_, axis=0)
        err = np.std(vars_, axis=0)

        if return_samples:
            return mean, err, np.array(vars_)
        else:
            return mean, err

    # ----------------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------------

    def parameter_summary(self, *, N_simruns=100, label_fixed=False):
        '''Compute the mean and std.dev. on each parameter.

        Computes and returns a dictionary with the mean and standard deviation
        of each parameter, as given by `parameter_means` and
        `np.sqrt(np.diag(parameter_vars))`.

        Parameters
        ----------
        N_simruns : int, optional
            The number of simulated runs used to compute the means and errors
            on each parameter, through `parameter_{means,vars}`.

        label_fixed : bool, optional
            If True, adds " (fixed)" to the end of any parameters which were
            fixed during fitting.

        Returns
        -------
        dict
            Dictionary of parameter labels and 2-tuples of mean and standard
            deviations.
        '''

        labels = self._get_labels(label_fixed=label_fixed)

        sr = self._sim_errors(N_simruns)
        mns, _ = self.parameter_means(sim_runs=sr, return_samples=False)
        vrs, _ = self.parameter_vars(sim_runs=sr, return_samples=False)
        std = np.sqrt(np.diag(vrs))

        return {lbl: (mns[ind], std[ind]) for ind, lbl in enumerate(labels)}

    def print_summary(self, out=None, content='all', *, N_simruns=100):
        '''Write a short summary of the run results and metadata.

        Write out (to a file or stdout) a short summary of the final
        median and 1σ parameter values, as well as some metadata surrounding
        the fitting run setup, such as fixed parameters, and statistics on
        the run progression, like the effective sample size and efficiency.

        Parameters
        ----------
        out : None or file-like object, str, or pathlib.Path, optional
            The file to write out the summary to. If None (default) will be
            printed to stdout.

        content : {'all', 'results', 'metadata'}
            Which parts of the summary to write. If "results", will print only
            the parameter values. If "metadata", will print only the run
            metadata. If "all" (default), prints both.

        N_simruns : int, optional
            The number of simulated runs used to compute the means and errors
            on each parameter, through `parameter_{means,vars}`.
        '''
        # TODO add more 2nd level results, like comments on BH masses, etc

        if out is None:
            out = sys.stdout

        mssg = f'{self}'
        mssg += f'\n{"=" * len(mssg)}\n'

        # RESULTS

        # organize this arg (content) more like it is in cum_mass plots
        if content == 'all' or content == 'results':

            sr = self._sim_errors(N_simruns)

            mns, σ_mns = self.parameter_means(sim_runs=sr, return_samples=False)
            vrs, σ_vrs = self.parameter_vars(sim_runs=sr, return_samples=False)
            std, σ_std = np.sqrt(np.diag(vrs)), np.sqrt(np.diag(σ_vrs))

            # median and 16, 84 percentiles of all params
            labels = self._get_labels()

            mssg += f'{" " * 8}{"Mean":^14} | {"Std. Dev.":^14}\n'

            logging.debug(f"printing summary of {labels} -> {mns}")

            for ind, param in enumerate(labels):

                logging.debug(f"---> ({ind}) {param} {mns[ind]}")

                if 'fixed' in param:
                    mssg += (f'{param[:-8]:>5} = {mns[ind]:.3f} '
                             f'({"fixed":^14}) | ')
                    mssg += f'{"-" * 14}\n'
                else:
                    mssg += (f'{param:>5} = {mns[ind]:.3f} '
                             f'(±{σ_mns[ind]:.3f}) | ')
                    mssg += (f'{std[ind]:.3f} (±{σ_std[ind]:.3f})\n')

        if content == 'all' or content == 'setup':

            # INFO OF RUN
            mssg += f'\nRun Setup'
            mssg += f'\n{"=" * 9}\n'

            with self._openfile('metadata') as mdata:

                mssg += 'Fixed parameters:\n'
                fixed = mdata['fixed_params'].attrs
                if fixed:
                    for k, v in fixed.items():
                        mssg += f'    {k} = {v}\n'
                else:
                    mssg += '    None\n'

                mssg += 'Excluded components:\n'
                exc = mdata['excluded_likelihoods']
                if exc.size > 0:
                    for i, v in enumerate(exc):
                        mssg += f'    ({i}) {v}\n'
                else:
                    mssg += '    None\n'

                # TODO add specified bounds/priors
                # mssg += 'Specified prior bounds'

        if content == 'all' or content == 'metadata':

            mssg += f'\nRun Metadata'
            mssg += f'\n{"=" * 12}\n'

            mssg += f'{"ESS":>6} = {self.ESS:.2f}\n'
            mssg += f'{"AIC":>6} = {self.AIC:.2f}\n'
            mssg += f'{"BIC":>6} = {self.BIC:.2f}\n'
            mssg += f'{"logL0":>6} = {np.max(self.results.logl):.2f}\n'
            mssg += (f'{"logz":>6} = {self.results.logz[-1]:.2f} '
                     f'(±{self.results.logzerr[-1]:.2f})\n')
            mssg += f'{"niter":>6} = {int(self.results.niter)}\n'
            mssg += f'{"ncall":>6} = {int(np.sum(self.results.ncall))}\n'
            mssg += f'{"eff":>6} = {float(self.results.eff):.2f}\n'

        out.write(mssg)


# --------------------------------------------------------------------------
# Collections of Runs
# --------------------------------------------------------------------------


def _check_for_operator(func):
    '''Decorator which parses param str for math operations to apply.'''
    import functools
    import operator

    opers = {'+': operator.add, '-': operator.sub,
             '*': operator.mul, '/': operator.truediv}

    # TODO also need to implement this for latex_label if want that to work

    def _get_param_or_scalar(self, param, *args, **kwargs):
        '''Check if this is a constant number instead of a parameter.'''
        try:
            return func(self, param.strip(), *args, **kwargs)
        except ValueError as err:
            try:
                return [float(param.strip()),] * len(self)
            except ValueError:
                raise err

    @functools.wraps(func)
    def _operator_decorator(self, param, *args, **kwargs):

        if found_op := (set(param) & opers.keys()):

            if len(found_op) > 1:
                mssg = "More than one operation not supported"
                raise ValueError(mssg)

            op_name = found_op.pop()

            param1, param2 = param.split(op_name)

            res1 = _get_param_or_scalar(self, param1, *args, **kwargs)
            # res1 = func(self, param1.strip(), *args, **kwargs)

            res2 = _get_param_or_scalar(self, param2, *args, **kwargs)
            # res2 = func(self, param2.strip(), *args, **kwargs)

            final = list(map(opers[op_name], res1, res2))

        else:
            final = func(self, param, *args, **kwargs)

        return final

    return _operator_decorator


class _Annotator:
    '''Figure hook which annotates clicked points with the cluster names.

    When hooked into a figure, allows the user of an interactive plot to
    click on a certain data point and have that datapoint be highlighted and
    the name of the corresponding cluster be shown in an annotation.

    Works only for scatter plots, and `picker=True` must be given during the
    scatter plot function call.

    This hook is tied to a given figure through
    `fig.canvas.mpl_connect('pick_event', self)`. Plotting multiple times on
    the same axes, or using multiple axes on a single figure, may produce
    nonsensical or incorrect results (but will also work fine sometimes).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to annotate.

    ax : matplotlib.axes.Axes
        The axes instance to place the annotation on.
        Technically does not have to be the same ax the data was plotted on.

    runs : list of _SingleRunAnalysis
        Collection of run objects which correspond to (and are in the same
        order as) the data plotted. These are used to retrieve the string
        `name`.

    xdata, ydata : np.ndarray
        The x and y data plotted. Used to highlight upon clicking. If incorrect
        data is given, will highlight the incorrect locations on the figure.

    loc : str, optional
        The location of the annotation box. Options are the same as those
        describing `matplotlib.legend.Legend` locations (except 'best').

    **annot_kw : dict
        All other arguments are passed to the `AnchoredText` object.
    '''

    highlight_style = {'marker': 'D', 'linestyle': 'None',
                       'mfc': 'none', 'mec': 'red', 'mew': 2.0}

    def set_text(self, text):
        ''''Get the corresponding `TextArea` and set its text.'''
        return self.annotation.get_child().set_text(text)

    def set_highlight(self, x, y):
        '''Plot a highlighting marker at the selected location.'''
        self.highlight, = self.ax.plot(x, y, **self.highlight_style)

    def remove_highlight(self):
        '''Remove any already placed highlighting marker.'''
        if self.highlight:
            self.highlight.remove()
            self.highlight = None

    def __init__(self, fig, ax, runs, xdata, ydata,
                 loc='upper right', **annot_kw):
        self.fig, self.ax = fig, ax
        self.runs, self.xdata, self.ydata = runs, xdata, ydata

        self.fig.canvas.mpl_connect('pick_event', self)

        self.cur_ind = None

        # initialize annotation box
        self.annotation = mpl_obx.AnchoredText(None, loc=loc, **annot_kw)
        self.ax.add_artist(self.annotation)
        self.annotation.set_visible(False)

        self.highlight = None

    def __call__(self, event):
        '''Call signature connected to a figure and run at a `pick_event`'''
        ind = event.ind[0]

        cluster = self.runs[ind].name

        # get rid of the current highlight point
        self.remove_highlight()

        # rehitting the same one, hide the annotation and highlight
        if ind == self.cur_ind:
            self.cur_ind = None

            self.annotation.set_visible(False)

            self.set_text(None)

        # hitting new one, reset the text, ensure its visible and add highlight
        else:
            self.cur_ind = ind

            self.annotation.set_visible(True)

            self.set_text(cluster)

            self.set_highlight(self.xdata[ind], self.ydata[ind])

        self.fig.canvas.draw()


class RunCollection(_RunAnalysis):
    '''Analysis and visualization of an collection of multiple runs.

    Provides a number of flexible plotting, output and summary methods useful
    for the analysis of distributions, relationships and correlations between
    the results of multiple runs at once. This class is meant to enable the
    analysis of a larger population of fits of different clusters at once, and
    explore the relationships in cluster parameters and make comparisons with
    results from the literature.
    Multiple different runs of fitting on the same cluster may also be given,
    though care should be taken that each has it's own `.name` to avoid
    confusion.

    Parameters
    ----------
    runs, *, sort=True

    runs : list of _SingleRunAnalysis
        List of run objects which will make up this collection. In theory,
        MCMC and Nested sampling runs can both be used interchangeably.

    sort : bool, optional
        If True (default), the given list of runs will be sorted by their
        `.name` attributes. Sorting decides the positioning of the runs in
        some plot functions.
    '''

    _src = None
    models = None

    def __str__(self):
        mssg = f"Collection of Runs"

        if self._src:
            mssg += f" from {self._src}"

        return mssg

    def __len__(self):
        return self.runs.__len__()

    # ----------------------------------------------------------------------
    # Interacting with Runs
    # ----------------------------------------------------------------------

    @property
    def names(self):
        '''List of `.name`s of each run in this collection.'''
        return [r.name for r in self.runs]

    def __iter__(self):
        '''Return an iterator over the individual runs in this collection.'''
        # Important that the order of self.runs (and thus this iter) is constant
        return iter(self.runs)

    def __add__(self, other):
        '''Add the runs of two RunCollections together and return new object.'''

        # TODO make this and __or__ preserve stuff like cmap

        new_runs = self.runs + other.runs

        if repeated_names := set(self.names) & set(other.names):
            mssg = f"Runs {repeated_names} repeated in both {self} and {other}"
            raise ValueError(mssg)

        return RunCollection(new_runs, sort=False)

    def __or__(self, other):
        '''Return a new RunCollection with merged runs from self and other
        with runs from other taking priority when in both (runs identified by
        their name).
        '''

        self_runs = dict(zip(self.names, self.runs))
        other_runs = dict(zip(other.names, other.runs))

        new_runs = list((self_runs | other_runs).values())

        new_runs.sort(key=lambda r: (self.names + other.names).index(r.name))

        return RunCollection(new_runs, sort=False)

    def get_run(self, name):
        '''Return the a single run from this collection with a given `name`.'''
        for run in self.runs:
            if run.name == name:
                return run
        else:
            mssg = f"No Run found with name {name}"
            raise ValueError(mssg)

    def filter_runs(self, pattern, sort_by=None, sort=True, **kwargs):
        '''Filter all runs based on names and return a new object with them.

        Based on a given string pattern, filters out all runs within this
        collection matching this pattern (as based on `fnmatch.filter`) and
        returns a new `RunCollection` instance with only those filtered runs.

        Parameters
        ----------
        pattern : str or list of str
            A pattern used to filter all run names, using the glob rules
            provided by `fnmatch`.
            Also allowed is a list of cluster names, which will filter on
            the matching runs. A list of names will not use pattern matching,
            and the given names must match exactly.

        sort_by : {'old', 'new', None}, optional
            Sort runs in new collection by either the order in this collection
            or the list of names (if it is a list, otherwise does None).
            If None (default), simply passes `sort` to the new collection init
            and sorting is handled there, by name. This argument is only used
            if `sort` is True.

        sort : bool, optional
            Whether or not to sort this run. If `sort_by` is None, this
            argument is passed to the new run collection init.

        **kwargs : dict
            All other arguments are passed to the new RunCollection object.

        Returns
        -------
        RunCollection
            A new run collection instance based on the filtered out runs.
        '''
        import fnmatch

        try:
            filtered_names = fnmatch.filter(self.names, pattern)

        except TypeError:

            try:
                filtered_names = list(set(self.names) & set(pattern))

            except TypeError:

                mssg = (f'expected str pattern or list of names, '
                        f'not {type(pattern)}')

                raise TypeError(mssg)

        if not filtered_names:
            mssg = f"No matched runs found with pattern {pattern}"
            raise ValueError(mssg)

        if sort:
            if sort_by == 'old':
                filtered_names.sort(key=lambda n: self.names.index(n))
                sort = False

            elif sort_by == 'new':
                filtered_names.sort(key=lambda n: pattern.index(n))
                sort = False

            else:
                pass

        runs = [self.get_run(r) for r in filtered_names]

        rc = RunCollection(runs, sort=sort, **kwargs)
        rc.cmap = self.cmap

        return rc

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------

    def __init__(self, runs, *, sort=True):

        if sort:
            runs.sort(key=lambda run: run.name)

        self.runs = runs

        labels = runs[0]._get_labels(label_fixed=False)

        # TODO this `equal_weights...` breaks when using MCMCRun's
        self._params = [dict(zip(labels, r._get_equal_weight_chains()[1].T))
                        for r in runs]

        self._mdata = [{k: [v, ] for k, v in r.obs.mdata.items()}
                       for r in self.runs]

    @classmethod
    def from_dir(cls, directory, pattern='**/*hdf', strict=False,
                 *args, sampler='nested', run_kwargs=None, **kwargs):
        '''Initialize a run collection based on run files found in a directory.

        Search for run output files (as created by the relevant fitting
        functions) in a given directory and create a new RunCollection instance
        based on the run objects created from them.

        `NestedRun` or `MCMCRun` instances will be created for found file.
        Which class is used must be consistent over all runs, and specified
        a priori.

        Parameters
        ----------
        directory : str or pathlib.Path
            Path to the directory to search for files within.

        pattern : str, optional
            The glob pattern used to find all files within the given directory.
            Should be tuned in order to only return valid run files.
            Default is to search (recursively) for all HDF files within
            the directory.

        strict : bool, optional
            If True, will raise an RuntimeError if any discovered file fails
            when creating a run class.

        sampler : {'nested', 'mcmc'}, optional
            Whether to initialize each run as either a `NestedRun` or `MCMCRun`.

        run_kwargs : dict, optional
            Optional arguments passed to all individual run initialization.

        *args, **kwargs
            All other arguments are passed to the new RunCollection object.
        '''

        cls._src = f'{directory}/{pattern}'

        directory = pathlib.Path(directory)

        if sampler == 'nested':
            run_cls = NestedRun
        elif sampler == 'mcmc':
            run_cls = MCMCRun
        else:
            mssg = "Invalid sampler. Must be one of {'nested', 'mcmc'}"
            raise ValueError(mssg)

        if run_kwargs is None:
            run_kwargs = {}

        runs = []

        for fn in directory.glob(pattern):

            try:
                run = run_cls(fn, **run_kwargs)
                run.name = run.obs.cluster

            except KeyError as err:

                mssg = f'Failed to create run for {fn}: {err}'

                if strict:
                    raise RuntimeError(mssg)
                else:
                    logging.debug(mssg)
                    continue

            runs.append(run)

        if not runs:
            mssg = f"No valid runs found in {directory}"
            raise RuntimeError(mssg)

        return cls(runs, *args, **kwargs)

    @classmethod
    def from_files(cls, file_list, strict=False,
                   *args, sampler='nested', run_kwargs=None, **kwargs):
        '''Initialize a run collection based on a list of run files.

        Given a list of paths to a number of run output files (as created by
        the relevant fitting functions), creates a new RunCollection instance
        based on the run objects created from them.

        `NestedRun` or `MCMCRun` instances will be created for found file.
        Which class is used must be consistent over all runs, and specified
        a priori.

        Parameters
        ----------
        file_list : list of str
            A list of paths to valid run output files.

        strict : bool, optional
            If True, will raise an RuntimeError if any discovered file fails
            when creating a run class.

        sampler : {'nested', 'mcmc'}, optional
            Whether to initialize each run as either a `NestedRun` or `MCMCRun`.

        run_kwargs : dict, optional
            Optional arguments passed to all individual run initialization.

        *args, **kwargs
            All other arguments are passed to the new RunCollection object.
        '''

        if not file_list:
            mssg = f"`file_list` must not be empty"
            raise ValueError(mssg)

        if sampler == 'nested':
            run_cls = NestedRun
        elif sampler == 'mcmc':
            run_cls = MCMCRun
        else:
            mssg = "Invalid sampler. Must be one of {'nested', 'mcmc'}"
            raise ValueError(mssg)

        if run_kwargs is None:
            run_kwargs = {}

        runs = []

        for file in file_list:

            file = pathlib.Path(file).resolve()

            if not file.exists():
                mssg = f"No such file: '{file}'"
                raise FileNotFoundError(mssg)

            try:
                run = run_cls(file, **run_kwargs)
                run.name = run.obs.cluster

            except KeyError as err:

                mssg = f'Failed to create run for {file}: {err}'

                if strict:
                    raise RuntimeError(mssg)
                else:
                    logging.debug(mssg)
                    continue

            runs.append(run)

        return cls(runs, *args, **kwargs)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _update(self):
        '''Quickly update all run params, in case something has changed.'''
        labels = self.runs[0]._get_labels(label_fixed=False)
        self._params = [dict(zip(labels, r._get_equal_weight_chains()[1].T))
                        for r in self.runs]

        self._mdata = [{k: [v, ] for k, v in r.obs.mdata.items()}
                       for r in self.runs]

    def _get_from_run(self, param):
        '''Get chains from either stored params or the runs directly.'''

        # try to get it from the best-fit params or metadata
        try:
            chains = [
                {**self._params[ind], **self._mdata[ind]}[param]
                for ind, run in enumerate(self.runs)
            ]

        except KeyError as err:

            # otherwise try to get from model properties

            try:
                chains = [[getattr(run, param), ] for run in self.runs]

            except AttributeError:
                mssg = f'No such parameter "{param}" was found'
                raise ValueError(mssg) from err

        return chains

    def _get_from_model(self, param, *, with_units=True, **kwargs):
        '''Get chains one of the attributes from models (like BH mass)

        If havent generated models already (using the get_*models function),
        then they will be computed here, with all **kwargs pass to it.
        if N is passed, will gen CI models, otherwise normal mean models.
        '''

        # Compute models now
        if self.models is None:
            if 'N' in kwargs or 'load' in kwargs:
                self.get_CImodels(**kwargs)
            else:
                # try to load a CI first, then revert to a single model
                try:
                    self.get_CImodels(load=True, **kwargs)

                except RuntimeError:
                    self.get_models(**kwargs)

        data = getattr(self.models, param)

        if not with_units:
            try:
                data = [ds.value for ds in data]
            except AttributeError:
                pass

        # return the full dataset for each run
        return data

    def _get_param(self, param, *, sigma=1, **kwargs):
        '''Return median, -+1σ for a θ, metadata or model quant for all runs'''

        # get parameter chains
        chains = self._get_param_chains(param, **kwargs)

        # Keep units, if they've got them (optional kwarg to _get_from_model)
        base = u.Quantity if isinstance(chains[0], u.Quantity) else np.array

        # Compute the statistics based on the chains
        if sigma == 0:
            q = [50.]
        elif sigma == 1:
            q = [50., 15.87, 84.13]
        elif sigma == 2:
            q = [50., 2.80, 15.87, 84.13, 97.72]
        else:
            raise ValueError(f'Invalid sigma {sigma} (0, 1, 2)')

        out = base([np.nanpercentile(ds, q=q) for ds in chains]).T
        out[1:] = np.abs(out[1:] - out[0])

        return out

    @_check_for_operator
    def _get_param_chains(self, param, *,
                          allow_model=True, force_model=False, **kwargs):
        '''Return the full chain for a θ, metadata or model quant for all runs.

        `allow_model=False` if you want to really avoid model params (i.e. dont
        want to compute the models) all kwargs are passed to get_model otherwise

        `force_model=True` if you want to skip the run params entirely and force
        `_get_from_model` (useful for getting some things like scaled `ra`)

        One operation (+-*/) can be included to return two different parameters
        combined with said operation.
        '''

        try:
            if logged := param.startswith('log_'):
                param = param[4:]
        except AttributeError:
            # pass gracefully, as this should be allowed to fail below
            logged = False
            pass

        # try to get it from the best-fit params, metadata or run stats
        try:
            if force_model:
                mssg = '`force_model` is True, must set `allow_model=True`'
                raise ValueError(mssg)

            chains = self._get_from_run(param)

        # otherwise try to get from model properties
        # this is only worst case because may take a long time to gen models
        except ValueError as err:

            if allow_model:
                try:
                    chains = self._get_from_model(param, **kwargs)

                except AttributeError:
                    mssg = f'No such parameter "{param}" was found in models'
                    raise ValueError(mssg) from err
            else:
                raise err

        if logged:

            scale = 1
            if hasattr(chains[0], 'unit'):
                scale /= chains[0].unit

            chains = [np.log10(ch * scale) for ch in chains]

        return chains

    def _get_latex_labels(self, param, *, with_units=True, force_model=False):
        '''Return the given param name in math mode, for plotting.'''

        try:
            if logged := param.startswith('log_'):
                param = param[4:]
        except AttributeError:
            # pass gracefully, as this should be allowed to fail below
            logged = False
            pass

        if param == 'ra' and force_model:
            param = 'ra_model'

        label = _get_latex_label(param, with_units=with_units)

        if logged:
            # TODO obviously currently fails for operation-param pairs
            label = fr'$\log_{{10}}\left( {label.strip("$")} \right)$'

        return label

    def _add_colours(self, ax, mappable, cparam, clabel=None, *, alpha=1.,
                     add_colorbar=True, extra_artists=None, math_label=True,
                     fix_cbar_ticks=True, cbounds=None):
        '''Add colours to all artists and add the relevant colorbar to ax.
        Unnecessarily complicated to account for diverse artists (violinplot).
        '''
        import matplotlib.colorbar as mpl_cbar

        def set_colour(art, clr):
            try:
                art.set_color(clr)
            except ValueError as err:
                mssg = (f"Could not set colour '{clr}'. Colours must be a "
                        "valid model parameter, matplotlib colour or float")

                # try to fix what we broke a bit. Will reset all colours, but
                #   otherwise a very ugly error may explode upon canvas drawing
                art.set_color(None)

                raise ValueError(mssg) from err

        # Get colour values
        try:
            cvalues, *_ = self._get_param(cparam, with_units=False)
            clabel = cparam if clabel is None else clabel

        # catch ValueError (invalid string) or TypeError (array of numeric vals)
        except (ValueError, TypeError):
            cvalues = cparam

        # coerce everythign to numpy array to handle various possible inputs
        cvalues = np.atleast_1d(cvalues)

        # If cvalues looks like they might be valid plt colours, move on
        if cvalues.dtype.kind in 'US':
            colors = cvalues

            add_colorbar = False
            # TODO not sure about this error, it could be cparam was supposed
            #   to be a model param, but was a typo or something, not this err
            # if add_colorbar:
            #     mssg = "Cannot add a colourbar given explicit named colours"
            #     raise ValueError(mssg)

        # otherwise, map the values to the class' colormap
        else:

            if cbounds is None:
                cbounds = cvalues.min(), cvalues.max()

            cnorm = mpl_clr.Normalize(*cbounds)
            colors = self.cmap(cnorm(cvalues))

            colors[:, -1] = alpha

        # apply colour to all artists
        if mappable is not None:
            # mappable.set_color(colors)
            set_colour(mappable, colors)
            mappable.cmap = self.cmap

        if extra_artists is not None:
            for artist in extra_artists:

                # Set colors normally
                try:
                    # artist.set_color(colors)
                    set_colour(artist, colors)

                # If fails, attempt to set one colour at a time
                except (ValueError, AttributeError) as err:

                    if colors.shape[0] == 1:
                        colors = np.repeat(colors, len(self), axis=0)

                    try:
                        for i, subart in enumerate(artist):
                            # subart.set_color(colors[i])
                            set_colour(subart, colors[i])

                    except (ValueError, TypeError):
                        mssg = f'Cannot `set_color` of extra artist "{artist}"'
                        raise TypeError(mssg) from err

        if add_colorbar:
            # make ax for colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)

            # make colorbar
            cbar = mpl_cbar.Colorbar(cax, mappable, cmap=self.cmap)

            clabel = self._get_latex_labels(clabel) if math_label else clabel
            cbar.set_label(clabel)

            # if desired, explicitly set ticks at 25% intervals of bar
            if fix_cbar_ticks:
                cticks = [0, .25, .5, .75, 1.]
                ctick_labels = [f'{t:.2f}' for t in cnorm.inverse(cticks)]
                cbar.set_ticks(cticks, labels=ctick_labels)

            return cbar

        else:
            return None

    def _dissect_scatter_kwargs(self, plot_kw):
        '''Attempt to coerce some `plot` kwargs to work with `scatter`.
        Unfortunately necessary because there are a number of arguments to
        these functions which accomplish the same thing, but are named
        slightly differently, because the functions use different artist types.
        '''

        scatter_kw = plot_kw.copy()

        # marker style kwargs

        if 'ms' in scatter_kw:
            scatter_kw['s'] = scatter_kw.pop('ms')**2
        elif 'markersize' in scatter_kw:
            scatter_kw['s'] = scatter_kw.pop('markersize')**2

        if 'mec' in scatter_kw:
            scatter_kw['ec'] = scatter_kw.pop('mec')
        elif 'markeredgecolor' in scatter_kw:
            scatter_kw['ec'] = scatter_kw.pop('markeredgecolor')

        if 'mew' in scatter_kw:
            scatter_kw['lw'] = scatter_kw.pop('mew')
        elif 'markeredgewidth' in scatter_kw:
            scatter_kw['lw'] = scatter_kw.pop('markeredgewidth')

        # bad idea to use this anyways, use full c or something like clr_param
        if 'mfc' in scatter_kw:
            scatter_kw['fc'] = scatter_kw.pop('mfc')
        elif 'markerfacecolor' in scatter_kw:
            scatter_kw['fc'] = scatter_kw.pop('markerfacecolor')

        # place scatter points very slightly above errorbars
        if 'zorder' in scatter_kw:
            scatter_kw['zorder'] += 0.0001
        else:
            scatter_kw['zorder'] = 2.0001

        return scatter_kw

    # ----------------------------------------------------------------------
    # Model Collection Visualizers
    # ----------------------------------------------------------------------

    def get_models(self, **kwargs):
        '''Return a `ModelCollection` instance corresponding to these runs.

        The visualizer collection is initialized through the
        `ModelCollection.from_chains` classmethod, with the chains from each
        run in this collection, and based on the single average model.

        Parameters
        ----------
        **kwargs : dict
            All arguments are passed to the `from_chains` classmethod, with
            `ci=False`.

        Returns
        -------
        ModelCollection
            The created model collection visualization object.
        '''

        # chains = [run.parameter_means(1)[0] for run in self.runs]
        chains = [run._get_equal_weight_chains()[1] for run in self.runs]

        obs_list = [run.obs for run in self.runs]
        ev_list = [run._evolved for run in self.runs]
        kw_list = [run._get_model_kwargs(note_flexible_BHs=True)
                   for run in self.runs]

        mc = ModelCollection.from_chains(chains, obs_list, ci=False,
                                         evolved=ev_list, model_kws=kw_list,
                                         **kwargs)

        # save a copy of models here
        self.models = mc

        return mc

    def get_CImodels(self, N=100, Nprocesses=1, add_errors=False, shuffle=True,
                     load=True):
        '''Return a CI `ModelCollection` instance corresponding to these runs.

        The visualizer collection is initialized through the
        `ModelCollection.from_chains` classmethod, with the chains from each
        run in this collection and using `N` samples, if `load` is False,
        otherwise will attempt to use the
        `ModelCollection.load` classmethod, assuming a CI model has already
        been created and saved to this same file, under the `model` group,
        in each run.

        Parameters
        ----------
        N : int, optional
            The number of samples to use in computing the confidence intervals.

        Nprocesses : int, optional
            The number of processes to use in a `multiprocessing.Pool` passed
            to the CI model initializer. Defaults to only 1 cpu.

        add_errors : bool, optional
            Optionally add the statistical and sampling errors, not normally
            accounted for, to the chain of samples used. Only relevant to
            nested sampling runs.

        shuffle : bool, optional
            Optionally shuffle the chains before passing on. Useful when `N`
            is smaller than the full sample size, to avoid biasing the
            resulting CIs.

        load : bool, optional
            If True, will attempt to load CI models, rather than creating a
            new one.

        Returns
        -------
        CIModelVisualizer
            The created model visualization (with confidence intervals) object.
        '''
        import multiprocessing

        obs_list = [run.obs for run in self.runs]
        ev_list = [run._evolved for run in self.runs]

        if load:
            filenames = [run._filename for run in self.runs]
            mc = ModelCollection.load(filenames, observations=obs_list,
                                      evolved=ev_list)

        else:
            chains = []
            kw_list = []

            for run in self.runs:
                _, ch = run._get_equal_weight_chains(add_errors=add_errors)

                if shuffle:
                    np.random.default_rng().shuffle(ch, axis=0)

                chains.append(ch)
                kw_list.append(run._get_model_kwargs(note_flexible_BHs=True))

            with multiprocessing.Pool(processes=Nprocesses) as pool:

                mc = ModelCollection.from_chains(chains, obs_list, ci=True, N=N,
                                                 pool=pool, evolved=ev_list,
                                                 model_kws=kw_list)

        # save a copy of models here
        self.models = mc

        return mc

    # ----------------------------------------------------------------------
    # Iterative plots
    # ----------------------------------------------------------------------

    def iter_plots(self, plot_func, yield_run=False, *args, **kwargs):
        '''Iterator yielding a call to `plot_func` for each run.'''
        for run in self.runs:
            fig = getattr(run, plot_func)(*args, **kwargs)

            yield (fig, run) if yield_run else fig

    def save_plots(self, plot_func, fn_pattern=None, save_kw=None, size=None,
                   remove_name=True, *args, **kwargs):
        '''Iterate over calls to `plot_func` on each run and save the figures.

        Iterates over the `iter_plots` function and saves all individual
        figures to separate files, under a custom iterative file naming schema.

        Parameters
        ----------
        plot_func : str
            The name of the plotting function called on each run.

        fn_pattern : str, optional
            A format string, which is passed the argument "cluster"
            representing each run's name, and is used to create the filename
            each figure is saved under.
            Defaults to `f'./{cluster}_{plot_func[5:]}'`.

        save_kw : dict, optional
            Optional arguments are passed to the `fig.savefig` function.

        size : 2-tuple of float, optional
            Optional resizing of the figure, using `fig.set_size_inches`.

        remove_name : bool, optional
            Remove the sometimes present cluster name placed into the
            figure's `suptitle`.

        *args, **kwargs
            All other arguments are passed to `iter_plots`.
        '''

        if fn_pattern is None:
            fn_pattern = f'./{{cluster}}_{plot_func[5:]}'

        if save_kw is None:
            save_kw = {}

        for fig, run in self.iter_plots(plot_func, True, *args, **kwargs):

            if size is not None:
                fig.set_size_inches(size)

            if remove_name:
                fig.suptitle(None)

            save_kw['fname'] = fn_pattern.format(cluster=run.name)

            fig.savefig(**save_kw)

            plt.close(fig)

    # ----------------------------------------------------------------------
    # Comparison plots
    # ----------------------------------------------------------------------

    def plot_a3_FeH(self, fig=None, ax=None, show_kroupa=False,
                    *args, **kwargs):
        '''Special case of `plot_relation` with "a3" and "FeH".'''

        fig = self.plot_relation('FeH', 'a3', fig, ax, *args, **kwargs)

        if show_kroupa:

            ax = fig.gca()

            ax.axhline(y=2.3, color='r')

            ax2 = ax.secondary_yaxis('left')

            ax2.set_yticks([2.3], [r'Kroupa ($\alpha_3=2.3$)'], c='r')

        return fig

    def plot_relation(self, param1, param2, fig=None, ax=None, *,
                      show_pearsonr=False, force_model=False,
                      annotate=False, annotate_kwargs=None,
                      clr_param=None, clr_kwargs=None, label=None, marker='o',
                      **kwargs):
        '''Plot relationship between two parameters across all runs.

        Plots a scatter plot (with errorbars) of `param1` by `param2` using the
        median and 1σ error values from each run in this collection.

        Parameters
        ----------
        param1 : str
            Name of the parameter to plot on the x-axis.

        param2 : str
            Name of the parameter to plot on the y-axis.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        show_pearsonr : bool, optional
            Optionally compute the "Pearson-r" statistic for this data and
            place it in a text box in the bottom right of the ax.

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        annotate : bool, optional
            Optionally create a hook to this figure allowing the interactive
            annotating of selected cluster names. See `_Annotator` for more
            details.

        annotate_kwargs : dict, optional
            Optional arguments passed to the `_Annotator` instance.

        clr_param : str, optional
            Defines the colour of the plotted points. If the name of a
            parameter, will colour each point by the respective value of that
            parameter in each run, otherwise will accept a single colour, or
            array of colours for each run.

        clr_kwargs : dict, optional
            Optional arguments passed to the `_add_colours` function.

        label : str, optional
            Set a label that will be displayed in the legend.

        marker : str, optional
            The marker style. See `matplotlib.markers` for more information.

        **kwargs : dict
            All other arguments are passed to `ax.errorbar` and `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)
        sc_kwargs = self._dissect_scatter_kwargs(kwargs)

        x, *dx = self._get_param(param1, force_model=force_model)
        y, *dy = self._get_param(param2, force_model=force_model)

        errbar = ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', **kwargs)
        points = ax.scatter(x, y, picker=True, marker=marker,
                            label=label, **sc_kwargs)

        ax.set_xlabel(self._get_latex_labels(param1, force_model=force_model))
        ax.set_ylabel(self._get_latex_labels(param2, force_model=force_model))

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            err_artists = itertools.chain.from_iterable(errbar[1:])

            self._add_colours(ax, points, clr_param,
                              extra_artists=err_artists, **clr_kwargs)

        elif not (kwargs.keys() & {'c', 'color'}):
            # Ensure that the points and lines are the same colour
            for ch in errbar.get_children():
                ch.set_color(points.get_facecolor())

        if annotate:

            if annotate_kwargs is None:
                annotate_kwargs = {}

            _Annotator(fig, ax, self.runs, x, y, **annotate_kwargs)

        if show_pearsonr:
            # TODO include uncertainties using (Curran, 2015) method
            from scipy.stats import pearsonr
            r, p = pearsonr(x, y)
            text = '\n'.join((fr'$\rho={r:.2f}$', fr'$p={p:.2%}$%'))
            ax.add_artist(mpl_obx.AnchoredText(text, loc='lower right'))

        return fig

    def plot_lit_comp(self, param, truths, e_truths=None, src_truths='',
                      fig=None, ax=None, *,
                      annotate=False, annotate_kwargs=None,
                      clr_param=None, clr_kwargs=None,
                      residuals=False, diagonal=True,
                      force_model=False, label=None, marker='o', **kwargs):
        '''Plot comparison between parameter values and "truths".

        Plots a scatter plot (with errorbars) of `param` by the values
        provided in the `truths` array, using the
        median and 1σ error values from each run in this collection.

        This is meant to compare one-to-one the results of fits against
        "true" values, representing the same parameter, from the literature.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        truths : np.ndarray[Nruns]
            Array of "truth" values, to plot on the y-axis.

        e_truths : np.ndarray[Nruns], optional
            Array of uncertainties on the "truth" values.

        src_truths : str, optional
            The source of the "truths", included in the y-axis label.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        annotate : bool, optional
            Optionally create a hook to this figure allowing the interactive
            annotating of selected cluster names. See `_Annotator` for more
            details.

        annotate_kwargs : dict, optional
            Optional arguments passed to the `_Annotator` instance.

        clr_param : str, optional
            Defines the colour of the plotted points. If the name of a
            parameter, will colour each point by the respective value of that
            parameter in each run, otherwise will accept a single colour, or
            array of colours for each run.

        clr_kwargs : dict, optional
            Optional arguments passed to the `_add_colours` function.

        residuals : bool, optional
            Add an ax to the bottom of the figure showing the residuals between
            the run results and the "truths".

        diagonal : bool, optional
            Include a background one-to-one line along the diagonal.

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        label : str, optional
            Set a label that will be displayed in the legend.

        marker : str, optional
            The marker style. See `matplotlib.markers` for more information.

        **kwargs : dict
            All other arguments are passed to `ax.errorbar` and `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)
        sc_kwargs = self._dissect_scatter_kwargs(kwargs)

        x, *dx = self._get_param(param, force_model=force_model)
        y, dy = truths, e_truths

        errbar = ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', **kwargs)
        points = ax.scatter(x, y, picker=True, marker=marker,
                            label=label, **sc_kwargs)

        if diagonal:
            grid_kw = {
                'color': plt.rcParams.get('grid.color'),
                'linestyle': plt.rcParams.get('grid.linestyle'),
                'linewidth': plt.rcParams.get('grid.linewidth'),
                'alpha': plt.rcParams.get('grid.alpha'),
                'zorder': 0.5
            }
            ax.axline((0, 0), (1, 1), **grid_kw)

        prm_lbl = self._get_latex_labels(param, force_model=force_model)

        ax.set_xlabel(prm_lbl)
        ax.set_ylabel(prm_lbl + (f' ({src_truths})' if src_truths else ''))

        ax.set_xlim(0.)
        ax.set_ylim(0.)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            err_artists = itertools.chain.from_iterable(errbar[1:])

            self._add_colours(ax, points, clr_param,
                              extra_artists=err_artists, **clr_kwargs)

        elif not (kwargs.keys() & {'c', 'color'}):
            # Ensure that the points and lines are the same colour
            for ch in errbar.get_children():
                ch.set_color(points.get_facecolor())

        if residuals:
            clrs = points.get_facecolors()
            res_ax = self.add_residuals(ax, x, y, dx, dy, clrs, pad=0)
            res_ax.set_xlabel(param)

        if annotate:

            if annotate_kwargs is None:
                annotate_kwargs = {}

            _Annotator(fig, ax, self.runs, x, y, **annotate_kwargs)

        return fig

    def plot_lit_relation(self, param,
                          lit, e_lit=None, param_lit='', src_lit='',
                          fig=None, ax=None, *, lit_on_x=False,
                          clr_param=None, clr_kwargs=None, residuals=False,
                          annotate=False, annotate_kwargs=None,
                          force_model=False, label=None, marker='o', **kwargs):
        '''Plot comparison between parameter values and arbitrary data.

        Plots a scatter plot (with errorbars) of `param` by the values
        provided in the `lit` array, using the
        median and 1σ error values from each run in this collection.

        This is meant to showcase relationships between the parameter results
        and other external parameter values, ostensibly from the literature.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        lit : np.ndarray[Nruns]
            Array of external data values, to plot on the y-axis.

        e_lit : np.ndarray[Nruns], optional
            Array of uncertainties on the `lit` values.

        param_lit : str, optional
            The name of the parameter represented in the `lit` values,
            included in the y-axis label.

        src_lit : str, optional
            The source of the `lit`, included in the y-axis label.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        lit_on_x : bool, optional
            Optionally flip the axes, plotting the `lit` values on the
            x-axis and the runs on the y-axis.

        clr_param : str, optional
            Defines the colour of the plotted points. If the name of a
            parameter, will colour each point by the respective value of that
            parameter in each run, otherwise will accept a single colour, or
            array of colours for each run.

        clr_kwargs : dict, optional
            Optional arguments passed to the `_add_colours` function.

        residuals : bool, optional
            Add an ax to the bottom of the figure showing the residuals between
            the run results and the "truths".

        annotate : bool, optional
            Optionally create a hook to this figure allowing the interactive
            annotating of selected cluster names. See `_Annotator` for more
            details.

        annotate_kwargs : dict, optional
            Optional arguments passed to the `_Annotator` instance.

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        label : str, optional
            Set a label that will be displayed in the legend.

        marker : str, optional
            The marker style. See `matplotlib.markers` for more information.

        **kwargs : dict
            All other arguments are passed to `ax.errorbar` and `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)
        sc_kwargs = self._dissect_scatter_kwargs(kwargs)

        x, *dx = self._get_param(param, force_model=force_model)
        y, dy = lit, e_lit

        xlabel = self._get_latex_labels(param, force_model=force_model)
        ylabel = (self._get_latex_labels(param_lit, force_model=force_model)
                  + (f' ({src_lit})' if src_lit else ''))

        # optionally flip the x and y
        if lit_on_x:
            x, y = y, x
            dx, dy = dy, dx
            xlabel, ylabel = ylabel, xlabel

        errbar = ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', **kwargs)
        points = ax.scatter(x, y, picker=True, marker=marker,
                            label=label, **sc_kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            err_artists = itertools.chain.from_iterable(errbar[1:])

            self._add_colours(ax, points, clr_param,
                              extra_artists=err_artists, **clr_kwargs)

        elif not (kwargs.keys() & {'c', 'color'}):
            # Ensure that the points and lines are the same colour
            for ch in errbar.get_children():
                ch.set_color(points.get_facecolor())

        if residuals:
            clrs = points.get_facecolors()
            res_ax = self.add_residuals(ax, x, y, dx, dy, clrs, pad=0)
            res_ax.set_xlabel(param)

        if annotate:

            if annotate_kwargs is None:
                annotate_kwargs = {}

            _Annotator(fig, ax, self.runs, x, y, **annotate_kwargs)

        return fig

    def plot_lit_dist(self, param, truths, e_truths=None, src_truths='',
                      fig=None, ax=None, *,
                      kde=True, show_normal=True, kde_color='tab:blue',
                      show_FWHM=True,
                      annotate=False, annotate_kwargs=None,
                      residuals=False, force_model=False, **kwargs):
        '''Plot hist of the fractional difference between param and "truths".

        Plots a scatter plot (with errorbars) of `param` by the values
        provided in the `truths` array, using the
        median and 1σ error values from each run in this collection.

        This is meant to compare one-to-one the results of fits against
        "true" values, representing the same parameter, from the literature.

        Plots a histogram (or smoothed KDE) of the fractional difference
        distribution between the run parameters values "true" values of the
        same parameter, from the literature.

        The fractional difference is given by
        `(param - truths) / sqrt(e_param^2 + e_truths^2)`
        which, if in perfect agreement, should resemble a Gaussian centred on
        0 with a width of 1.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        truths : np.ndarray[Nruns]
            Array of "truth" values.

        e_truths : np.ndarray[Nruns], optional
            Array of uncertainties on the "truth" values. If None,
            will be taken as all 0.

        src_truths : str, optional
            The source of the "truths", included in the y-axis label.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        kde : bool, optional
            Whether to plot a smooth Gaussian KDE, or a simple histogram.

        show_normal : bool, optional
            Optionally overplot a Gaussian centred at 0 with a width of 1,
            representing perfect agreement.

        kde_color : str, optional
            The colour of the filled KDE.

        show_FWHM : bool, optional
            If `show_normal` is True, also place in a text box the difference
            in the FWHM between the KDE and the normal plot.

        annotate : bool, optional
            Optionally create a hook to this figure allowing the interactive
            annotating of selected cluster names. See `_Annotator` for more
            details.

        annotate_kwargs : dict, optional
            Optional arguments passed to the `_Annotator` instance.

        residuals : bool, optional
            Add an ax to the bottom of the figure showing the residuals between
            the run results and the "truths".

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        **kwargs : dict
            All other arguments are passed to `ax.fill_between`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        x, *dx = self._get_param(param, with_units=False,
                                 force_model=force_model)
        dx = np.mean(dx, axis=0)
        y, dy = truths, e_truths

        if dy is None:
            dy = np.zeros_like(dx)

        if dy.ndim >= 2:
            dy = np.mean(dy, axis=0)

        frac = (x - y) / np.sqrt(dx**2 + dy**2)

        prm_lbl = self._get_latex_labels(param, with_units=False).strip('$')
        lit_lbl = (fr'{prm_lbl[:-1]},\mathrm{{lit}}}}' if '_' in prm_lbl
                   else fr'{prm_lbl}_{{\mathrm{{lit}}}}')  # tempermental
        label = (
            fr'$\frac{{{prm_lbl} - {lit_lbl}}}'
            fr'{{\sigma_{{{prm_lbl} - {lit_lbl}}}}}$'
        )
        ax.set_xlabel(label)

        # Plot a filled KDE distribution
        if kde:
            from scipy.stats import gaussian_kde, norm
            import scipy.interpolate as interp

            color = mpl_clr.to_rgb(kde_color)
            facecolor = color + (0.33, )

            # get param distributions
            domain = np.linspace(-1.1 * frac.max(), frac.max() * 1.1, 500)

            distribution = gaussian_kde(frac)(domain)

            distribution /= interp.UnivariateSpline(
                domain, distribution, k=1, s=0, ext=1
            ).integral(-np.inf, np.inf)

            ax.fill_between(domain, 0, distribution,
                            color=color, facecolor=facecolor, **kwargs)

            if show_normal:

                normal = norm.pdf(domain)
                normal /= interp.UnivariateSpline(
                    domain, distribution, k=1, s=0, ext=1
                ).integral(-np.inf, np.inf)

                ax.plot(domain, normal, 'k--')

                if show_FWHM:

                    # diff = np.sqrt(8 * np.log(2)) * (np.std(frac) - 1)
                    # text = fr'$\Delta \mathrm{{FWHM}} = {diff:.2f}$'
                    div = np.sqrt(8 * np.log(2)) * (np.std(frac) / 1)
                    text = fr'$\Delta \mathrm{{FWHM}} = {div:.2f}$'
                    ax.add_artist(mpl_obx.AnchoredText(text, loc='upper right'))

            ax.set_ylim(bottom=0)
            ax.set_xlim(domain.min(), domain.max())

        # plot a simple histogram
        else:

            ax.hist(frac, label=label, **kwargs)

        ax.set_title(src_truths)

        return fig

    # ----------------------------------------------------------------------
    # Summary plots
    # ----------------------------------------------------------------------

    def plot_param_means(self, param, fig=None, ax=None,
                         clr_param=None, clr_kwargs=None,
                         force_model=False, **kwargs):
        '''Plot the mean and 1σ values of `param` along all runs.

        Plots, with error bars, the mean values of the given parameter for
        each run, spaced equally along the x-axis, sorted by run name.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this parameter. Should be a
            part of the given `fig`.

        clr_param : str, optional
            Defines the colour of the plotted points. If the name of a
            parameter, will colour each point by the respective value of that
            parameter in each run, otherwise will accept a single colour, or
            array of colours for each run.

        clr_kwargs : dict, optional
            Optional arguments passed to the `_add_colours` function.

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        **kwargs : dict
            All other arguments are passed to `ax.errorbar` and `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''
        fig, ax = self._setup_artist(fig, ax)
        sc_kwargs = self._dissect_scatter_kwargs(kwargs)

        mean, *err = self._get_param(param, force_model=force_model)

        xticks = np.arange(len(self.runs))

        labels = self.names

        errbar = ax.errorbar(x=xticks, y=mean, yerr=err, fmt='none', **kwargs)
        points = ax.scatter(x=xticks, y=mean, picker=True, **sc_kwargs)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            err_artists = itertools.chain.from_iterable(errbar[1:])

            self._add_colours(ax, points, clr_param,
                              extra_artists=err_artists, **clr_kwargs)

        elif not (kwargs.keys() & {'c', 'color'}):
            # Ensure that the points and lines are the same colour
            for ch in errbar.get_children():
                ch.set_color(points.get_facecolor())

        ax.set_xticks(xticks, labels=labels, rotation=45,
                      ha='right', rotation_mode="anchor")

        ax.grid(visible=True, axis='x')

        ax.set_ylabel(self._get_latex_labels(param, force_model=force_model))

        return fig

    def plot_param_bar(self, param, fig=None, ax=None,
                       clr_param=None, clr_kwargs=None,
                       force_model=False, **kwargs):
        '''Plot the mean and 1σ values of `param` along all runs as a bar chart.

        Plots, with error bars, the mean values of the given parameter for
        each run as a bar chart beginning at 0.0, spaced equally along the
        x-axis, sorted by run name.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this parameter. Should be a
            part of the given `fig`.

        clr_param : str, optional
            Defines the colour of the plotted points. If the name of a
            parameter, will colour each point by the respective value of that
            parameter in each run, otherwise will accept a single colour, or
            array of colours for each run.

        clr_kwargs : dict, optional
            Optional arguments passed to the `_add_colours` function.

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        **kwargs : dict
            All other arguments are passed to `ax.bar`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''
        fig, ax = self._setup_artist(fig, ax)

        mean, *err = self._get_param(param, force_model=force_model)

        xticks = np.arange(len(self.runs))

        labels = self.names

        bars = ax.bar(x=xticks, height=mean, yerr=err, **kwargs)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            clr_kwargs.setdefault('alpha', 0.3)

            self._add_colours(ax, None, clr_param,
                              extra_artists=(bars,), **clr_kwargs)

        ax.set_xticks(xticks, labels=labels, rotation=45,
                      ha='right', rotation_mode="anchor")

        ax.set_ylabel(self._get_latex_labels(param, force_model=force_model))

        return fig

    def plot_param_violins(self, param, fig=None, ax=None,
                           clr_param=None, clr_kwargs=None,
                           color=None, alpha=0.3, edgecolor='k', edgewidth=1.0,
                           quantiles=[0.9772, 0.8413, 0.5, 0.1587, 0.0228],
                           force_model=False, **kwargs):
        '''Plot a violin plot showing the parameter distributions for all runs.

        Plots a violin plot with the full posterior distributions of a
        parameter for each run, spaced equally along the x-axis, sorted by run
        name.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this parameter. Should be a
            part of the given `fig`.

        clr_param : str, optional
            Defines the colour of the plotted points. If the name of a
            parameter, will colour each point by the respective value of that
            parameter in each run, otherwise will accept a single colour, or
            array of colours for each run.

        clr_kwargs : dict, optional
            Optional arguments passed to the `_add_colours` function.

        color : str, optional
            Fallback default (single) colour for all distributions.
            `clr_param` has precedence over this.

        alpha : float, optional
            Transparency value applied to all distributions.

        edgecolor : str, optional
            The color of the border placed around each distribution.

        edgewidth : float, optional
            The width of the border placed around each distribution.

        quantiles : list of float, optional
            The quantiles shown as ticks on the central errorbars inside the
            distributions.

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        **kwargs : dict
            All other arguments are passed to `ax.violinplot`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''
        fig, ax = self._setup_artist(fig, ax)

        chains = self._get_param_chains(param, with_units=False,
                                        force_model=force_model)

        # filter out all nans (causes violinplot to fail silently)
        chains = [ch[~np.isnan(ch)] for ch in chains]

        xticks = np.arange(len(self.runs))

        labels = self.names

        quantiles = np.array(quantiles)
        if quantiles.ndim < 2:
            quantiles = np.tile(quantiles, (len(self.runs), 1)).T

        Nquant = quantiles.shape[0]

        kwargs.setdefault('showextrema', False)

        parts = ax.violinplot(chains, positions=xticks, quantiles=quantiles,
                              **kwargs)

        # optionally draw a vert between max quantiles
        if 'cbars' not in parts and 'cquantiles' in parts:
            segs = np.array(parts['cquantiles'].get_segments())[:, 0, 1]

            mins, maxes = [], []

            for i, xi in enumerate(xticks):
                si = segs[i * Nquant:(i + 1) * Nquant]
                mins.append(si.min())
                maxes.append(si.max())

            parts['cbars'] = ax.vlines(xticks, mins, maxes)

        # handle and add colours (clr_param has precedence over color)
        if clr_param is None:
            clr_param = color  # let _add_colour handle this

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            clr_kwargs.setdefault('alpha', alpha)

            quant_arts = parts.pop('cquantiles', None)

            self._add_colours(ax, None, clr_param,
                              extra_artists=parts.values(), **clr_kwargs)

            # if Nquants > 1 have to manually repeat then add colours
            #   separately, due to how plt.LineCollection handles colours
            if quant_arts is not None:

                # Unpack colour values, to handle arrays and params here
                try:
                    clr_param, *_ = self._get_param(clr_param, with_units=False)
                except (ValueError, TypeError):
                    pass

                clr = np.repeat(clr_param, Nquant)

                self._add_colours(ax, quant_arts, clr, **clr_kwargs)

        for part in parts['bodies']:
            part.set_alpha(alpha)
            part.set_edgecolor(edgecolor)
            part.set_linewidth(edgewidth)

        ax.set_xticks(xticks, labels=labels, rotation=45,
                      ha='right', rotation_mode="anchor")

        ax.grid(visible=True, axis='x')

        ax.set_ylabel(self._get_latex_labels(param, force_model=force_model))

        return fig

    def plot_param_hist(self, param, fig=None, ax=None, kde=False,
                        force_model=False, **kwargs):
        '''Plot a histogram representing the sum of all distributions of param.

        Plots a histogram (or smoothed Gaussian KDE) representing the sum
        (or convolution) of the distributions of this parameter over all runs.

        Parameters
        ----------
        param : str
            Name of the parameter to plot.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this parameter. Should be a
            part of the given `fig`.

        kde : bool, optional
            Whether to plot a smooth Gaussian KDE, or a simple histogram.

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        **kwargs : dict
            All other arguments are passed to `ax.fill_between` or `ax.hist`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''
        # TODO is a liiittle bit invalid if chains don't all have same N

        fig, ax = self._setup_artist(fig, ax)

        chains = self._get_param_chains(param, force_model=force_model)
        chains = [ch[~np.isnan(ch)] for ch in chains]
        chains = np.concatenate(chains)

        # Plot a filled KDE distribution
        if kde:
            from scipy.stats import gaussian_kde
            import scipy.interpolate as interp

            # get param distributions
            domain = np.linspace(chains.min(), chains.max(), 500)

            distribution = gaussian_kde(chains)(domain)

            distribution /= interp.UnivariateSpline(
                domain, distribution, k=1, s=0, ext=1
            ).integral(-np.inf, np.inf)

            ax.fill_between(domain, 0, distribution, **kwargs)

            ax.set_ylim(bottom=0)

        # plot a simple histogram
        else:

            ax.hist(chains, **kwargs)

        ax.set_ylabel(self._get_latex_labels(param, force_model=force_model))

        return fig

    def plot_param_corner(self, params=None, fig=None, *,
                          include_FeH=True, include_BH=False, include_rt=False,
                          log_radii=False, force_model=False, **kwargs):
        '''Plot a "corner plot" showing relationship between parameters.

        Plots a Nparam-Nparam lower-triangular "corner" plot showing the mean
        and 1σ values for all parameters for all runs.

        Parameters
        ----------
        params : list of str, optional
            The list of parameters to plot. If not given, defaults to the
            typical 13 free model parameters, and any included by the
            `include_*` arguments.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_RunAnalysis._setup_multi_artist` for more details.

        include_FeH : bool, optional
            If True, the metallicity `FeH` is included in the default params.

        include_BH : bool, optional
            If True, the black hole mass `BH_mass` is included in the
            default params.

        include_rt : bool, optional
            If True, the tidal radius `rt` is included in the default params.

        log_radii : bool, optional
            If True, the radii included in the default params are logged.

        force_model : bool, optional
            Force these parameter values to be taken from model quantities.
            Can be useful when some parameter names overlap (e.g. "ra").

        **kwargs : dict
            All other arguments are passed to `plot_relation`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        if params is None:
            params = ['W0', 'M', 'log_rh' if log_radii else 'rh', 'ra', 'g',
                      'delta', 's2', 'F', 'a1', 'a2', 'a3', 'BHret', 'd']

            if include_FeH:
                params += ['FeH']

            if include_BH:
                params += ['BH_mass']

            if include_rt:
                params += ['log_rt' if log_radii else 'rt']

        # setup axes
        Nparams = len(params)
        Nrows = Ncols = Nparams - 1

        # TODO redo this using the subplot_mosaic logic to make alot easier
        fig, axes = self._setup_multi_artist(fig, (Nrows, Ncols),
                                             constrained_layout=False,
                                             sharex='col', sharey='row')
        axes = axes.reshape((Nrows, Ncols))

        # TODO these are not ideal, lots of conflicting labels and ticks
        # Setup axis layout (from `corner`).
        factor = 2.0  # size of side of one panel
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
        whspace = 0.05  # size of width/height margin
        plotdim = factor * (Nrows - 1) + factor * (Ncols - 2.) * whspace
        dim = lbdim + plotdim + trdim  # total size

        # Format figure.
        lb = lbdim / dim
        tr = (lbdim + plotdim) / dim
        fig.subplots_adjust(left=lb,
                            bottom=lb,
                            right=tr,
                            top=tr,
                            wspace=whspace,
                            hspace=whspace)

        for i, py in enumerate(params[1:]):

            for j, px in enumerate(params[:-1]):

                ax = axes[i, j]

                if j > i:
                    ax.remove()
                    continue

                else:

                    self.plot_relation(px, py, fig=fig, ax=ax,
                                       force_model=force_model, **kwargs)

                # set labels on bottom row
                if i + 1 == Nrows:
                    xlabel = self._get_latex_labels(px, force_model=force_model)
                    # rotate_ticks(ax, 'x')
                    ax.set_xlabel(xlabel)
                    # ax.xaxis.set_label_coords(0.5, -0.3)
                else:
                    ax.set_xlabel('')

                # Set labels on leftmost col
                if j == 0:
                    ylabel = self._get_latex_labels(py, force_model=force_model)
                    # rotate_ticks(ax, 'y')
                    ax.set_ylabel(ylabel)
                    # ax.yaxis.set_label_coords(-0.3, 0.5)
                else:
                    ax.set_ylabel('')

        return fig

    def summary_dataframe(self, *, params='all',
                          include_FeH=True, include_BH=False,
                          math_labels=False):
        '''Return a Dataframe with the median and 1σ param values for all runs.

        Constructs and returns a table (in the form of a `pandas.Dataframe`)
        of the median and 1σ values of all included parameters, with each row
        representing a run.

        Parameters
        ----------
        params : list of str, optional
            The list of parameters to include in the returned table.
            If not given, defaults to the typical 13 free model parameters,
            and any included by the `include_*` arguments.

        include_FeH : bool, optional
            If True, the metallicity `FeH` is included in the default params.

        include_BH : bool, optional
            If True, the black hole related parameters (`BH_mass`, `BH_num`,
            `f_BH`, `f_rem`) are included in the default params.

        math_labels : bool, optional
            If True, the column names will be labelled with latex math.
            See `_get_latex_labels` for more information.

        Returns
        -------
        pandas.Dataframe
            The full table of parameter values.
        '''
        import pandas as pd
        # TODO pandas isn't in the setup requirements

        # Get name of all desired parameters

        if params == 'all':
            labels = self.runs[0]._get_labels(label_fixed=False)

        else:
            labels = params

        if include_FeH:
            labels = ['FeH'] + labels

        if include_BH:
            labels += ['BH_mass', 'BH_num', 'f_BH', 'f_rem']

        # Fill in a dictionary of column data

        data = {}

        data['Cluster'] = [run.name for run in self.runs]

        for param in labels:

            name = self._get_latex_labels(param)[1:] if math_labels else param
            sig = ((r'$-1\sigma\_', r'$+1\sigma\_') if math_labels
                   else ('-1σ_', '+1σ_'))

            median, σ_down, σ_up = self._get_param(param)
            data[f'{"$" if math_labels else ""}{name}'] = median
            data[f'{sig[0]}{name}'], data[f'{sig[1]}{name}'] = σ_down, σ_up

        # Create dataframe

        return pd.DataFrame.from_dict(data)

    def output_summary(self, outfile=sys.stdout, params='all', style='latex', *,
                       include_FeH=False, include_BH=False, math_labels=False,
                       substack_errors=False, **kwargs):
        '''Output a table of the median and 1σ param values for all runs.

        Constructs and writes out a table of the median and 1σ values of all
        included parameters, with each row representing a run.

        Parameters
        ----------
        outfile : file, optional
            Output file handler to write the summary to. Defaults to printing
            to "stdout".

        params : list of str, optional
            The list of parameters to include in the outputted table.
            If not given, defaults to the typical 13 free model parameters,
            and any included by the `include_*` arguments.

        style : {'table', 'latex', 'hdf', 'csv', 'html'}, optional
            Type of output file to create. Defaults to 'latex'. Most formats
            use the corresponding `to_*` method on a `pandas.Dataframe`.

        include_FeH : bool, optional
            If True, the metallicity `FeH` is included in the default params.

        include_BH : bool, optional
            If True, the black hole related parameters (`BH_mass`, `BH_num`,
            `f_BH`, `f_rem`) are included in the default params.

        math_labels : bool, optional
            If True, the column names will be labelled with latex math.
            See `_get_latex_labels` for more information.

        substack_errors : bool, optional
            If True, and the given style is `latex`, the errors will be
            written, not as a separate column, but within a "substack" on
            each value.

        **kwargs : dict
            All other arguments are passed to the corresponding
            `to_*` method on a `pandas.Dataframe`.

        Returns
        -------
        pandas.Dataframe
            The dataframe used to write the output file.
        '''

        def _round_sf(*values, max_prec=7):
            import decimal

            # get Decimal representations of each value
            decs = [decimal.Decimal(fi) for fi in values]

            # determine the smallest precision
            try:
                pos = min([di.adjusted() for di in decs if di != 0.])
            except ValueError:
                # catch if they're all zero
                pos = -np.inf

            # limit it to max_prec
            pos = max(pos, -max_prec)

            # get pos in terms of a fixed 10**pos
            exp = decimal.Decimal((0, (1,), pos))

            return [str(di.quantize(exp)) for di in decs]

        # get dataframe

        df = self.summary_dataframe(params=params,
                                    include_FeH=include_FeH,
                                    include_BH=include_BH,
                                    math_labels=math_labels)

        # output in desired format

        kwargs.setdefault('index', False)

        if style in ('table', 'dat'):
            df.to_string(buf=outfile, **kwargs)

        elif style == 'latex':
            if substack_errors:
                # have to manually change to substack before outputting

                # get only the actual values columns
                for prm in df.columns[1::3]:

                    # find the corresponding errors
                    errnames = (
                        (fr'$-1\sigma\_{prm[1:]}', fr'$+1\sigma\_{prm[1:]}')
                        if math_labels else (f'-1σ_{prm}', f'+1σ_{prm}')
                    )

                    errs = (df[errnames[0]], df[errnames[1]])

                    # rewrite the column with substack errors
                    # TODO if include_FeH or any other with 0 err, will truncate
                    sub = []
                    for row, val in enumerate(df[prm]):
                        v, eu, ed = _round_sf(val, errs[1][row], errs[0][row])
                        sub.append(fr'\({v}\substack{{+{eu} \\ -{ed}}}\)')

                    df[prm] = sub

                    # delete the error columns
                    del df[errnames[0]], df[errnames[1]]

            kwargs.setdefault('escape', False)
            kwargs.setdefault('float_format', '%.4f')
            df.to_latex(buf=outfile, **kwargs)

        elif style == 'hdf':
            df.to_hdf(outfile, **kwargs)

        elif style == 'csv':
            df.to_csv(outfile, **kwargs)

        elif style == 'html':
            df.to_html(buf=outfile, **kwargs)

        else:
            mssg = ("Invalid style. Must be one of "
                    "{'table', 'latex', 'hdf', 'csv', 'html'}")
            raise ValueError(mssg)

        return df
