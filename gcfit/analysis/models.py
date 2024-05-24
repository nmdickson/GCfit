from .. import util
from ..util import mass
from ..core.data import Observations, FittableModel

import h5py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr
import matplotlib.legend as mpl_leg
import matplotlib.ticker as mpl_tick
import astropy.visualization as astroviz

import logging
import pathlib
from collections import abc


__all__ = ['ModelVisualizer', 'CIModelVisualizer', 'ObservationsVisualizer',
           'SampledVisualizer', 'ModelCollection']


def _get_model(theta, observations):
    '''Compute a model based on `theta` and fail quietly'''
    try:
        return FittableModel(theta, observations=observations)
    except ValueError:
        logging.warning(f"Model did not converge with {theta=}")
        return None

# --------------------------------------------------------------------------
# Individual model visualizers
# --------------------------------------------------------------------------


class _ClusterVisualizer:
    '''Base class for all visualizers of all Model types.'''

    _MARKERS = ('o', '^', 'D', '+', 'x', '*', 's', 'p', 'h', 'v', '1', '2')

    # Default xaxis limits for all profiles. Set by inits, can be reset by user
    rlims = None

    _cmap = plt.cm.jet

    @property
    def cmap(self):
        return plt.cm.get_cmap(self._cmap)

    @cmap.setter
    def cmap(self, cm):
        if isinstance(cm, mpl_clr.Colormap) or (cm in plt.colormaps()):
            self._cmap = cm
        elif cm is None:
            self._cmap = plt.rcParams['image.cmap']
        else:
            mssg = f"{cm} is not a registered colormap, see `plt.colormaps`"
            raise ValueError(mssg)

    # -----------------------------------------------------------------------
    # Artist setups
    # -----------------------------------------------------------------------

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
        '''setup a figure with multiple axes, using subplots or subfigures

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

        **subkw : dict, optional
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

    def _set_ylabel(self, ax, label, unit=None, label_position='left', *,
                    residual_ax=None, inline_latex=True):
        '''Set the label for the quantity on the y axis of this plot.'''

        tick_prms = dict(which='both',
                         labelright=(label_position == 'right'),
                         labelleft=(label_position == 'left'))

        if label_position == 'top':

            ax.set_ylabel('')

            try:
                ax.set_title(label)
            except AttributeError as err:
                try:
                    ax._parent.set_title(label)
                except AttributeError:
                    raise err

        else:

            if unit is not None:

                # try to get an automatically formatted unit string if possible
                try:
                    if not isinstance(unit, u.Unit):
                        unit = u.Unit(unit)

                    if unit is not u.dimensionless_unscaled:

                        fmt = "latex_inline" if inline_latex else "latex"
                        fmtunit = unit.to_string(fmt)

                        label += fr' $\left[{fmtunit.strip("$")}\right]$'

                except ValueError:
                    pass

            ax.set_ylabel(label)

            ax.yaxis.set_label_position(label_position)
            ax.yaxis.set_tick_params(**tick_prms)

            if residual_ax is not None:

                residual_ax.yaxis.set_label_position(label_position)
                residual_ax.yaxis.set_tick_params(**tick_prms)
                residual_ax.yaxis.set_ticks_position('both')

        ax.yaxis.set_ticks_position('both')

    def _set_xlabel(self, ax, label='Distance from centre', unit=None, *,
                    residual_ax=None, remove_all=False, inline_latex=True):
        '''Set the label for the quantity on the x axis of this plot.'''

        bottom_ax = ax if residual_ax is None else residual_ax

        if unit is not None:

            # try to get an automatically formatted unit string if possible
            try:
                if not isinstance(unit, u.Unit):
                    unit = u.Unit(unit)

                if unit is not u.dimensionless_unscaled:

                    fmt = "latex_inline" if inline_latex else "latex"
                    fmtunit = unit.to_string(fmt)

                    label += fr' $\left[{fmtunit.strip("$")}\right]$'

            except ValueError:
                pass

        bottom_ax.set_xlabel(label)

        # if has residual ax, remove the ticks/labels on the top ax
        if residual_ax is not None:
            ax.set_xlabel('')
            ax.xaxis.set_tick_params(bottom=False, labelbottom=False)

        # if desired, simply remove everything
        if remove_all:
            bottom_ax.set_xlabel('')
            bottom_ax.xaxis.set_tick_params(bottom=False, labelbottom=False)

    # -----------------------------------------------------------------------
    # Unit support
    # -----------------------------------------------------------------------

    def _support_units(method):
        '''Decorator adding units support to any wrapped plotting function.'''
        import functools

        @functools.wraps(method)
        def _unit_decorator(self, *args, **kwargs):

            # convert based on median distance parameter
            eqvs = util.angular_width(self.d)

            with astroviz.quantity_support(), u.set_enabled_equivalencies(eqvs):
                return method(self, *args, **kwargs)

        return _unit_decorator

    # -----------------------------------------------------------------------
    # Plotting functionality
    # -----------------------------------------------------------------------

    def _get_median(self, percs):
        '''From an array of data percentiles, return the median array.'''
        return percs[percs.shape[0] // 2] if percs.ndim > 1 else percs

    def _get_err(self, dataset, key):
        '''Gather the error variables corresponding to `key` from `dataset`.'''
        try:
            return dataset[f'Δ{key}']
        except KeyError:
            try:
                return np.c_[dataset[f'Δ{key},down'], dataset[f'Δ{key},up']].T
            except KeyError:
                return None

    def _plot_model(self, ax, data, intervals=None, *,
                    x_data=None, x_unit='pc', y_unit=None,
                    scale=1.0, background=0.0,
                    CI_kwargs=None, **kwargs):
        '''Base plotting function for all model profiles.

        Plots a given array of y-values representing a radial profile
        from a given model, on `ax`, as a solid connected line.
        If `data` is a higher dimensional array, will overplot a desired number
        of confidence intervals.

        This base function is to be used by `plot_profile` to handle the
        plotting of all model quantities.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            An axes instance on which to plot this model profile.

        data : u.Quantity[Ninterval, Nr]
            Array of model profile data to plot. Final dimension should be
            the same size as `self.r`. If two dimensional, the first dimension
            must be an odd size, with the median profile in the middle row
            and each other an extra σ away.

        intervals : int, optional
            The number of σ confidence intervals to plot (e.g. intervals=2 will
            plot the median, 1σ and 2σ). The first dimension of `data` must
            be of size 1 + 2(intervals).

        x_data : u.Quantity[Nr], optional
            Alternative x-axis data to use instead of `self.r`.

        x_unit, y_unit : u.Unit, optional
            Units to convert the x and y data to. By default, x-units are in
            parsecs and y-units are whatever units `data` is passed in with.

        scale : float, optional
            An optional y-scaling factor applied to the `data`.

        background : float, optional
            An optional background level to be subtracted from the `data`.

        CI_kwargs : dict, optional
            Optional arguments passed to `ax.fill_between` when overplotting
            confidence intervals.

        **kwargs : dict
            All other arguments are passed to `ax.plot` when plotting median
            profile.
        '''

        CI_kwargs = dict() if CI_kwargs is None else CI_kwargs

        # ------------------------------------------------------------------
        # Evaluate the shape of the data array to determine confidence
        # intervals, if applicable
        # ------------------------------------------------------------------

        if data is None or data.ndim == 0:
            return

        elif data.ndim == 1:
            data = data.reshape((1, data.size))

        if not (data.shape[0] % 2):
            mssg = 'Invalid `data`, must have odd-numbered zeroth axis shape'
            raise ValueError(mssg)

        midpoint = data.shape[0] // 2

        if intervals is None:
            intervals = midpoint

        elif intervals > midpoint:
            mssg = f'{intervals}σ is outside stored range of {midpoint}σ'
            raise ValueError(mssg)

        # ------------------------------------------------------------------
        # Convert any units desired
        # ------------------------------------------------------------------

        data *= scale

        x_domain = self.r if x_data is None else x_data

        if x_unit:
            x_domain = x_domain.to(x_unit)

        if y_unit:
            data = data.to(y_unit)

        # ------------------------------------------------------------------
        # Subtract background value
        # ------------------------------------------------------------------

        data -= (background << data.unit)

        # ------------------------------------------------------------------
        # Plot the median (assumed to be the middle axis of the intervals)
        # ------------------------------------------------------------------

        median = data[midpoint]

        med_plot, = ax.plot(x_domain, median, **kwargs)

        # ------------------------------------------------------------------
        # Plot confidence intervals successively from the midpoint
        # ------------------------------------------------------------------

        output = [med_plot]

        CI_kwargs.setdefault('color', med_plot.get_color())

        alpha = 0.8 / (intervals + 1)
        for sigma in range(1, intervals + 1):

            CI = ax.fill_between(
                x_domain, data[midpoint + sigma], data[midpoint - sigma],
                alpha=(1 - alpha), **CI_kwargs
            )

            output.append(CI)

            alpha += alpha

        return output

    def _plot_data(self, ax, dataset, y_key, *,
                   x_key='r', x_unit='pc', y_unit=None,
                   err_transform=None, scale=1.0, background=0.0, **kwargs):
        '''Base plotting function for all observed data profiles.

        Plots the observational data corrseponding to the variable `y_key`
        from a given dataset, with correspoonding errorbars (if they exist).

        This base function is to be used by `plot_profile` to handle the
        plotting of all observational data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            An axes instance on which to plot this observational data.

        dataset : gcfit.data.Dataset
            The observational dataset to load the variable and errors from.

        y_key : str
            The variable name of the observational data to plot.
            Must exist within `dataset`.

        x_key : str, optional
            Alternative x-axis variable name to use instead of `dataset['r']`.
            Must exist within `dataset`.

        x_unit, y_unit : u.Unit, optional
            Units to convert the x and y data to. By default, x-units are in
            parsecs and y-units are in the stored data units.

        err_transform : callable, optional
            An optional transformation function which, if given, will be
            called on the y-errors before plotting. Must accept an array of
            error values and return an array of the same size.

        scale : float, optional
            An optional y-scaling factor applied to the variable data.

        background : float, optional
            An optional background level to be subtracted from the variable
            data.

        **kwargs : dict
            All other arguments are passed to `ax.errorbar`.

        Returns
        -------
        matplotlib.ErrorbarContainer
            The outputs from the call to `plt.errorbars` using the given
            dataset.
        '''

        # ------------------------------------------------------------------
        # Get data and relevant errors for plotting
        # ------------------------------------------------------------------

        xdata = dataset[x_key]
        ydata = dataset[y_key] * scale

        xerr = self._get_err(dataset, x_key)
        yerr = self._get_err(dataset, y_key) * scale

        # ------------------------------------------------------------------
        # Convert any units desired
        # ------------------------------------------------------------------

        if x_unit is not None:
            xdata = xdata.to(x_unit)

        if y_unit is not None:
            ydata = ydata.to(y_unit)

        # ------------------------------------------------------------------
        # Subtract background value
        # ------------------------------------------------------------------

        ydata -= (background << ydata.unit)

        # ------------------------------------------------------------------
        # If given, transform errors based on `err_transform` function
        # ------------------------------------------------------------------

        if err_transform is not None:
            yerr = err_transform(yerr)

        # ------------------------------------------------------------------
        # Setup default plotting details, style, labels
        # ------------------------------------------------------------------

        kwargs.setdefault('mfc', None)
        kwargs.setdefault('mec', 'k')
        kwargs.setdefault('mew', 0.3)
        kwargs.setdefault('linestyle', 'None')
        kwargs.setdefault('zorder', 10)  # to force marker and bar to be higher

        label = dataset.cite()
        if 'm' in dataset.mdata:
            label += fr' ($m={dataset.mdata["m"]}\ M_\odot$)'

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------

        return ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr,
                           label=label, **kwargs)

    def _plot_profile(self, ax, ds_pattern, y_key, model_data, *,
                      y_unit=None, residuals=False, legend=False,
                      color=None, data_color=None, model_color=None,
                      mass_bins=None, model_label=None, label_masses=True,
                      res_kwargs=None, data_kwargs=None, model_kwargs=None,
                      **kwargs):
        '''Base plotting function for all profiles.

        Base function handling the plotting of all profile-based (i.e. `self.r`
        on the x-axis) model plots. This includes both the full model profiles
        and observational datasets.
        This function determines what needs to be plotted, and calls the
        `_plot_model` and `_plot_data` functions as necessary.

        Model data for all datasets matching `ds_pattern` will be plotted
        separately with different markers.

        Each mass (both model and data) will be plotted  with it's own colour,
        as decided by the usual matplotlib colour cycle (color=None),
        unless `data_color`, `model_color` or `color` are supplied, in which
        case they will take precedence (in that order).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            An axes instance on which to plot this observational data.

        ds_pattern : str
            A pattern string to filter all dataset names on, using `fnmatch`.

        y_key : str
            The variable name of the observational data to plot.
            Must exist within each filtered dataset.

        model_data : u.Quantity[Nmasses, Ninterval, Nr]
            Array of model profile data to plot. Final dimension should be
            the same size as `self.r`.

        y_unit : u.Unit, optional
            Units to convert the y axis to.

        residuals : bool, optional
            If True, automatically adds a residuals axis including all
            plotted datasets, using `_add_residuals`.

        res_kwargs, data_kwargs, model_kwargs : dict, optional
            Optional arguments passed to `_add_residuals`, `_plot_data`, and
            `_plot_model`, respectively.

        color : str, optional
            Fallback default (single) colour for all plots.
            By default, will utilize the internal matplotlib colour cycler,
            which can be customized using `plt.rcParams["axes.prop_cycle"]`.
            Both `data_color` and `model_color` have precedence over this.

        data_color : str or list of str, optional
            The colour passed to `_plot_data`, used to colour all observational
            data (markers and errorbars). If a list of colours is given, it
            must match (or be longer than) the number of datasets found by
            the given pattern.

        model_color : str, optional
            The colour passed to `_plot_model`, used to colour all model
            profiles (transparency added to colour in confidence intervals).
            If a colour is *not* given, and multiple masses have been plotted
            (either through the plotted data or given `mass_bins`), then the
            colours of model profiles will be taken from the corresponding data
            of the same masses.

        mass_bins : list of int, optional
            The mass bins from the model to plot. This should be a list of
            indices corresponding to the desired mass bins in the given
            `model_data` (i.e. along axis 0).
            Note that not all subclasses store the full range of modelled mass
            bins, and this argument may not be functional.
            By default, plots the mass bin given by `self.star_bin`. This is
            also the bin assumed to represent all plotted data, if there is
            no tracer mass stored.

        model_label : str, optional
            An optional label to be given to each plotted model, to appear in
            a legend.
            Defaults to no label, unless `label_masses` is True, in which case
            the default prefix is simply "Model ".

        label_masses : bool, optional
            If True, applies a postfix to all model labels indicating their
            masses. Note that this may only function if `mass_bins` are
            explicitly given. If you want this to apply when only using the
            single default mass bin, must also supply
            `mass_bins=[self.star_bin]`.

        **kwargs : dict
            All other arguments are passed to both `_plot_data` and
            `_plot_model`. Arguments must be valid for both.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The provided axes instance, with all profiles and data plotted on
            it.

        res_ax : None or matplotlib.axes.Axes
            None, if `residuals` is False, otherwise the automatically created
            axes instance containing the residuals plot.
        '''

        ds_pattern = ds_pattern or ''

        strict = kwargs.pop('strict', False)

        # Restart marker styles each plotting call
        markers = iter(self._MARKERS)

        if res_kwargs is None:
            res_kwargs = {}

        if data_kwargs is None:
            data_kwargs = {}

        if model_kwargs is None:
            model_kwargs = {}

        # Unless specified, each mass bin should cycle colour from matplotlib
        default_color = color

        data_color = data_color or default_color
        model_color = model_color or default_color

        # if mass bins are supplied, label all models, for use in legends
        if label_masses:
            label_masses = bool(mass_bins)

        if label_masses and model_label is None:
            model_label = 'Model'

        # ------------------------------------------------------------------
        # Determine the relevant datasets to the given pattern
        # ------------------------------------------------------------------

        # If this cluster has no observations, force basically `show_obs=False`
        if self.obs is None:
            ds_pattern, datasets = '', {}

        else:
            # TODO optionally exclude any "excluded_datasets"?
            # TODO since this is not sorted, hard to pass in list of spec colors
            datasets = self.obs.filter_datasets(ds_pattern)

        if strict and ds_pattern and not datasets:
            mssg = (f"No datasets matching '{ds_pattern}' exist in {self.obs}."
                    f"To plot models without data, set `show_obs=False`")
            # raise DataError
            raise KeyError(mssg)

        # ------------------------------------------------------------------
        # Iterate over the datasets, keeping track of all relevant masses
        # and calling `_plot_data`
        # ------------------------------------------------------------------

        masses = {}

        # Allow passing list of data colours to override the mass-colour match
        if not (list_of_clr := (isinstance(data_color, abc.Collection)
                                and not isinstance(data_color, str))):
            data_color = [data_color, ] * len(datasets)

        for i, (key, dset) in enumerate(datasets.items()):

            mrk = next(markers)

            # get mass bin of this dataset, for later model plotting
            if 'm' in dset.mdata:
                m = dset.mdata['m'] * u.Msun
                mass_bin = np.where(self.mj == m)[0][0]
            else:
                mass_bin = self.star_bin

            if mass_bin in masses and not list_of_clr:
                clr = masses[mass_bin][0][0].get_color()
            else:
                clr = data_color[i]

            # plot the data
            try:
                line = self._plot_data(ax, dset, y_key, marker=mrk, color=clr,
                                       y_unit=y_unit, **data_kwargs, **kwargs)

            except KeyError as err:
                if strict:
                    raise err
                else:
                    # warnings.warn(err.args[0])
                    continue

            masses.setdefault(mass_bin, [])

            masses[mass_bin].append(line)

        # ------------------------------------------------------------------
        # Based on the masses of data plotted, plot the corresponding axes of
        # the model data, calling `_plot_model`
        # ------------------------------------------------------------------

        res_ax = None

        if model_data is not None:

            # ensure that the data is (mass bin, intervals, r domain)
            if len(model_data.shape) != 3:
                raise ValueError("invalid model data shape")

            # Apply custom supplied mass_bins
            if mass_bins:
                masses = dict.fromkeys(mass_bins, None) | masses

            # No data plotted and no mass bins supplied, default to star_bin
            if not masses:
                masses = {self.star_bin: None}

            for mbin, errbars in masses.items():

                try:
                    ymodel = model_data[mbin, :, :]
                except IndexError:
                    mssg = (f"Mass bin index {mbin} is out of "
                            f"range (0-{self.mj.size - 1})")
                    raise ValueError(mssg)

                # if no model color specified *and* multiple masses exists, use
                #   corresponding data colours, otherwise use default
                if (model_color is None and errbars is not None
                        and len(masses) > 1):
                    clr = errbars[0][0].get_color()
                else:
                    clr = model_color

                label = model_label
                if label_masses:
                    label += fr' ($m={self.mj[mbin].value:.2f}\ M_\odot$)'

                self._plot_model(ax, ymodel, color=clr, y_unit=y_unit,
                                 label=label, **model_kwargs, **kwargs)

                if residuals:
                    res_ax = self._add_residuals(ax, ymodel, errbars,
                                                 res_ax=res_ax, y_unit=y_unit,
                                                 **res_kwargs)

        # Adjust x limits
        if self.rlims is not None:
            ax.set_xlim(*self.rlims)

        # Create legend (remove if no handles found)
        if legend:

            # The empty legends warning is apparently impossible to catch, so
            # we'll do it ourselves, and save the stdout clutter
            handles, labels = mpl_leg._get_legend_handles_labels([ax])

            if len(handles) > 0:
                ax.legend(handles, labels)

        return ax, res_ax

    # -----------------------------------------------------------------------
    # Plot extras
    # -----------------------------------------------------------------------

    def _add_residuals(self, ax, ymodel, errorbars, percentage=False, *,
                       show_chi2=False, xmodel=None, y_unit=None, size="15%",
                       res_ax=None, divider_kwargs=None):
        '''Append an extra axis to `ax` for plotting residuals.

        Automatically appends a new axis to the the bottom of the given `ax`,
        and plots the residuals between the given model and plotted data on it.

        The models will be plotted with the median along the y=0 axis, and the
        data values will attempt to be read from the output of plotted
        errorbars. Honestly can't guarantee this will be correct if you use
        weird transformed axes units/scales.
        The styling of the data will attempt to match the primary plot as
        closely as possible.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            An axes instance on which to plot this observational data.

        ymodel : u.Quantity[Ninterval, Nr]
            Array of model profile data to plot. Must be equivalent to
            that plotted on `ax` using `_plot_model`.

        errorbars : matplotlib.ErrorbarContainer
            The outputs from a call to `plt.errorbars`, as returned from
            `_plot_data`.

        percentage : bool, optional
            Whether to plot the residuals in physical units (model - data;
            default) or in percentages (100 * (model - data) / data).

        show_chi2 : bool, optional
            If True, will add an annotation with the computed Chi-squared
            value between all data and the plotted model median.

        x_model : u.Quantity[Nr], optional
            Alternative x-axis data to use instead of `self.r`.

        y_unit : u.Unit, optional
            Units to convert the y axis to.

        size : str or float, optional
            The size of the appended residuals axes, with respect to the
            primary axes.
            See `mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`
            for more information. Defaults to "15%".

        res_ax : matplotlib.axes.Axes, optional
            Optionally provide an already created axis to plot residuals on.
            This is useful for overplotting multiple residuals (i.e. for
            multiple datasets).

        Returns
        -------
        matplotlib.axes.Axes
            The created axes instance containing the residuals plot.
        '''
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if errorbars is None:
            errorbars = []

        if divider_kwargs is None:
            divider_kwargs = {}

        # ------------------------------------------------------------------
        # Get model data and spline
        # ------------------------------------------------------------------

        if xmodel is None:
            xmodel = self.r

        if y_unit is not None:
            ymodel = ymodel.to(y_unit)

        ymedian = self._get_median(ymodel)

        yspline = util.QuantitySpline(xmodel, ymedian)

        # ------------------------------------------------------------------
        # Setup axes, adding a new smaller axe for the residual underneath,
        # if it hasn't already been created (and passed to `res_ax`)
        # ------------------------------------------------------------------

        if res_ax is None:

            divider = make_axes_locatable(ax)
            res_ax = divider.append_axes('bottom', size=size, pad=0, sharex=ax)

            res_ax.grid()

            res_ax.set_xscale(ax.get_xscale())

            res_ax.spines['top'].set(**divider_kwargs)

        # ------------------------------------------------------------------
        # Plot the model line, hopefully centred on zero
        # ------------------------------------------------------------------

        if percentage:
            baseline = 100 * (ymodel - ymedian) / ymodel
        else:
            baseline = ymodel - ymedian

        self._plot_model(res_ax, baseline, color='k')

        # ------------------------------------------------------------------
        # Get data from the plotted errorbars
        # ------------------------------------------------------------------

        chi2 = 0.

        for errbar in errorbars:

            # --------------------------------------------------------------
            # Get the actual datapoints, and the hopefully correct units
            # --------------------------------------------------------------

            xdata, ydata = errbar[0].get_data()
            ydata = ydata.to(ymedian.unit)

            # --------------------------------------------------------------
            # Grab relevant formatting (colours and markers)
            # --------------------------------------------------------------

            # unfortunately `artist.update_from` refuses to work here
            mfc = errbar[0].get_mfc()
            mec = errbar[0].get_mec()
            mew = errbar[0].get_mew()
            ms = errbar[0].get_ms()
            mrk = errbar[0].get_marker()

            # --------------------------------------------------------------
            # Parse the errors from the size of the errorbar lines (messy)
            # --------------------------------------------------------------

            xerr = yerr = None

            if errbar.has_xerr:
                xerr_lines = errbar[2][0]
                yerr_lines = errbar[2][1] if errbar.has_yerr else None
            elif errbar.has_yerr:
                xerr_lines, yerr_lines = None, errbar[2][0]
            else:
                xerr_lines = yerr_lines = None

            if xerr_lines:

                xerr_segs = xerr_lines.get_segments() << xdata.unit

                xerr = u.Quantity([np.abs(seg[:, 0] - xdata[i])
                                   for i, seg in enumerate(xerr_segs)]).T

            if yerr_lines:

                yerr_segs = yerr_lines.get_segments() << ydata.unit

                if percentage:
                    yerr = 100 * np.array([
                        np.abs(seg[:, 1] - ydata[i]) / ydata[i]
                        for i, seg in enumerate(yerr_segs)
                    ]).T

                else:
                    yerr = u.Quantity([np.abs(seg[:, 1] - ydata[i])
                                       for i, seg in enumerate(yerr_segs)]).T

            # --------------------------------------------------------------
            # Compute the residuals and plot them
            # --------------------------------------------------------------

            if percentage:
                res = 100 * (ydata - yspline(xdata)) / yspline(xdata)
            else:
                res = ydata - yspline(xdata)

            res_ax.errorbar(xdata, res, xerr=xerr, yerr=yerr,
                            color=mfc, mec=mec, mew=mew, marker=mrk, ms=ms,
                            linestyle='none')

            # --------------------------------------------------------------
            # Optionally compute chi-squared statistic
            # --------------------------------------------------------------

            if show_chi2:
                chi2 += np.sum((res / yerr)**2)

        if show_chi2:
            fake = plt.Line2D([], [], label=fr"$\chi^2={chi2:.2f}$")
            res_ax.legend(handles=[fake], handlelength=0, handletextpad=0)

        # ------------------------------------------------------------------
        # Label y-axes
        # ------------------------------------------------------------------

        if percentage:
            res_ax.set_ylabel(r'Residuals')
            res_ax.yaxis.set_major_formatter(mpl_tick.PercentFormatter())
        else:
            res_ax.set_ylabel(f'Residuals [{res_ax.get_ylabel()}]')

        # ------------------------------------------------------------------
        # Set bounds at 100% or less
        # ------------------------------------------------------------------

        if percentage:
            ylim = res_ax.get_ylim()
            res_ax.set_ylim(max(ylim[0], -100), min(ylim[1], 100))

        return res_ax

    def _add_hyperparam(self, ax, ymodel, xdata, ydata, yerr):
        '''Unused. Supposed to add the value of bayesian hyperparameters'''
        # TODO this is still a complete mess
        #   numerically correct, just need better way to track and place them.

        yspline = util.QuantitySpline(self.r, ymodel)

        if hasattr(ax, 'aeff_text'):
            aeff_str = ax.aeff_text.get_text()
            aeff = float(aeff_str[aeff_str.rfind('$') + 1:])

        else:
            # TODO figure out best place to place this at
            ax.aeff_text = ax.text(0.1, 0.3, '')
            aeff = 0.

        aeff += util.hyperparam_effective(ydata, yspline(xdata), yerr)

        ax.aeff_text.set_text(fr'$\alpha_{{eff}}=${aeff:.4e}')

    # -----------------------------------------------------------------------
    # Observables plotting
    # -----------------------------------------------------------------------

    @_support_units
    def plot_LOS(self, fig=None, ax=None,
                 show_obs=True, residuals=False, *,
                 x_unit='pc', y_unit='km/s', legend=True,
                 label_position='top', verbose_label=True, blank_xaxis=False,
                 res_kwargs=None, **kwargs):
        r'''Plot the line-of-sight velocity dispersion profiles.

        Plots the `self.LOS` model profiles, and the relevant observational
        datasets (variable "σ" within the dataset pattern
        "*velocity_dispersion*"), corresponding to the line-of-sight velocity
        dispersion profiles.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        show_obs : bool, optional
            Whether to also plot the corresponding observational datasets on
            top of the relevant model profile. Defaults to True.

        residuals : bool, optional
            Whether to, if also plotting data, append a residuals ax to the
            figure. Defaults to False.

        x_unit, y_unit : u.Unit, optional
            Units to convert the x and y axes to. By default, x-units are in
            parsecs and y-units are in kilometres per second.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top" (default), will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label.

        verbose_label : bool, optional
            If True (default), quantity label will be "LOS Velocity Dispersion",
            otherwise "$\sigma_{\mathrm{LOS}}$".

        blank_xaxis : bool, optional
            If True, will remove the tick markers and label from the x-axis.
            May be useful if stacking multiple profiles vertically.

        res_kwargs : dict, optional
            Optional arguments passed to `_add_residuals`.

        **kwargs : dict
            All other arguments are passed to `_plot_profile`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*velocity_dispersion*', 'σ'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        ax, res_ax = self._plot_profile(ax, pattern, var, self.LOS,
                                        strict=strict, residuals=residuals,
                                        x_unit=x_unit, y_unit=y_unit,
                                        legend=legend, res_kwargs=res_kwargs,
                                        **kwargs)

        if verbose_label:
            label = 'LOS Velocity Dispersion'
        else:
            label = r'$\sigma_{\mathrm{LOS}}$'

        self._set_ylabel(ax, label, y_unit, label_position, residual_ax=res_ax)
        self._set_xlabel(ax, unit=x_unit, residual_ax=res_ax,
                         remove_all=blank_xaxis)

        return fig

    @_support_units
    def plot_pm_tot(self, fig=None, ax=None,
                    show_obs=True, residuals=False, *,
                    x_unit='pc', y_unit='mas/yr', legend=True,
                    label_position='top', verbose_label=True, blank_xaxis=False,
                    res_kwargs=None, **kwargs):
        r'''Plot the total proper motion dispersion profiles.

        Plots the `self.pm_tot` model profiles, and the relevant observational
        datasets (variable "PM_tot" within the dataset pattern
        "*proper_motion*"), corresponding to the total proper motion
        dispersion profiles.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        show_obs : bool, optional
            Whether to also plot the corresponding observational datasets on
            top of the relevant model profile. Defaults to True.

        residuals : bool, optional
            Whether to, if also plotting data, append a residuals ax to the
            figure. Defaults to False.

        x_unit, y_unit : u.Unit, optional
            Units to convert the x and y axes to. By default, x-units are in
            parsecs and y-units are in milliarcseconds per year.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top" (default), will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label.

        verbose_label : bool, optional
            If True (default), quantity label will be "Total PM Dispersion",
            otherwise "$\sigma_{\mathrm{PM},\mathrm{tot}}$".

        blank_xaxis : bool, optional
            If True, will remove the tick markers and label from the x-axis.
            May be useful if stacking multiple profiles vertically.

        res_kwargs : dict, optional
            Optional arguments passed to `_add_residuals`.

        **kwargs : dict
            All other arguments are passed to `_plot_profile`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_tot'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        ax, res_ax = self._plot_profile(ax, pattern, var, self.pm_tot,
                                        strict=strict, residuals=residuals,
                                        x_unit=x_unit, y_unit=y_unit,
                                        legend=legend, res_kwargs=res_kwargs,
                                        **kwargs)

        if verbose_label:
            label = "Total PM Dispersion"
        else:
            label = r'$\sigma_{\mathrm{PM},\mathrm{tot}}$'

        self._set_ylabel(ax, label, y_unit, label_position, residual_ax=res_ax)
        self._set_xlabel(ax, unit=x_unit, residual_ax=res_ax,
                         remove_all=blank_xaxis)

        return fig

    @_support_units
    def plot_pm_ratio(self, fig=None, ax=None,
                      show_obs=True, residuals=False, *,
                      x_unit='pc', blank_xaxis=False, legend=True,
                      label_position='top', verbose_label=True,
                      res_kwargs=None, **kwargs):
        r'''Plot the proper motion dispersion anisotropy profiles.

        Plots the `self.pm_ratio` model profiles, and the relevant observational
        datasets (variable "PM_ratio" within the dataset pattern
        "*proper_motion*"), corresponding to the proper motion dispersion
        anisotropy profiles, defined as the ratio of the tangential over
        radial proper motion dispersions.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        show_obs : bool, optional
            Whether to also plot the corresponding observational datasets on
            top of the relevant model profile. Defaults to True.

        residuals : bool, optional
            Whether to, if also plotting data, append a residuals ax to the
            figure. Defaults to False.

        x_unit : u.Unit, optional
            Units to convert the x axis to. By default, x-units are in
            parsecs.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top" (default), will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label.

        verbose_label : bool, optional
            If True (default), quantity label will be "PM Anisotropy Ratio",
            otherwise "$\sigma_{\mathrm{PM},\mathrm{T}}
            / \sigma_{\mathrm{PM},\mathrm{R}}$".

        blank_xaxis : bool, optional
            If True, will remove the tick markers and label from the x-axis.
            May be useful if stacking multiple profiles vertically.

        res_kwargs : dict, optional
            Optional arguments passed to `_add_residuals`.

        **kwargs : dict
            All other arguments are passed to `_plot_profile`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_ratio'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        ax, res_ax = self._plot_profile(ax, pattern, var, self.pm_ratio,
                                        strict=strict, residuals=residuals,
                                        x_unit=x_unit, legend=True,
                                        res_kwargs=res_kwargs, **kwargs)

        if verbose_label:
            label = "PM Anisotropy Ratio"
        else:
            label = (r'$\sigma_{\mathrm{PM},\mathrm{T}} / '
                     r'\sigma_{\mathrm{PM},\mathrm{R}}$')

        self._set_ylabel(ax, label, None, label_position, residual_ax=res_ax)
        self._set_xlabel(ax, unit=x_unit, residual_ax=res_ax,
                         remove_all=blank_xaxis)

        return fig

    @_support_units
    def plot_pm_T(self, fig=None, ax=None,
                  show_obs=True, residuals=False, *,
                  x_unit='pc', y_unit='mas/yr', legend=True,
                  label_position='top', verbose_label=True, blank_xaxis=False,
                  res_kwargs=None, **kwargs):
        r'''Plot the tangential proper motion dispersion profiles.

        Plots the `self.pm_T` model profiles, and the relevant observational
        datasets (variable "PM_T" within the dataset pattern
        "*proper_motion*"), corresponding to the tangential proper motion
        dispersion profiles.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        show_obs : bool, optional
            Whether to also plot the corresponding observational datasets on
            top of the relevant model profile. Defaults to True.

        residuals : bool, optional
            Whether to, if also plotting data, append a residuals ax to the
            figure. Defaults to False.

        x_unit, y_unit : u.Unit, optional
            Units to convert the x and y axes to. By default, x-units are in
            parsecs and y-units are in milliarcseconds per year.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top" (default), will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label.

        verbose_label : bool, optional
            If True (default), quantity label will be "Tangential PM
            Dispersion", otherwise "$\sigma_{\mathrm{PM},\mathrm{T}}$".

        blank_xaxis : bool, optional
            If True, will remove the tick markers and label from the x-axis.
            May be useful if stacking multiple profiles vertically.

        res_kwargs : dict, optional
            Optional arguments passed to `_add_residuals`.

        **kwargs : dict
            All other arguments are passed to `_plot_profile`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_T'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        ax, res_ax = self._plot_profile(ax, pattern, var, self.pm_T,
                                        strict=strict, residuals=residuals,
                                        x_unit=x_unit, y_unit=y_unit,
                                        legend=legend, res_kwargs=res_kwargs,
                                        **kwargs)

        if verbose_label:
            label = "Tangential PM Dispersion"
        else:
            label = r'$\sigma_{\mathrm{PM},\mathrm{T}}$'

        self._set_ylabel(ax, label, y_unit, label_position, residual_ax=res_ax)
        self._set_xlabel(ax, unit=x_unit, residual_ax=res_ax,
                         remove_all=blank_xaxis)

        return fig

    @_support_units
    def plot_pm_R(self, fig=None, ax=None,
                  show_obs=True, residuals=False, *,
                  x_unit='pc', y_unit='mas/yr', legend=True,
                  label_position='top', verbose_label=True, blank_xaxis=False,
                  res_kwargs=None, **kwargs):
        r'''Plot the radial proper motion dispersion profiles.

        Plots the `self.pm_T` model profiles, and the relevant observational
        datasets (variable "PM_T" within the dataset pattern
        "*proper_motion*"), corresponding to the radial proper motion
        dispersion profiles.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        show_obs : bool, optional
            Whether to also plot the corresponding observational datasets on
            top of the relevant model profile. Defaults to True.

        residuals : bool, optional
            Whether to, if also plotting data, append a residuals ax to the
            figure. Defaults to False.

        x_unit, y_unit : u.Unit, optional
            Units to convert the x and y axes to. By default, x-units are in
            parsecs and y-units are in milliarcseconds per year.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top" (default), will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label.

        verbose_label : bool, optional
            If True (default), quantity label will be "Radial PM Dispersion",
            otherwise "$\sigma_{\mathrm{PM},\mathrm{R}}$".

        blank_xaxis : bool, optional
            If True, will remove the tick markers and label from the x-axis.
            May be useful if stacking multiple profiles vertically.

        res_kwargs : dict, optional
            Optional arguments passed to `_add_residuals`.

        **kwargs : dict
            All other arguments are passed to `_plot_profile`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_R'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        ax, res_ax = self._plot_profile(ax, pattern, var, self.pm_R,
                                        strict=strict, residuals=residuals,
                                        x_unit=x_unit, y_unit=y_unit,
                                        legend=legend, res_kwargs=res_kwargs,
                                        **kwargs)

        if verbose_label:
            label = "Radial PM Dispersion"
        else:
            label = r'$\sigma_{\mathrm{PM},\mathrm{R}}$'

        self._set_ylabel(ax, label, y_unit, label_position, residual_ax=res_ax)
        self._set_xlabel(ax, unit=x_unit, residual_ax=res_ax,
                         remove_all=blank_xaxis)

        return fig

    @_support_units
    def plot_number_density(self, fig=None, ax=None,
                            show_background=False, subtract_background=False,
                            show_obs=True, residuals=False, *, legend=True,
                            x_unit='pc', y_unit='1/pc2', scale_to='model',
                            label_position='top', verbose_label=True,
                            blank_xaxis=False, res_kwargs=None,
                            data_kwargs=None, model_kwargs=None, **kwargs):
        r'''Plot the projected number density profiles.

        Plots the `self.numdens` model profiles, and the relevant observational
        datasets (variable "Σ" within the dataset pattern
        "*number_density*"), corresponding to the project number density
        profiles.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this relation. Should be a
            part of the given `fig`.

        show_background : bool, optional
            If True, also plots a horizontal line representing the value of
            the background density, as taken from the observational datasets
            (if possible).

        subtract_background : bool, optional
            If True, subtracts the background density value from the
            number density model profiles and data.

        show_obs : bool, optional
            Whether to also plot the corresponding observational datasets on
            top of the relevant model profile. Defaults to True.

        residuals : bool, optional
            Whether to, if also plotting data, append a residuals ax to the
            figure. Defaults to False.

        x_unit, y_unit : u.Unit, optional
            Units to convert the x and y axes to. By default, x-units are in
            parsecs and y-units are in stars per parsec squared.

        scale_to : {"model", "data"}, optional
            Number densities are typically scaled to match the normalization
            of the data in GCfit, as they are fit to the shape, not value of
            the number densities. This argument allows the absolute values of
            the plotted quantities to be scaled to either the
            model (data / K; default) or the data (model * K).

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top" (default), will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label.

        verbose_label : bool, optional
            If True (default), quantity label will be "Tangential PM
            Dispersion", otherwise "$\sigma_{\mathrm{PM},\mathrm{T}}$".

        blank_xaxis : bool, optional
            If True, will remove the tick markers and label from the x-axis.
            May be useful if stacking multiple profiles vertically.

        res_kwargs, data_kwargs, model_kwargs : dict, optional
            Optional arguments passed to `_add_residuals`, `_plot_data`, and
            `_plot_model`, respectively.

        **kwargs : dict
            All other arguments are passed to `_plot_profile`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        # TODO add minor ticks to y axis

        def quad_nuisance(err):
            return np.sqrt(err**2 + (self.s2 << u.arcmin**-4))

        # ------------------------------------------------------------------
        # Setup the figures
        # ------------------------------------------------------------------

        fig, ax = self._setup_artist(fig, ax)

        ax.set_xscale('log')
        ax.set_yscale('log')

        # ------------------------------------------------------------------
        # Determine the datasets to plot
        # ------------------------------------------------------------------

        if show_obs:
            pattern, var = '*number_density*', 'Σ'
            strict = show_obs == 'strict'

            if data_kwargs is None:
                data_kwargs = {}

            data_kwargs.setdefault('err_transform', quad_nuisance)

        else:
            pattern = var = None
            strict = False

        # ------------------------------------------------------------------
        # Compute the scaling relations if necessary
        # ------------------------------------------------------------------

        try:
            if scale_to == 'model':
                if data_kwargs is None:
                    data_kwargs = {}
                data_kwargs.setdefault('scale', 1 / self.K_scale[self.star_bin])

            elif scale_to == 'data':
                if model_kwargs is None:
                    model_kwargs = {}
                model_kwargs.setdefault('scale', self.K_scale[self.star_bin])

        except TypeError:
            pass

        # ------------------------------------------------------------------
        # Compute and optionally plot the background
        # ------------------------------------------------------------------

        if show_background and subtract_background:
            mssg = ("Cannot both 'show_background' and 'subtract_background' "
                    "at same time")
            raise ValueError(mssg)

        elif show_background or subtract_background:

            # --------------------------------------------------------------
            # Compute the background
            # --------------------------------------------------------------

            if self.obs is None:
                mssg = "No observations for this model; cannot plot background"
                raise RuntimeError(mssg)

            try:
                nd = list(self.obs.filter_datasets('*number*').values())[-1]
                background = nd.mdata['background'] << nd['Σ'].unit

                if scale_to == 'model' and self.K_scale is not None:
                    background /= self.K_scale[self.star_bin]

            except IndexError as err:
                mssg = ('No number density profile data found, '
                        'cannot compute background')
                raise RuntimeError(mssg) from err
            except KeyError as err:
                mssg = 'No background level found in number density metadata'
                raise RuntimeError(mssg) from err

            # --------------------------------------------------------------
            # Plot the background
            # --------------------------------------------------------------

            if show_background:
                ax.axhline(y=background, ls='--', c='black', alpha=0.66)

            elif subtract_background:
                data_kwargs['background'] = background

        # ------------------------------------------------------------------
        # Plot number denstiy profiles
        # ------------------------------------------------------------------

        ax, res_ax = self._plot_profile(ax, pattern, var, self.numdens,
                                        strict=strict, residuals=residuals,
                                        x_unit=x_unit, y_unit=y_unit,
                                        legend=legend,
                                        model_kwargs=model_kwargs,
                                        data_kwargs=data_kwargs,
                                        res_kwargs=res_kwargs, **kwargs)

        # ------------------------------------------------------------------
        # Add labels
        # ------------------------------------------------------------------

        if verbose_label:
            label = 'Number Density'
        else:
            label = r'$\Sigma$'

        self._set_ylabel(ax, label, y_unit, label_position, residual_ax=res_ax)
        self._set_xlabel(ax, unit=x_unit, residual_ax=res_ax,
                         remove_all=blank_xaxis)

        return fig

    @_support_units
    def plot_all(self, fig=None, sharex=True, **kwargs):
        '''Plot all primary model radial profiles in one figure.

        Plots the six primary radial profile quantities used for fitting
        (excluding notably the mass functions and pulsar accelerations).
        That is, clockwise from the top left, number density, total PM,
        tangential PM, radial PM, PM ratio and LOS dispersion profiles.

        Simply sets up a figure with six axes and calls the various relevant
        profile plotting functions to populate each ax.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_multi_artist` for more details.

        sharex : bool, optional
            If True, created subplots will all share the same x-axis.
            Will also remove the x axis ticks and labels on all but the bottom
            row.

        **kwargs : dict
            All other arguments are passed to each plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        # TODO working with residuals here is hard because constrianed_layout
        #   doesn't seem super aware of them

        # ------------------------------------------------------------------
        # Setup figure
        # ------------------------------------------------------------------

        fig, axes = self._setup_multi_artist(fig, (3, 2), sharex=sharex)

        axes = axes.reshape((3, 2))

        res_kwargs = dict(size="25%", show_chi2=False, percentage=True)
        kwargs.setdefault('res_kwargs', res_kwargs)

        # ------------------------------------------------------------------
        # Left Plots
        # ------------------------------------------------------------------

        # Number Density

        show_numdens_background, bg_lim = False, None

        if self.obs is not None and kwargs.get('show_obs', True):

            if nd := list(self.obs.filter_datasets('*number*').values()):
                nd = nd[0]

                if 'background' in nd.mdata:

                    show_numdens_background = True
                    bg_lim = 0.9 * nd.mdata['background'] << nd['Σ'].unit

        self.plot_number_density(fig=fig, ax=axes[0, 0], label_position='left',
                                 blank_xaxis=True,
                                 show_background=show_numdens_background,
                                 **kwargs)

        if self.numdens is not None and bg_lim is not None:
            bg_lim = min([bg_lim, np.abs(self.numdens[..., :-2].min())])

        # Sometime deBoer lists BGlev=0.000, but only due to digits cut off
        if bg_lim is not None and bg_lim <= 0.0:
            bg_lim = 1e-3

        axes[0, 0].set_ylim(bottom=bg_lim)

        # Line-of-Sight Velocity Dispersion

        self.plot_LOS(fig=fig, ax=axes[1, 0], label_position='left',
                      blank_xaxis=True, **kwargs)

        axes[1, 0].set_ylim(bottom=0.0)

        # Proper Motion Anisotropy

        self.plot_pm_ratio(fig=fig, ax=axes[2, 0], label_position='left',
                           **kwargs)

        axes[2, 0].set_ylim(bottom=0.4, top=max(axes[2, 0].get_ylim()[1], 1.2))

        # ------------------------------------------------------------------
        # Right Plots
        # ------------------------------------------------------------------

        # Total Proper Motion Dispersion

        self.plot_pm_tot(fig=fig, ax=axes[0, 1], label_position='left',
                         blank_xaxis=True, **kwargs)

        axes[0, 1].set_ylim(bottom=0.0)

        # Tangential Proper Motion Dispersion

        self.plot_pm_T(fig=fig, ax=axes[1, 1], label_position='left',
                       blank_xaxis=True, **kwargs)

        axes[1, 1].set_ylim(bottom=0.0)

        # Radial Proper Motion Dispersion

        self.plot_pm_R(fig=fig, ax=axes[2, 1], label_position='left',
                       **kwargs)

        axes[2, 1].set_ylim(bottom=0.0)

        # ------------------------------------------------------------------
        # Style plots
        # ------------------------------------------------------------------

        # brute force clear out any "residuals" labels
        for ax in fig.axes:
            if 'Residual' in ax.get_ylabel():
                ax.set_ylabel('')

        fig.align_ylabels()

        return fig

    # ----------------------------------------------------------------------
    # Mass Function Plotting
    # ----------------------------------------------------------------------

    @_support_units
    def plot_mass_func(self, fig=None, show_obs=True, show_fields=True, *,
                       PI_legend=False, propid_legend=False,
                       label_unit='arcmin', model_color=None, model_label=None,
                       logscaled=False, field_kw=None, **kwargs):
        """Plot present day mass functions in various radial bins.

        Plots each of the stored `self.mass_func` radial (present-day) mass
        function bins (i.e. binned star counts by mean star mass), in a
        large figure of stacked axes, as well as the corresponding
        observational datasets.

        Each grouping of mass functions bins (i.e. under a certain data
        source proposal) will be plotted with it's own colour and sorted to be
        next to one another.

        Optionally, a panel will be added to the left of the figure showing
        the related field outlines on the sky (using `plot_MF_fields`).

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place all axes on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_multi_artist` for more details.

        show_obs : bool, optional
            Whether to also plot the corresponding observational datasets on
            top of the relevant model profile. Defaults to True.

        show_fields : bool, optional
            Whether to include an extra panel (on the left of the figure)
            showing the observed mass function fields (using `plot_MF_fields`).

        PI_legend : bool, optional
            If True, also notes the PI name of each proposal grouping under
            each first radial bin annotation.
            Only applies if `show_fields` is False, as otherwise they will be
            noted in that panel better.

        propid_legend : bool, False
            If True, also notes the ID of each proposal grouping under each
            first radial bin annotation.
            Only applies if `show_fields` is False, as otherwise they will be
            noted in that panel better.

        label_unit : u.Unit, optional
            The unit used to denote each radial bin in it's corresponding label.
            Does not change anything about what is plotted, only the label.
            Defaults to arcminutes (').

        model_color : str, optional
            Optionally colour all model profiles with a specific colour. This
            will give the same colour to *all* radial bins, regardless of
            proposal grouping. By default, models will match the colour of
            their corresponding data, as dictated within `self.mass_func`, or
            as set by the default matplotlib colour cycler, if not set.

        model_label : str, optional
            Optionally add a figure legend with this label corresponding to the
            model profiles. Only valid if a single specific model colour was
            given through `model_color`. The legend placement won't be great,
            and this is mostly for inspection when overplotting multiple models
            on a single figure.

        logscaled : bool, optional
            If True, applies a log scaling to the the x (i.e. mass) axis.

        field_kw : dict, optional
            Optional arguments passed to `plot_MF_fields` if `show_fields` is
            True.

        **kwargs : dict
            All other arguments are passed to each `plt.errorbars` call.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        """

        # TODO support using other x units (arcsec, pc) here and in MF plot

        if self.obs is None:
            show_obs = False

        if self.mass_func is None:
            # Should only occur for ObservationsVisualizer with no MF data
            #   at which point I'm not sure what you're even trying to plot
            #   (this avoids a very ugly error)
            raise ValueError("No mass function data exists to plot")

        # ------------------------------------------------------------------
        # Setup axes, splitting into two columns if necessary and adding the
        # extra ax for the field plot if desired
        # ------------------------------------------------------------------

        ax_ind = 0

        N_rbins = sum([len(d) for d in self.mass_func.values()])

        if N_rbins > 5:
            shape = ((int(np.ceil(N_rbins / 2)), int(np.floor(N_rbins / 2))), 2)
        else:
            shape = ((N_rbins, ), 1)  # this format ensures subfigs are created

        # If adding the fields, include an extra column on the left for it
        if show_fields:
            shape = ((1, *shape[0]), shape[1] + 1)

        # if it looks like this comes from a past call, try to preserve MF ax
        elif fig is not None and len(fig.axes) == (N_rbins + 1):
            shape = ((1, *shape[0]), shape[1] + 1)
            ax_ind = 1

        fig, axes = self._setup_multi_artist(fig, shape, sharex=True)

        axes = axes.T.flatten()

        # ------------------------------------------------------------------
        # If desired, use the `plot_MF_fields` method to show the fields
        # ------------------------------------------------------------------

        if show_fields:

            ax = axes[ax_ind]

            if field_kw is None:
                field_kw = {}

            field_kw.setdefault('radii', [])
            field_kw.setdefault('unit', label_unit)

            self.plot_MF_fields(fig, ax, **field_kw)

            ax.set_aspect('equal')

            ax_ind += 1

        # ------------------------------------------------------------------
        # Iterate over each PI, gathering data to plot
        # ------------------------------------------------------------------

        kwargs.setdefault('mfc', None)
        kwargs.setdefault('mec', 'k')
        kwargs.setdefault('mew', 0.3)
        kwargs.setdefault('linestyle', 'None')
        kwargs.setdefault('marker', 'o')

        for PI in sorted(self.mass_func,
                         key=lambda k: self.mass_func[k][0]['r1']):

            bins = self.mass_func[PI]

            label = ''
            if PI_legend:
                label += pathlib.Path(PI).name

            # Get data for this PI

            if show_obs:

                mf = self.obs[PI]

                mbin_mean = (mf['m1'] + mf['m2']) / 2.
                mbin_width = mf['m2'] - mf['m1']

                N = mf['N'] / mbin_width
                ΔN = mf['ΔN'] / mbin_width

                if propid_legend:
                    label += f" ({mf.mdata['proposal']})"

            # Make sure we're not already labeling in a field plot
            if show_fields and field_kw.get('add_legend', True):
                label = ''

            # --------------------------------------------------------------
            # Iterate over radial bin dicts for this PI
            # --------------------------------------------------------------
            # TODO make plotting calls a bit more uniform with profiles

            for rind, rbin in enumerate(bins):

                ax = axes[ax_ind]

                data_clr = rbin.get('colour', None)

                # ----------------------------------------------------------
                # Plot observations
                # ----------------------------------------------------------

                if show_obs:

                    r_mask = ((mf['r1'] == rbin['r1'])
                              & (mf['r2'] == rbin['r2']))

                    N_data = N[r_mask].value
                    err_data = ΔN[r_mask].value

                    err = self.F * err_data

                    pnts = ax.errorbar(mbin_mean[r_mask], N_data, yerr=err,
                                       color=data_clr, **kwargs)

                    data_clr = pnts[0].get_color()

                # ----------------------------------------------------------
                # Plot model. Doesn't utilize the `_plot_profile` method, as
                # this is *not* a profile, but does use similar, but simpler,
                # logic
                # ----------------------------------------------------------

                # If really desired, don't match model colour to bins
                model_clr = model_color if model_color is not None else data_clr

                # The mass domain is provided explicitly, to support visualizers
                # which don't store the entire mass range (e.g. CImodels)
                mj = rbin['mj']

                dNdm = rbin['dNdm']

                midpoint = dNdm.shape[0] // 2

                median = dNdm[midpoint]

                # TODO zorder needs work here, noticeable when colors dont match
                med_plot, = ax.plot(mj, median, color=model_clr)

                alpha = 0.8 / (midpoint + 1)
                for sigma in range(1, midpoint + 1):

                    ax.fill_between(
                        mj,
                        dNdm[midpoint + sigma],
                        dNdm[midpoint - sigma],
                        alpha=1 - alpha, color=model_clr
                    )

                    alpha += alpha

                if logscaled:
                    ax.set_xscale('log')

                ax.set_xlabel(None)

                # TODO would be nice to use scientific notation on yaxis, but
                #   it's hard to get it working nicely

                # ----------------------------------------------------------
                # "Label" each bin with it's radial bounds.
                # Uses fake text to allow for using loc='best' from `legend`.
                # Really this should be a part of plt (see matplotlib#17946)
                # ----------------------------------------------------------

                r1 = rbin['r1'].to(label_unit)
                r2 = rbin['r2'].to(label_unit)
                lu = ''

                if label_unit in ('arcmin', 'arcminute'):
                    r1, r2, lu = r1.value, r2.value, "'"

                elif label_unit in ('arcsec', 'arcsecond'):
                    r1, r2, lu = r1.value, r2.value, '"'

                fake = plt.Line2D([], [],
                                  label=f"r = {r1:.2f}{lu}-{r2:.2f}{lu}")
                handles = [fake]

                leg_kw = {'handlelength': 0, 'handletextpad': 0}

                # If this is the first bin, also add a PI tag
                if label and not rind:
                    lbl_fake = plt.Line2D([], [], label=label)
                    handles.append(lbl_fake)
                    leg_kw['labelcolor'] = ['k', data_clr]

                ax.legend(handles=handles, **leg_kw)

                ax_ind += 1

        # ------------------------------------------------------------------
        # Put labels on subfigs
        # ------------------------------------------------------------------

        for sf in fig.subfigs[show_fields:]:

            # sf.supxlabel(r'Mass [$M_\odot$]')
            sf.axes[-1].set_xlabel(r'Mass [$M_\odot$]')

        fig.subfigs[show_fields].supylabel(r'$\mathrm{d}N / \mathrm{d}m$')

        # TODO this is a pretty messy way to do this, but legends are messy
        #   location placement is a bit broken, but plots are so busy already
        if model_label:

            # TODO can't get "outside" loc keyword to work in subfigs
            loc = dict(loc='upper center', ncols=3)

            if len(shape[0]) == 3:  # has field plot
                sf = fig.subfigs[0]
                loc['bbox_to_anchor'] = (0.5, 1.0)

            else:
                sf = fig.subfigs[-1]
                loc['bbox_to_anchor'] = (0.5, 1.05)

            if model_color is None:
                mssg = ("cannot label a model which was not explicitly given a "
                        "single separate colour (`model_color`)")
                raise ValueError(mssg)

            lbl_fake = plt.Line2D([], [], label=model_label, color=model_color)

            if sf.legends:
                old_leg = sf.legends[0]
                handles = old_leg.legendHandles + [lbl_fake]

                sf.legend(handles=handles, **loc)
                old_leg.remove()

            else:
                sf.legend(handles=[lbl_fake], **loc)

        return fig

    @_support_units
    def plot_MF_fields(self, fig=None, ax=None, *, unit='arcmin', radii=("rh",),
                       grid=True, label_grid=False, add_legend=True):
        '''Plot the on-sky HST fields corresponding to each mass function PI.

        Plots the representation of the observed on-sky photometry footprints
        for each stored mass function groupings, with the origin on the cluster
        centre.
        Each grouping of mass functions bins (i.e. under a certain data
        source proposal) will be plotted with it's own colour.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot the fields. Should be a
            part of the given `fig`.

        unit : u.Unit, optional
            Units to convert the x and y axes and grid lines to.
            By default, the units are arcminutes.

        radii : list of {'rh', 'ra', 'rt', 'r0', 'rhp', 'rv'}, optional
            A collection of radius values to be overplotted (as circles)
            on the plot. If the given radii have corresponding uncertainties,
            they will also be shaded in. By default plots the half-mass radius.

        grid : bool, optional
            If True (default), adds a circular grid at 2 arcminute intervals,
            out to the tidal radius.

        label_grid : bool, optional
            Optionally adds labels on to the figure noting the size of each
            grid interval, if `grid` is True.

        add_legend : bool, optional
            If True (default), adds a legend with an entry for each grouping
            with the proposal PI name and ID.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.

        See Also
        --------
        util.mass.Field:
            Class representing a given mass function field, containing all
            associated plotting methods.
        '''

        import shapely.geometry as geom

        fig, ax = self._setup_artist(fig, ax)

        # Centre dot
        ax.plot(0, 0, 'kx')

        # ------------------------------------------------------------------
        # Parse units
        # ------------------------------------------------------------------
        # TODO do we support non-angular units correctly?

        unit = u.Unit(unit)

        if unit in (u.arcmin, u.arcsec):
            unit_lbl = fr"\mathrm{{{unit}}}"
            short_unit_lbl = f"{unit:latex}".strip('$')
        else:
            unit_lbl = short_unit_lbl = f"{unit:latex}".strip('$')

        # ------------------------------------------------------------------
        # Iterate over each PI and it's radial bins
        # ------------------------------------------------------------------

        for PI, bins in self.mass_func.items():

            lbl = f"{pathlib.Path(PI).name}"

            if self.obs is not None:
                lbl = f"{lbl} ({self.obs[PI].mdata['proposal']})"

            for rbin in bins:

                # ----------------------------------------------------------
                # Plot the field using this `Field` slice's own plotting method
                # ----------------------------------------------------------

                clr = rbin.get("colour", None)

                rbin['field'].plot(ax, fc=clr, alpha=0.7, ec='k',
                                   label=lbl, unit=unit)

                # make this label private so it's only added once to legend
                lbl = f'_{lbl}'

        # ------------------------------------------------------------------
        # If desired, add a "pseudo" grid in the polar projection, at 2
        # arcmin intervals, up to the rt
        # ------------------------------------------------------------------

        # Ensure the gridlines don't affect the axes scaling
        ax.autoscale(False)

        if grid:
            # TODO this should probably use distance to furthest field

            try:
                rt = np.nanmedian(self.rt).to_value(unit) + 2
            except AttributeError:
                rt = (20 << u.arcmin).to_value(unit)

            stepsize = (2 << u.arcmin).to_value(unit)
            ticks = np.arange(2, rt, stepsize)

            # make sure this grid matches normal grids
            grid_kw = {
                'color': plt.rcParams.get('grid.color'),
                'linestyle': plt.rcParams.get('grid.linestyle'),
                'linewidth': plt.rcParams.get('grid.linewidth'),
                'alpha': plt.rcParams.get('grid.alpha'),
                'zorder': 0.5
            }

            for gr in ticks:
                circle = np.array(geom.Point(0, 0).buffer(gr).exterior.coords).T
                gr_line, = ax.plot(*circle, **grid_kw)

                if label_grid:
                    ax.annotate(f'${gr:.0f}{short_unit_lbl}$',
                                xy=(circle[0].max(), 0),
                                color=grid_kw['color'])

        # ------------------------------------------------------------------
        # Try to plot the various radii quantities from this model, if desired
        # ------------------------------------------------------------------

        valid_rs = {'rh', 'ra', 'rt', 'r0', 'rhp', 'rv'}

        q = [84.13, 50., 15.87]

        for r_type in radii:

            # This is to explicitly avoid very ugly exceptions from geom
            if r_type not in valid_rs:
                mssg = f'radii must be one of {valid_rs}, not `{r_type}`'
                raise TypeError(mssg)

            r_lbl = f'$r_{{{r_type[1:]}}}$'

            radius = getattr(self, r_type).to(unit)

            σr_u, r, σr_l = np.nanpercentile(radius, q=q)

            # Plot median as line
            mid = np.array(geom.Point(0, 0).buffer(r.value).exterior.coords).T
            r_line = ax.plot(*mid, ls='--')[0]
            ax.text(0, mid[1].max() * 1.05, r_lbl,
                    c=r_line.get_color(), ha='center')

            # Plot a polygon slice for the uncertainties
            try:
                outer = mass.Field(geom.Point(0, 0).buffer(σr_u.value))
                uncert = outer.slice_radially(σr_l, σr_u)

                uncert.plot(ax, alpha=0.1, fc=r_line.get_color())

            # Couldn't plot, probably because this is a non-CI model
            except ValueError:
                pass

        # ------------------------------------------------------------------
        # Add plot labels and legends
        # ------------------------------------------------------------------

        ax.set_xlabel(rf'$\Delta\,\mathrm{{RA}}\ \left[{unit_lbl}\right]$')
        ax.set_ylabel(rf'$\Delta\,\mathrm{{DEC}}\ \left[{unit_lbl}\right]$')

        if add_legend:
            # TODO figure out a better way of handling this always using best?
            ax.legend(loc='upper left' if grid else 'best')

        return fig

    # -----------------------------------------------------------------------
    # Model plotting
    # -----------------------------------------------------------------------

    @_support_units
    def plot_density(self, fig=None, ax=None, kind='all', *,
                     x_unit='pc', label_position='left', colors=None):
        '''Plot model density profiles.

        Plots the radial mass density (`self.rho_*`) profiles of the total,
        main sequence, and dark remnant objects in this model.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot the profiles. Should be a
            part of the given `fig`.

        kind : list of {'tot', 'rem', 'MS', 'WD', 'NS', 'BH'}, optional
            The density profiles to plot. By default plots all of them.

        x_unit : u.Unit, optional
            Units to convert the x axis to. By default, x-units are in parsecs.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top", will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label. Defaults to the left.

        colors : dict, optional
            A dictionary of colours, where the keys correspond to the given
            `kind`. Each object type has it's own default colour.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        if colors is None:
            colors = {}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Surface Mass Density')

        # Total density
        if 'tot' in kind:
            self._plot_profile(ax, None, None, self.rho_tot,
                               x_unit=x_unit, model_label="Total",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("tot", "tab:cyan"))

        # Total Remnant density
        if 'rem' in kind:
            self._plot_profile(ax, None, None, self.rho_rem,
                               x_unit=x_unit, model_label="Remnants",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("rem", "tab:purple"))

        # Main sequence density
        if 'MS' in kind:
            self._plot_profile(ax, None, None, self.rho_MS,
                               x_unit=x_unit, model_label="Main-sequence stars",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("MS", "tab:orange"))

        if 'WD' in kind:
            self._plot_profile(ax, None, None, self.rho_WD,
                               x_unit=x_unit, model_label="White Dwarfs",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("WD", "tab:green"))

        if 'NS' in kind:
            self._plot_profile(ax, None, None, self.rho_NS,
                               x_unit=x_unit, model_label="Neutron Stars",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("NS", "tab:red"))

        # Black hole density
        if 'BH' in kind:
            self._plot_profile(ax, None, None, self.rho_BH,
                               x_unit=x_unit, model_label="Black Holes",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("BH", "tab:gray"))

        ax.set_yscale("log")
        ax.set_xscale("log")

        self._set_ylabel(ax, 'Mass Density', self.rho_tot.unit, label_position)
        self._set_xlabel(ax, unit=x_unit)

        ax.legend(loc='upper center', ncol=len(kind), fancybox=True)

        return fig

    @_support_units
    def plot_surface_density(self, fig=None, ax=None, kind='all', *,
                             x_unit='pc', label_position='left', colors=None):
        '''Plot model surface density profiles.

        Plots the radial surface mass density (`self.Sigma_*`) profiles of the
        total, main sequence, and dark remnant objects in this model.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot the profiles. Should be a
            part of the given `fig`.

        kind : list of {'tot', 'rem', 'MS', 'WD', 'NS', 'BH'}, optional
            The density profiles to plot. By default plots all of them.

        x_unit : u.Unit, optional
            Units to convert the x axis to. By default, x-units are in parsecs.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top", will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label. Defaults to the left.

        colors : dict, optional
            A dictionary of colours, where the keys correspond to the given
            `kind`. Each object type has it's own default colour.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        if colors is None:
            colors = {}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Surface Mass Density')

        # Total density
        if 'tot' in kind:
            self._plot_profile(ax, None, None, self.Sigma_tot,
                               x_unit=x_unit, model_label="Total",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("tot", "tab:cyan"))

        # Total Remnant density
        if 'rem' in kind:
            self._plot_profile(ax, None, None, self.Sigma_rem,
                               x_unit=x_unit, model_label="Remnants",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("rem", "tab:purple"))

        # Main sequence density
        if 'MS' in kind:
            self._plot_profile(ax, None, None, self.Sigma_MS,
                               x_unit=x_unit, model_label="Main-sequence stars",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("MS", "tab:orange"))

        if 'WD' in kind:
            self._plot_profile(ax, None, None, self.Sigma_WD,
                               x_unit=x_unit, model_label="White Dwarfs",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("WD", "tab:green"))

        if 'NS' in kind:
            self._plot_profile(ax, None, None, self.Sigma_NS,
                               x_unit=x_unit, model_label="Neutron Stars",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("NS", "tab:red"))

        # Black hole density
        if 'BH' in kind:
            self._plot_profile(ax, None, None, self.Sigma_BH,
                               x_unit=x_unit, model_label="Black Holes",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("BH", "tab:gray"))

        ax.set_yscale("log")
        ax.set_xscale("log")

        self._set_ylabel(ax, 'Surface Mass Density', self.Sigma_tot.unit,
                         label_position)
        self._set_xlabel(ax, unit=x_unit)

        ax.legend(loc='upper center', ncol=len(kind), fancybox=True)

        return fig

    @_support_units
    def plot_cumulative_mass(self, fig=None, ax=None, kind='all', *,
                             x_unit='pc', label_position='left', colors=None):
        '''Plot model cumulative mass profiles.

        Plots the radial cumulative mass profiles of the total,
        main sequence, and dark remnant objects in this model.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot the profiles. Should be a
            part of the given `fig`.

        kind : list of {'tot', 'rem', 'MS', 'WD', 'NS', 'BH'}, optional
            The mass profiles to plot. By default plots all of them.

        x_unit : u.Unit, optional
            Units to convert the x axis to. By default, x-units are in parsecs.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top", will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label. Defaults to the left.

        colors : dict, optional
            A dictionary of colours, where the keys correspond to the given
            `kind`. Each object type has it's own default colour.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        if colors is None:
            colors = {}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Cumulative Mass')

        # Total density
        if 'tot' in kind:
            self._plot_profile(ax, None, None, self.cum_M_tot,
                               x_unit=x_unit, model_label="Total",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("tot", "tab:cyan"))

        # Main sequence density
        if 'MS' in kind:
            self._plot_profile(ax, None, None, self.cum_M_MS,
                               x_unit=x_unit, model_label="Main-sequence stars",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("MS", "tab:orange"))

        if 'WD' in kind:
            self._plot_profile(ax, None, None, self.cum_M_WD,
                               x_unit=x_unit, model_label="White Dwarfs",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("WD", "tab:green"))

        if 'NS' in kind:
            self._plot_profile(ax, None, None, self.cum_M_NS,
                               x_unit=x_unit, model_label="Neutron Stars",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("NS", "tab:red"))

        # Black hole density
        if 'BH' in kind:
            self._plot_profile(ax, None, None, self.cum_M_BH,
                               x_unit=x_unit, model_label="Black Holes",
                               mass_bins=[0], label_masses=False,
                               color=colors.get("BH", "tab:gray"))

        ax.set_yscale("log")
        ax.set_xscale("log")

        self._set_ylabel(ax, rf'$M_{{enc}}$', self.cum_M_tot.unit,
                         label_position)
        self._set_xlabel(ax, unit=x_unit)

        ax.legend(loc='lower center', ncol=len(kind), fancybox=True)

        return fig

    @_support_units
    def plot_remnant_fraction(self, fig=None, ax=None, *, show_total=True,
                              x_unit='pc', label_position='left'):
        '''Plot model stellar and remnant mass fraction profiles.

        Plots the radial mass fraction profile of the main sequence stars
        (`mass_MS / mass_tot`) and the dark remnants (`mass_rem / mass_tot`).

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot the profiles. Should be a
            part of the given `fig`.

        show_total : bool, optional
            If True (default), places an anchored box in the upper centre of
            the ax showing the total remnant mass fraction of the cluster.

        x_unit : u.Unit, optional
            Units to convert the x axis to. By default, x-units are in parsecs.

        label_position : {'top', 'left', 'right'}, optional
            Where to place the quantity (y) label. If "top", will be
            set as the ax title, otherwise will be set to one side. If on a
            side, will also attempt to correctly append the units to the end
            of the label. Defaults to the left.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Remnant Fraction")

        ax.set_xscale("log")

        self._plot_profile(ax, None, None, self.frac_M_MS,
                           x_unit=x_unit, model_label="Main-sequence stars",
                           mass_bins=[0], label_masses=False)
        self._plot_profile(ax, None, None, self.frac_M_rem,
                           x_unit=x_unit, model_label="Remnants",
                           mass_bins=[0], label_masses=False)

        label = r"Mass fraction $M_{MS}/M_{tot}$, $M_{remn}/M_{tot}$"
        self._set_ylabel(ax, label, None, label_position)
        self._set_xlabel(ax, unit=x_unit)

        ax.set_ylim(0.0, 1.0)

        if show_total:
            from matplotlib.offsetbox import AnchoredText

            tot = np.nanpercentile(self.f_rem, q=[50., 15.87, 84.13]).value
            tot[1:] = np.abs(tot[1:] - tot[0])

            lbl = f'{tot[0]:.2f}^{{+{tot[1]:.2f}}}_{{-{tot[2]:.2f}}}\\,\\%'

            txt = AnchoredText(fr'$f_{{\mathrm{{remn}}}}={lbl}$',
                               frameon=True, loc='upper center')

            txt.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(txt)

        ax.legend()

        return fig

    # -----------------------------------------------------------------------
    # Goodness of fit statistics
    # -----------------------------------------------------------------------

    @_support_units
    def _compute_profile_chi2(self, ds_pattern, y_key, model_data, *,
                              x_key='r', err_transform=None, reduced=True):
        '''Compute chi2 for this dataset (pattern)'''

        chi2 = 0.

        # ensure that the data is (mass bin, intervals, r domain)
        if len(model_data.shape) != 3:
            raise ValueError("invalid model data shape")

        # ------------------------------------------------------------------
        # Determine the relevant datasets to the given pattern
        # ------------------------------------------------------------------

        ds_pattern = ds_pattern or ''

        datasets = self.obs.filter_datasets(ds_pattern)

        # ------------------------------------------------------------------
        # Iterate over the datasets, computing chi2 for each
        # ------------------------------------------------------------------

        for dset in datasets.values():

            # --------------------------------------------------------------
            # get mass bin of this dataset
            # --------------------------------------------------------------

            if 'm' in dset.mdata:
                m = dset.mdata['m'] * u.Msun
                mass_bin = np.where(self.mj == m)[0][0]
            else:
                mass_bin = self.star_bin

            # --------------------------------------------------------------
            # get data values
            # --------------------------------------------------------------

            try:
                xdata = dset[x_key]  # x_key='r'
                ydata = dset[y_key]

            except KeyError:
                continue

            yerr = self._get_err(dset, y_key)

            if err_transform is not None:
                yerr = err_transform(yerr)

            yerr = yerr.to(ydata.unit)

            # --------------------------------------------------------------
            # get model values
            # --------------------------------------------------------------

            xmodel = self.r.to(xdata.unit)

            ymedian = self._get_median(model_data[mass_bin, :, :])

            # TEMPORRAY FIX FOR RATIO
            # ymedian[np.isnan(ymedian)] = 1.0 << ymedian.unit

            ymodel = util.QuantitySpline(xmodel, ymedian)(xdata).to(ydata.unit)

            # --------------------------------------------------------------
            # compute chi2
            # --------------------------------------------------------------

            denom = (ydata.size - 13) if reduced else 1.

            chi2 += np.nansum(((ymodel - ydata) / yerr)**2) / denom

        return chi2

    @_support_units
    def _compute_massfunc_chi2(self, *, reduced=True):
        '''Compute chi2 for all mass functions'''

        chi2 = 0.

        # ------------------------------------------------------------------
        # Iterate over each PI, gathering data
        # ------------------------------------------------------------------

        for PI in sorted(self.mass_func,
                         key=lambda k: self.mass_func[k][0]['r1']):

            bins = self.mass_func[PI]

            # Get data for this PI

            mf = self.obs[PI]

            mbin_mean = (mf['m1'] + mf['m2']) / 2.
            mbin_width = mf['m2'] - mf['m1']

            N = mf['N'] / mbin_width
            ΔN = mf['ΔN'] / mbin_width

            # --------------------------------------------------------------
            # Iterate over radial bin dicts for this PI
            # --------------------------------------------------------------

            for rind, rbin in enumerate(bins):

                # ----------------------------------------------------------
                # Get data
                # ----------------------------------------------------------

                r_mask = ((mf['r1'] == rbin['r1']) & (mf['r2'] == rbin['r2']))

                xdata = mbin_mean[r_mask]

                ydata = N[r_mask].value

                yerr = self.F * ΔN[r_mask].value

                # ----------------------------------------------------------
                # Get model
                # ----------------------------------------------------------

                xmodel = rbin['mj']

                ymedian = self._get_median(rbin['dNdm'])

                ymodel = util.QuantitySpline(xmodel, ymedian)(xdata)

                # TODO really should get this Nparam dynamically, if some fixed
                denom = (ydata.size - 13) if reduced else 1.

                chi2 += np.sum(((ymodel - ydata) / yerr)**2) / denom

        return chi2

    @property
    def chi2(self):
        '''Compute χ^2 between the median model and all observational data.'''

        # TODO seems to produce alot of infs?

        def numdens_nuisance(err):
            return np.sqrt(err**2 + (self.s2 << u.arcmin**-4))

        all_components = [
            {'ds_pattern': '*velocity_dispersion*', 'y_key': 'σ',
             'model_data': self.LOS},
            {'ds_pattern': '*proper_motion*', 'y_key': 'PM_tot',
             'model_data': self.pm_tot},
            {'ds_pattern': '*proper_motion*', 'y_key': 'PM_ratio',
             'model_data': self.pm_ratio},
            {'ds_pattern': '*proper_motion*', 'y_key': 'PM_T',
             'model_data': self.pm_T},
            {'ds_pattern': '*proper_motion*', 'y_key': 'PM_R',
             'model_data': self.pm_R},
            {'ds_pattern': '*number_density*', 'y_key': 'Σ',
             'model_data': self.numdens, 'err_transform': numdens_nuisance},
        ]

        chi2 = 0.

        for comp in all_components:
            chi2 += self._compute_profile_chi2(**comp)

        chi2 += self._compute_massfunc_chi2()

        return chi2


class ModelVisualizer(_ClusterVisualizer):
    '''Analysis and visualization of a single model.

    Provides a number of plotting methods useful for the analysis of a single
    (multimass) model instance, and it's fit to all relevant observational
    datasets.

    A number of relevant (plottable) quantities will be initially computed and
    stored to be used by the various plotting functions defined by the base
    class `_ClusterVisualizer`.

    Parameters
    ----------
    model : gcfit.Model
        Model instance to visualize.

    observations : gcfit.Observations or None, optional
        The `Observations` instance corresponding to this cluster.
        If None, the observations will first try to be read from the model
        instance itself (`model.observations`). If no observations can be found,
        this object will be unable to plot any observational data alongside
        it's modelled profiles.

    See Also
    --------
    _ClusterVisualizer : Base class providing all common plotting functions.
    '''

    @classmethod
    def from_chain(cls, chain, observations, method='median'):
        '''Initialize a visualizer based on a full chain of parameters.

        Classmethod which creates a single model visualizer object based on a
        full chain of parameter values, by reducing the chain to a single set
        of parameters (through the given `method`) and creating a `Model`
        from that to initialize this class with.

        Parameters
        ----------
        chain : np.ndarray[..., Nparams]
            Array containing chain of parameters values. Final axis must be
            of the size of the number of model parameters (13).

        observations : gcfit.Observations
            The `Observations` instance corresponding to this cluster.
            Will be passed to `FittableModel`.

        method : {"median", "mean", "final"}
            Method used to reduce the chain to a single set of parameters.
            "median" and "mean" find the average values, "final" will take the
            final iteration in the chain.

        Returns
        -------
        ModelVisualizer
            The created model visualization object.

        See Also
        --------
        gcfit.FittableModel : Model subclass used to initialize the model.
        '''

        reduc_methods = {'median': np.median, 'mean': np.mean,
                         'final': lambda ch, axis: ch[-1]}

        # if 3d (Niters, Nwalkers, Nparams)
        # if 2d (Nwalkers, Nparams)
        # if 1d (Nparams)
        chain = chain.reshape((-1, chain.shape[-1]))

        theta = reduc_methods[method](chain, axis=0)

        return cls(FittableModel(theta, observations), observations)

    @classmethod
    def from_theta(cls, theta, observations):
        '''Initialize a visualizer based on a single set of parameters.

        Classmethod which creates a single model visualizer object based on a
        set of parameter values, and uses that to initialize this class with.

        Parameters
        ----------
        theta : dict or list
            The set of model input parameters.
            Must either be a dict, or a full list of all 13 parameters.

        observations : gcfit.Observations
            The `Observations` instance corresponding to this cluster.
            Will be passed to `FittableModel`.

        Returns
        -------
        ModelVisualizer
            The created model visualization object.

        See Also
        --------
        gcfit.FittableModel : Model subclass used to initialize the model.
        '''
        return cls(FittableModel(theta, observations), observations)

    def __init__(self, model, observations=None):
        self.model = model
        self.obs = observations if observations else model.observations
        self.name = getattr(observations, 'cluster', 'Cluster Model')

        # various structural model attributes
        self.r0 = model.r0
        self.rh = model.rh
        self.rhp = model.rhp
        self.ra = model.ra
        self.rv = model.rv
        self.rt = model.rt
        self.mmean = model.mmean
        self.volume = model.volume

        # various fitting-related attributes
        self.F = model.theta['F']
        self.s2 = model.theta['s2']
        self.d = model.d

        self.r = model.r

        self.rlims = (9e-3, model.r.max().value + 5) << model.r.unit

        self._2πr = 2 * np.pi * model.r

        self.star_bin = model.nms - 1
        self.mj = model.mj

        # TODO if we have these here, should we have all quantites, to match CI?
        self.f_rem = model.rem.f
        self.f_BH = model.BH.f

        self.BH_rh = model.BH.rh
        self.spitzer_chi = model._spitzer_chi

        self.trh = model.trh
        self.N_relax = model.N_relax

        self.LOS = np.sqrt(self.model.v2pj)[:, np.newaxis, :]
        self.pm_T = np.sqrt(model.v2Tj)[:, np.newaxis, :]
        self.pm_R = np.sqrt(model.v2Rj)[:, np.newaxis, :]

        self.pm_tot = np.sqrt(0.5 * (self.pm_T**2 + self.pm_R**2))
        self.pm_ratio = self.pm_T / self.pm_R

        self._init_numdens(model, self.obs)
        self._init_massfunc(model, self.obs)

        self._init_surfdens(model, self.obs)
        self._init_dens(model, self.obs)

        self._init_mass_frac(model, self.obs)
        self._init_cum_mass(model, self.obs)

    # TODO alot of these init functions could be more homogenous
    @_ClusterVisualizer._support_units
    def _init_numdens(self, model, observations):
        '''Initialize number density quantities.'''

        model_nd = model.Sigmaj / model.mj[:, np.newaxis]

        nd = model_nd[:, np.newaxis, :]
        K = np.empty(nd.shape[0]) << u.dimensionless_unscaled  # one each mbin
        # TODO this K is only valid for the same mj as numdens obs anyways...

        # Check for observational numdens profiles, to compute scaling factors K
        #   but do not apply them to the numdens yet.
        if ((observations is not None)
                and (obs_nd := observations.filter_datasets('*number*'))):

            if len(obs_nd) > 1:
                mssg = ('Too many number density datasets, '
                        'computing scaling factor using only final dataset')
                logging.warning(mssg)

            obs_nd = list(obs_nd.values())[-1]
            obs_r = obs_nd['r'].to(model.r.unit)

            s2 = model.theta['s2'] << u.arcmin**-4
            obs_err = np.sqrt(obs_nd['ΔΣ']**2 + s2)

            for mbin in range(model_nd.shape[0]):

                nd_interp = util.QuantitySpline(model.r, model_nd[mbin, :])

                interpolated = nd_interp(obs_r).to(obs_nd['Σ'].unit)

                Kj = (np.nansum(obs_nd['Σ'] * interpolated / obs_err**2)
                      / np.nansum(interpolated**2 / obs_err**2))

                K[mbin] = Kj

        else:
            mssg = 'No number density datasets found, setting K=1'
            logging.info(mssg)

            K[:] = 1

        self.numdens = nd
        self.K_scale = K

    @_ClusterVisualizer._support_units
    def _init_massfunc(self, model, observations):
        '''Initialize mass function quantities, as dict of PIs.'''

        self.mass_func = {}

        # If no observations given, generate some demonstrative fields instead
        if observations is None:
            return self._spoof_empty_massfunc(model)

        PI_list = observations.filter_likelihoods(['*mass_function*'])

        densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                    for j in range(model.nms)]

        for i, (key, _, fields) in enumerate(PI_list):

            mf = observations[key]

            self.mass_func[key] = []

            clr = self.cmap(i / len(PI_list))

            rbins = np.unique(np.c_[mf['r1'], mf['r2']], axis=0)
            rbins.sort(axis=0)

            for field_slice, (r_in, r_out) in zip(fields, rbins):

                this_slc = {'r1': r_in, 'r2': r_out}

                this_slc['field'] = field_slice

                this_slc['colour'] = clr

                this_slc['dNdm'] = np.empty((1, model.nms))

                this_slc['mj'] = model.mj[:model.nms]

                sample_radii = field_slice._prev_sample.to(u.pc)

                for j in range(model.nms):

                    Nj = field_slice.MC_integrate(densityj[j], sample_radii)
                    widthj = (model.mj[j] * model.mbin_widths[j])

                    this_slc['dNdm'][0, j] = (Nj / widthj).value

                self.mass_func[key].append(this_slc)

    @_ClusterVisualizer._support_units
    def _spoof_empty_massfunc(self, model):
        '''Spoof an arbitrary massfunc, for use when no observations given'''
        import shapely

        densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                    for j in range(model.nms)]

        self.mass_func['Model'] = []

        limit = 3 * model.rh.to_value('pc')

        base = mass.Field(shapely.Point((0, 0)).buffer(10 * limit), unit='pc')

        domain = np.arange(0, limit, 1) * u.pc

        for r_in, r_out in np.c_[domain[:-1], domain[1:]]:

            this_slc = {'r1': r_in, 'r2': r_out}

            field_slice = base.slice_radially(r_in, r_out)

            this_slc['field'] = field_slice

            this_slc['colour'] = None

            this_slc['dNdm'] = np.empty((1, model.nms))

            this_slc['mj'] = model.mj[:model.nms]

            sample_radii = field_slice.MC_sample(5000).to(u.pc)

            for j in range(model.nms):

                Nj = field_slice.MC_integrate(densityj[j], sample_radii)
                widthj = (model.mj[j] * model.mbin_widths[j])

                this_slc['dNdm'][0, j] = (Nj / widthj).value

            self.mass_func['Model'].append(this_slc)

    @_ClusterVisualizer._support_units
    def _init_dens(self, model, observations):
        '''Initialize mass density quantities.'''

        shp = (np.newaxis, np.newaxis, slice(None))

        self.rho_tot = np.sum(model.rhoj, axis=0)[shp]
        self.rho_MS = np.sum(model.MS.rhoj, axis=0)[shp]
        self.rho_rem = np.sum(model.rem.rhoj, axis=0)[shp]
        self.rho_BH = np.sum(model.BH.rhoj, axis=0)[shp]
        self.rho_WD = np.sum(model.WD.rhoj, axis=0)[shp]
        self.rho_NS = np.sum(model.NS.rhoj, axis=0)[shp]

    @_ClusterVisualizer._support_units
    def _init_surfdens(self, model, observations):
        '''Initialize surface mass density quantities.'''

        shp = (np.newaxis, np.newaxis, slice(None))

        self.Sigma_tot = np.sum(model.Sigmaj, axis=0)[shp]
        self.Sigma_MS = np.sum(model.MS.Sigmaj, axis=0)[shp]
        self.Sigma_rem = np.sum(model.rem.Sigmaj, axis=0)[shp]
        self.Sigma_BH = np.sum(model.BH.Sigmaj, axis=0)[shp]
        self.Sigma_WD = np.sum(model.WD.Sigmaj, axis=0)[shp]
        self.Sigma_NS = np.sum(model.NS.Sigmaj, axis=0)[shp]

    @_ClusterVisualizer._support_units
    def _init_mass_frac(self, model, observations):
        '''Initialize mass fraction quantities.'''

        int_MS = util.QuantitySpline(self.r, self._2πr * self.Sigma_MS)
        int_rem = util.QuantitySpline(self.r, self._2πr * self.Sigma_rem)
        int_tot = util.QuantitySpline(self.r, self._2πr * self.Sigma_tot)

        mass_MS = np.empty((1, 1, self.r.size)) << u.Msun
        mass_rem = np.empty((1, 1, self.r.size)) << u.Msun
        mass_tot = np.empty((1, 1, self.r.size)) << u.Msun

        # TODO the rbins at the end always mess up fractional stuff, drop to 0
        mass_MS[0, 0, 0] = mass_rem[0, 0, 0] = mass_tot[0, 0, 0] = np.nan

        for i in range(1, self.r.size - 2):
            mass_MS[0, 0, i] = int_MS.integral(self.r[i], self.r[i + 1])
            mass_rem[0, 0, i] = int_rem.integral(self.r[i], self.r[i + 1])
            mass_tot[0, 0, i] = int_tot.integral(self.r[i], self.r[i + 1])

        self.frac_M_MS = mass_MS / mass_tot
        self.frac_M_rem = mass_rem / mass_tot

    @_ClusterVisualizer._support_units
    def _init_cum_mass(self, model, observations):
        '''Initialize cumulative mass quantities.'''

        int_tot = util.QuantitySpline(self.r, self._2πr * self.Sigma_tot)
        int_MS = util.QuantitySpline(self.r, self._2πr * self.Sigma_MS)
        int_BH = util.QuantitySpline(self.r, self._2πr * self.Sigma_BH)
        int_WD = util.QuantitySpline(self.r, self._2πr * self.Sigma_WD)
        int_NS = util.QuantitySpline(self.r, self._2πr * self.Sigma_NS)

        cum_tot = np.empty((1, 1, self.r.size)) << u.Msun
        cum_MS = np.empty((1, 1, self.r.size)) << u.Msun
        cum_BH = np.empty((1, 1, self.r.size)) << u.Msun
        cum_WD = np.empty((1, 1, self.r.size)) << u.Msun
        cum_NS = np.empty((1, 1, self.r.size)) << u.Msun

        for i in range(0, self.r.size):
            cum_tot[0, 0, i] = int_tot.integral(model.r[0], model.r[i])
            cum_MS[0, 0, i] = int_MS.integral(model.r[0], model.r[i])
            cum_BH[0, 0, i] = int_BH.integral(model.r[0], model.r[i])
            cum_WD[0, 0, i] = int_WD.integral(model.r[0], model.r[i])
            cum_NS[0, 0, i] = int_NS.integral(model.r[0], model.r[i])

        self.cum_M_tot = cum_tot
        self.cum_M_MS = cum_MS
        self.cum_M_WD = cum_WD
        self.cum_M_NS = cum_NS
        self.cum_M_BH = cum_BH


class CIModelVisualizer(_ClusterVisualizer):
    '''Analysis and visualization of a model, with confidence intervals.

    Provides a number of plotting methods useful for the analysis of a
    (multimass) model, and it's fit to all relevant observational datasets.
    This class is based on a full set of parameter chains, which allows for the
    estimation of errors on all plotted quantities, shown as confidence
    intervals shaded on all plots.

    A number of relevant (plottable) quantities (and their ±1σ and ±2σ
    confidence intervals) will be initially computed and
    stored to be used by the various plotting functions defined by the base
    class `_ClusterVisualizer`.

    Due to the time required to compute the large number of models needed, this
    class can also be saved to disk (through the `save` method) and later
    reloaded (through the `load` classmethod) instantly.

    See Also
    --------
    _ClusterVisualizer : Base class providing all common plotting functions.

    Notes
    -----
    This class should not be initialized directly, but through the `from_chain`
    classmethod.
    '''

    @_ClusterVisualizer._support_units
    def plot_f_rem(self, fig=None, ax=None, bins='auto', color='tab:blue',
                   verbose_label=True):
        r'''Plot the remnant fraction of this model.

        Plots a histogram of the values of the total remnant mass fraction
        (i.e. mass fraction in WD, NS, BH) in the given chain of models.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_ClusterVisualizer._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this quantity. Should be a
            part of the given `fig`.

        bins : int or sequence or str
            Bins to use creating the histogram of this quantity.
            See `plt.hist` for more details.

        color : color, optional
            The colour of the plotted histogram. This colour will be applied to
            the edge (border) of the histogram as is, and to the face at 33%
            transparency.

        verbose_label : bool, optional
            If True (default), quantity label will be "Remnant Fraction",
            otherwise "$f_{\mathrm{remn}}$".

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        color = mpl_clr.to_rgb(color)
        facecolor = color + (0.33, )

        ax.hist(self.f_rem, histtype='stepfilled',
                bins=bins, ec=color, fc=facecolor, lw=2)

        if verbose_label:
            label = "Remnant Fraction"
        else:
            label = r'$f_{\mathrm{remn}}$'

        self._set_xlabel(ax, label, unit=self.f_rem.unit)

        return fig

    @_ClusterVisualizer._support_units
    def plot_f_BH(self, fig=None, ax=None, bins='auto', color='tab:blue',
                  verbose_label=True):
        r'''Plot the BH fraction of this model.

        Plots a histogram of the values of the total black hole mass fraction
        (i.e. mass fraction in BH over total mass) in the given chain of models.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_ClusterVisualizer._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this quantity. Should be a
            part of the given `fig`.

        bins : int or sequence or str
            Bins to use creating the histogram of this quantity.
            See `plt.hist` for more details.

        color : color, optional
            The colour of the plotted histogram. This colour will be applied to
            the edge (border) of the histogram as is, and to the face at 33%
            transparency.

        verbose_label : bool, optional
            If True (default), quantity label will be "BH Mass Fraction",
            otherwise "$f_{\mathrm{BH}}$".

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        color = mpl_clr.to_rgb(color)
        facecolor = color + (0.33, )

        ax.hist(self.f_BH, histtype='stepfilled',
                bins=bins, ec=color, fc=facecolor, lw=2)

        if verbose_label:
            label = "BH Mass Fraction"
        else:
            label = r'$f_{\mathrm{BH}}$'

        self._set_xlabel(ax, label, unit=self.f_BH.unit)

        return fig

    @_ClusterVisualizer._support_units
    def plot_BH_mass(self, fig=None, ax=None, bins='auto', color='tab:blue',
                     verbose_label=True):
        r'''Plot the BH mass of this model.

        Plots a histogram of the values of the total black hole mass in the
        given chain of models.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_ClusterVisualizer._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this quantity. Should be a
            part of the given `fig`.

        bins : int or sequence or str
            Bins to use creating the histogram of this quantity.
            See `plt.hist` for more details.

        color : color, optional
            The colour of the plotted histogram. This colour will be applied to
            the edge (border) of the histogram as is, and to the face at 33%
            transparency.

        verbose_label : bool, optional
            If True (default), quantity label will be "BH Mass",
            otherwise "$\mathrm{M}_{\mathrm{BH}}$".

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        color = mpl_clr.to_rgb(color)
        facecolor = color + (0.33, )

        ax.hist(self.BH_mass, histtype='stepfilled',
                bins=bins, ec=color, fc=facecolor, lw=2)

        if verbose_label:
            label = "BH Mass"
        else:
            label = r'$\mathrm{M}_{\mathrm{BH}}$'

        self._set_xlabel(ax, label, unit=self.BH_mass.unit)

        return fig

    @_ClusterVisualizer._support_units
    def plot_BH_num(self, fig=None, ax=None, bins='auto', color='tab:blue',
                    verbose_label=True):
        r'''Plot the number of BHs in this model.

        Plots a histogram of the values of the total amount of black holes
        in the given chain of models.

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_ClusterVisualizer._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot this quantity. Should be a
            part of the given `fig`.

        bins : int or sequence or str
            Bins to use creating the histogram of this quantity.
            See `plt.hist` for more details.

        color : color, optional
            The colour of the plotted histogram. This colour will be applied to
            the edge (border) of the histogram as is, and to the face at 33%
            transparency.

        verbose_label : bool, optional
            If True (default), quantity label will be "BH Amount",
            otherwise "$\mathrm{N}_{\mathrm{BH}}$".

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        fig, ax = self._setup_artist(fig, ax)

        color = mpl_clr.to_rgb(color)
        facecolor = color + (0.33, )

        ax.hist(self.BH_num, histtype='stepfilled',
                bins=bins, ec=color, fc=facecolor, lw=2)

        if verbose_label:
            label = "BH Amount"
        else:
            label = r'$\mathrm{N}_{\mathrm{BH}}$'

        self._set_xlabel(ax, label, unit=self.BH_num.unit)

        return fig

    def __init__(self, observations):
        self.obs = observations
        self.name = observations.cluster

    @classmethod
    def from_chain(cls, chain, observations, N=100, *,
                   verbose=False, pool=None):
        '''Initialize a CI visualizer based on a full chain of parameters.

        Classmethod which creates a model visualizer object based on a
        full chain of parameter values, by computing a model for each step in
        the chain and determining the median and 1,2σ uncertainties, on each
        relevant quantity, between all the models.

        This function will take some time, as models must be computed for
        every step in the chain. Parallelizing it via a given multiprocessing
        `pool` is recommended.

        Parameters
        ----------
        chain : np.ndarray[..., Nparams]
            Array containing chain of parameters values. Final axis must be
            of the size of the number of model parameters (13).

        observations : gcfit.Observations
            The `Observations` instance corresponding to this cluster.
            Will be used to initialize every model in the chain.

        N : int
            The number of samples to use in computing the confidence intervals.
            The chosen number of samples will be taken from the end of the
            chain (`chain[..., -N:]`). Choose an N greater than the size of
            the chain to use all samples.

        verbose : bool, optional
            Optionally show a `tqdm` loading bar while computing the models.

        pool : Pool, optional
            A `multiprocessing.Pool` (or other pool-like object implementing
            an `imap_unordered` method) used to parallelize the computation
            of the models.

        Returns
        -------
        CIModelVisualizer
            The created CI model visualization object.

        See Also
        --------
        gcfit.FittableModel
            Model subclass used to initialize each model.

        gcfit.scripts.generate_model_CI
            Script used to generate and save model CIs for a number of clusters
            at once.
        '''

        import functools

        viz = cls(observations)

        # ------------------------------------------------------------------
        # Get info about the chain and set of models
        # ------------------------------------------------------------------

        # Flatten walkers, if not already
        chain = chain.reshape((-1, chain.shape[-1]))[-N:]

        # Truncate if N is larger than the given chain size
        N = chain.shape[0]

        viz.N = N

        median_chain = np.median(chain, axis=0)

        # TODO get these indices more dynamically
        viz.F = median_chain[7]
        viz.s2 = median_chain[6]
        viz.d = median_chain[12] << u.kpc

        # Setup the radial domain to interpolate everything onto
        # We estimate the maximum radius needed will be given by the model with
        # the largest value of the truncation parameter "g". This should be a
        # valid enough assumption for our needs. While we have it, we'll also
        # use this model to grab the other values we need, which shouldn't
        # change much between models, so using this extreme model is okay.
        # warning: in very large N samples, this g might be huge, and lead to a
        # very large rt. I'm not really sure yet how that might affect the CIs
        # or plots

        huge_model = FittableModel(chain[np.argmax(chain[:, 4])], viz.obs)

        viz.r = np.r_[0, np.geomspace(1e-5, huge_model.rt.value, 99)] << u.pc

        viz.rlims = (9e-3, viz.r.max().value + 5) << viz.r.unit

        # Assume that this example model has same nms bin as all models
        # This approximation isn't exactly correct (especially when Ndot != 0),
        # but close enough for plots
        viz.star_bin = 0

        # mj only contains nms and tracer bins (the only ones we plot anyways)
        # TODO right now tracers only used for velocities, and not numdens
        mj_MS = huge_model.mj[huge_model._star_bins][-1]
        mj_tracer = huge_model.mj[huge_model._tracer_bins]

        viz.mj = np.r_[mj_MS, mj_tracer]

        # ------------------------------------------------------------------
        # Setup the final full parameters arrays with dims of
        # [mass bins, intervals (from percentile of models), radial bins] for
        # all "profile" datasets
        # ------------------------------------------------------------------

        Nr = viz.r.size

        # velocities

        vel_unit = np.sqrt(huge_model.v2Tj).unit

        Nm = 1 + len(mj_tracer)

        vpj = np.full((Nm, N, Nr), np.nan) << vel_unit
        vTj, vRj, vtotj = vpj.copy(), vpj.copy(), vpj.copy()

        vaj = np.full((Nm, N, Nr), np.nan) << u.dimensionless_unscaled

        # mass density

        rho_unit = huge_model.rhoj.unit

        rho_tot = np.full((1, N, Nr), np.nan) << rho_unit
        rho_MS, rho_BH = rho_tot.copy(), rho_tot.copy()
        rho_WD, rho_NS = rho_tot.copy(), rho_tot.copy()

        # surface density

        Sigma_unit = huge_model.Sigmaj.unit

        Sigma_tot = np.full((1, N, Nr), np.nan) << Sigma_unit
        Sigma_MS, Sigma_BH = Sigma_tot.copy(), Sigma_tot.copy()
        Sigma_WD, Sigma_NS = Sigma_tot.copy(), Sigma_tot.copy()

        # Cumulative mass

        mass_unit = huge_model.M.unit

        cum_M_tot = np.full((1, N, Nr), np.nan) << mass_unit
        cum_M_MS, cum_M_BH = cum_M_tot.copy(), cum_M_tot.copy()
        cum_M_WD, cum_M_NS = cum_M_tot.copy(), cum_M_tot.copy()

        # Mass Fraction

        frac_M_MS = np.full((1, N, Nr), np.nan) << u.dimensionless_unscaled
        frac_M_rem = frac_M_MS.copy()

        f_rem = np.full(N, np.nan) << u.pct
        f_BH = np.full(N, np.nan) << u.pct

        # number density

        numdens = np.full((1, N, Nr), np.nan) << u.pc**-2
        K_scale = np.full((1,), np.nan) << u.dimensionless_unscaled
        # K_scale = np.full((Nm), np.nan) << u.Unit('pc2 / arcmin2')

        # mass function

        massfunc = viz._prep_massfunc(viz.obs)

        # massfunc = np.empty((N, N_rbins, huge_model.nms))

        for rbins in massfunc.values():
            for rslice in rbins:
                rslice['mj'] = huge_model.mj[:huge_model.nms]
                rslice['dNdm'] = np.full((N, huge_model.nms), np.nan)

        # BH mass

        BH_mass = np.full(N, np.nan) << u.Msun
        BH_num = np.full(N, np.nan) << u.dimensionless_unscaled

        # Structural params

        r0 = np.full(N, np.nan) << huge_model.r0.unit
        rt = np.full(N, np.nan) << huge_model.rt.unit
        rh = np.full(N, np.nan) << huge_model.rh.unit
        rhp = np.full(N, np.nan) << huge_model.rhp.unit
        ra = np.full(N, np.nan) << huge_model.ra.unit
        rv = np.full(N, np.nan) << huge_model.rv.unit
        mmean = np.full(N, np.nan) << huge_model.mmean.unit
        volume = np.full(N, np.nan) << huge_model.volume.unit

        # BH derived quantities

        BH_rh = np.full(N, np.nan) << huge_model.BH.rh.unit
        spitz_chi = np.full(N, np.nan) << u.dimensionless_unscaled

        # Relaxation times

        trh = np.full(N, np.nan) << u.Gyr
        N_relax = np.full(N, np.nan) << u.dimensionless_unscaled

        # ------------------------------------------------------------------
        # Setup iteration and pooling
        # ------------------------------------------------------------------

        get_model = functools.partial(_get_model, observations=viz.obs)

        try:
            _map = map if pool is None else pool.imap_unordered
        except AttributeError:
            mssg = ("Invalid pool, currently only support pools with an "
                    "`imap_unordered` method")
            raise ValueError(mssg)

        if verbose:
            import tqdm
            loader = tqdm.tqdm(enumerate(_map(get_model, chain)), total=N)

        else:
            loader = enumerate(_map(get_model, chain))

        # ------------------------------------------------------------------
        # iterate over all models in the sample and compute/store their
        # relevant parameters
        # ------------------------------------------------------------------

        for model_ind, model in loader:

            if model is None:
                # TODO would be better to extend chain so N are still computed
                # for now this ind will be filled with nan
                continue

            equivs = util.angular_width(model.d)

            # Velocities

            # convoluted way of going from a slice to a list of indices
            tracers = list(range(len(model.mj))[model._tracer_bins])

            for i, mass_bin in enumerate([model.nms - 1] + tracers):

                slc = (i, model_ind, slice(None))

                vTj[slc], vRj[slc], vtotj[slc], \
                    vaj[slc], vpj[slc] = viz._init_velocities(model, mass_bin)

            slc = (0, model_ind, slice(None))

            # Mass Densities

            rho_MS[slc], rho_tot[slc], rho_BH[slc], \
                rho_WD[slc], rho_NS[slc] = viz._init_dens(model)

            # Surface Densities

            Sigma_MS[slc], Sigma_tot[slc], Sigma_BH[slc], \
                Sigma_WD[slc], Sigma_NS[slc] = viz._init_surfdens(model)

            # Cumulative Mass distribution

            cum_M_MS[slc], cum_M_tot[slc], cum_M_BH[slc], \
                cum_M_WD[slc], cum_M_NS[slc] = viz._init_cum_mass(model)

            # Number Densities

            numdens[slc] = viz._init_numdens(model, equivs=equivs)

            # Mass Functions
            for rbins in massfunc.values():
                for rslice in rbins:

                    mf = rslice['dNdm']
                    mf[model_ind, ...] = viz._init_dNdm(model, rslice, equivs)

            # Mass Fractions

            frac_M_MS[slc], frac_M_rem[slc] = viz._init_mass_frac(model)

            f_rem[model_ind] = model.rem.f
            f_BH[model_ind] = model.BH.f

            # Black holes

            BH_mass[model_ind] = np.sum(model.BH.Mj)
            BH_num[model_ind] = np.sum(model.BH.Nj)

            # Structural params

            r0[model_ind] = model.r0
            rt[model_ind] = model.rt
            rh[model_ind] = model.rh
            rhp[model_ind] = model.rhp
            ra[model_ind] = model.ra
            rv[model_ind] = model.rv
            mmean[model_ind] = model.mmean
            volume[model_ind] = model.volume

            BH_rh[model_ind] = model.BH.rh
            spitz_chi[model_ind] = model._spitzer_chi

            trh[model_ind] = model.trh
            N_relax[model_ind] = model.N_relax

        # ------------------------------------------------------------------
        # compute and store the percentiles and medians
        # ------------------------------------------------------------------

        q = [97.72, 84.13, 50., 15.87, 2.28]

        axes = (1, 0, 2)  # `np.percentile` messes up the dimensions

        perc = np.nanpercentile

        viz.pm_T = np.transpose(perc(vTj, q, axis=1), axes)
        viz.pm_R = np.transpose(perc(vRj, q, axis=1), axes)
        viz.pm_tot = np.transpose(perc(vtotj, q, axis=1), axes)
        viz.pm_ratio = np.transpose(perc(vaj, q, axis=1), axes)
        viz.LOS = np.transpose(perc(vpj, q, axis=1), axes)

        viz.rho_MS = np.transpose(perc(rho_MS, q, axis=1), axes)
        viz.rho_tot = np.transpose(perc(rho_tot, q, axis=1), axes)
        viz.rho_BH = np.transpose(perc(rho_BH, q, axis=1), axes)
        viz.rho_WD = np.transpose(perc(rho_WD, q, axis=1), axes)
        viz.rho_NS = np.transpose(perc(rho_NS, q, axis=1), axes)

        viz.Sigma_MS = np.transpose(perc(Sigma_MS, q, axis=1), axes)
        viz.Sigma_tot = np.transpose(perc(Sigma_tot, q, axis=1), axes)
        viz.Sigma_BH = np.transpose(perc(Sigma_BH, q, axis=1), axes)
        viz.Sigma_WD = np.transpose(perc(Sigma_WD, q, axis=1), axes)
        viz.Sigma_NS = np.transpose(perc(Sigma_NS, q, axis=1), axes)

        viz.cum_M_MS = np.transpose(perc(cum_M_MS, q, axis=1), axes)
        viz.cum_M_tot = np.transpose(perc(cum_M_tot, q, axis=1), axes)
        viz.cum_M_BH = np.transpose(perc(cum_M_BH, q, axis=1), axes)
        viz.cum_M_WD = np.transpose(perc(cum_M_WD, q, axis=1), axes)
        viz.cum_M_NS = np.transpose(perc(cum_M_NS, q, axis=1), axes)

        viz.numdens = np.transpose(perc(numdens, q, axis=1), axes)
        K_scale[:] = viz._init_K_scale(viz.numdens)
        viz.K_scale = K_scale

        viz.mass_func = massfunc

        for rbins in viz.mass_func.values():
            for rslice in rbins:

                rslice['dNdm'] = perc(rslice['dNdm'], q, axis=0)

        viz.frac_M_MS = perc(frac_M_MS, q, axis=1)
        viz.frac_M_rem = perc(frac_M_rem, q, axis=1)

        viz.f_rem = f_rem
        viz.f_BH = f_BH

        viz.BH_mass = BH_mass
        viz.BH_num = BH_num

        viz.r0 = r0
        viz.rt = rt
        viz.rh = rh
        viz.rhp = rhp
        viz.ra = ra
        viz.rv = rv
        viz.mmean = mmean
        viz.volume = volume

        viz.BH_rh = BH_rh
        viz.spitzer_chi = spitz_chi

        viz.trh = trh
        viz.N_relax = N_relax

        return viz

    def _init_velocities(self, model, mass_bin):
        '''Initialize various velocity quantities.'''

        vT = np.sqrt(model.v2Tj[mass_bin])
        vT_interp = util.QuantitySpline(model.r, vT)
        vT = vT_interp(self.r)

        vR = np.sqrt(model.v2Rj[mass_bin])
        vR_interp = util.QuantitySpline(model.r, vR)
        vR = vR_interp(self.r)

        vtot = np.sqrt(0.5 * (model.v2Tj[mass_bin] + model.v2Rj[mass_bin]))
        vtot_interp = util.QuantitySpline(model.r, vtot)
        vtot = vtot_interp(self.r)

        va = np.sqrt(model.v2Tj[mass_bin] / model.v2Rj[mass_bin])
        finite = np.isnan(va)
        va_interp = util.QuantitySpline(model.r[~finite], va[~finite], ext=3)
        va = va_interp(self.r)

        vp = np.sqrt(model.v2pj[mass_bin])
        vp_interp = util.QuantitySpline(model.r, vp)
        vp = vp_interp(self.r)

        return vT, vR, vtot, va, vp

    def _init_dens(self, model):
        '''Initialize mass density quantities.'''

        rho_MS = np.sum(model.MS.rhoj, axis=0)
        rho_MS_interp = util.QuantitySpline(model.r, rho_MS)
        rho_MS = rho_MS_interp(self.r)

        rho_tot = np.sum(model.rhoj, axis=0)
        rho_tot_interp = util.QuantitySpline(model.r, rho_tot)
        rho_tot = rho_tot_interp(self.r)

        rho_BH = np.sum(model.BH.rhoj, axis=0)
        rho_BH_interp = util.QuantitySpline(model.r, rho_BH)
        rho_BH = rho_BH_interp(self.r)

        rho_WD = np.sum(model.WD.rhoj, axis=0)
        rho_WD_interp = util.QuantitySpline(model.r, rho_WD)
        rho_WD = rho_WD_interp(self.r)

        rho_NS = np.sum(model.NS.rhoj, axis=0)
        rho_NS_interp = util.QuantitySpline(model.r, rho_NS)
        rho_NS = rho_NS_interp(self.r)

        return rho_MS, rho_tot, rho_BH, rho_WD, rho_NS

    def _init_surfdens(self, model):
        '''Initialize surface mass density quantities.'''

        Sigma_MS = np.sum(model.MS.Sigmaj, axis=0)
        Sigma_MS_interp = util.QuantitySpline(model.r, Sigma_MS)
        Sigma_MS = Sigma_MS_interp(self.r)

        Sigma_tot = np.sum(model.Sigmaj, axis=0)
        Sigma_tot_interp = util.QuantitySpline(model.r, Sigma_tot)
        Sigma_tot = Sigma_tot_interp(self.r)

        Sigma_BH = np.sum(model.BH.Sigmaj, axis=0)
        Sigma_BH_interp = util.QuantitySpline(model.r, Sigma_BH)
        Sigma_BH = Sigma_BH_interp(self.r)

        Sigma_WD = np.sum(model.WD.Sigmaj, axis=0)
        Sigma_WD_interp = util.QuantitySpline(model.r, Sigma_WD)
        Sigma_WD = Sigma_WD_interp(self.r)

        Sigma_NS = np.sum(model.NS.Sigmaj, axis=0)
        Sigma_NS_interp = util.QuantitySpline(model.r, Sigma_NS)
        Sigma_NS = Sigma_NS_interp(self.r)

        return Sigma_MS, Sigma_tot, Sigma_BH, Sigma_WD, Sigma_NS

    def _init_cum_mass(self, model):
        '''Initialize cumulative mass quantities.'''

        # TODO it seems like the integrated mass is a bit less than total Mj?
        # TODO why doing all this instead of using model.mc?

        _2πr = 2 * np.pi * model.r

        cum_M_MS = _2πr * np.sum(model.MS.Sigmaj, axis=0)
        cum_M_MS_interp = util.QuantitySpline(model.r, cum_M_MS)
        cum_M_MS = [cum_M_MS_interp.integral(self.r[0], ri) for ri in self.r]

        cum_M_tot = _2πr * np.sum(model.Sigmaj, axis=0)
        cum_M_tot_interp = util.QuantitySpline(model.r, cum_M_tot)
        cum_M_tot = [cum_M_tot_interp.integral(self.r[0], ri) for ri in self.r]

        cum_M_BH = _2πr * np.sum(model.BH.Sigmaj, axis=0)
        cum_M_BH_interp = util.QuantitySpline(model.r, cum_M_BH)
        cum_M_BH = [cum_M_BH_interp.integral(self.r[0], ri) for ri in self.r]

        cum_M_WD = _2πr * np.sum(model.WD.Sigmaj, axis=0)
        cum_M_WD_interp = util.QuantitySpline(model.r, cum_M_WD)
        cum_M_WD = [cum_M_WD_interp.integral(self.r[0], ri) for ri in self.r]

        cum_M_NS = _2πr * np.sum(model.NS.Sigmaj, axis=0)
        cum_M_NS_interp = util.QuantitySpline(model.r, cum_M_NS)
        cum_M_NS = [cum_M_NS_interp.integral(self.r[0], ri) for ri in self.r]

        return cum_M_MS, cum_M_tot, cum_M_BH, cum_M_WD, cum_M_NS

    def _init_mass_frac(self, model):
        '''Initialize mass fraction quantities.'''

        _2πr = 2 * np.pi * model.r

        dens_tot = _2πr * np.sum(model.Sigmaj, axis=0)
        int_tot = util.QuantitySpline(model.r, dens_tot)
        mass_MS = np.empty((1, self.r.size))

        dens_MS = _2πr * np.sum(model.MS.Sigmaj, axis=0)
        int_MS = util.QuantitySpline(model.r, dens_MS)
        mass_rem = np.empty((1, self.r.size))

        dens_rem = _2πr * np.sum(model.rem.Sigmaj, axis=0)
        int_rem = util.QuantitySpline(model.r, dens_rem)
        mass_tot = np.empty((1, self.r.size))

        mass_MS[0, 0] = mass_rem[0, 0] = mass_tot[0, 0] = np.nan

        for i in range(1, self.r.size - 2):
            mass_MS[0, i] = int_MS.integral(self.r[i], self.r[i + 1]).value
            mass_rem[0, i] = int_rem.integral(self.r[i], self.r[i + 1]).value
            mass_tot[0, i] = int_tot.integral(self.r[i], self.r[i + 1]).value

        return mass_MS / mass_tot, mass_rem / mass_tot

    def _init_numdens(self, model, equivs=None):
        '''Initialize number density quantities.'''

        model_nd = model.Sigmaj[model.nms - 1] / model.mj[model.nms - 1]

        nd_interp = util.QuantitySpline(model.r, model_nd)

        return nd_interp(self.r).to('pc-2', equivs)

    def _init_K_scale(self, numdens):
        '''Initialize the K scale on the number density.'''

        nd_interp = util.QuantitySpline(self.r, self._get_median(numdens[0]))

        equivs = util.angular_width(self.d)

        if obs_nd := self.obs.filter_datasets('*number_density*'):

            if len(obs_nd) > 1:
                mssg = ('Too many number density datasets, '
                        'computing scaling factor using only final dataset')
                logging.warning(mssg)

            obs_nd = list(obs_nd.values())[-1]
            obs_r = obs_nd['r'].to(self.r.unit, equivs)

            # TODO this s2 isn't technically 100% accurate here
            s2 = self.s2 << u.arcmin**-4
            obs_err = np.sqrt(obs_nd['ΔΣ']**2 + s2)

            interpolated = nd_interp(obs_r).to(obs_nd['Σ'].unit, equivs)

            K = (np.nansum(obs_nd['Σ'] * interpolated / obs_err**2)
                 / np.nansum(interpolated**2 / obs_err**2))

        else:
            mssg = 'No number density datasets found, setting K=1'
            logging.info(mssg)

            K = 1

        return K

    def _prep_massfunc(self, observations):
        '''Prepare the mass function dictionaries to be filled in.'''

        massfunc = {}

        PI_list = observations.filter_likelihoods(['*mass_function*'])

        for i, (key, _, fields) in enumerate(PI_list):

            mf = observations[key]

            massfunc[key] = []

            clr = self.cmap(i / len(PI_list))

            rbins = np.unique(np.c_[mf['r1'], mf['r2']], axis=0)
            rbins.sort(axis=0)

            for field_slice, (r_in, r_out) in zip(fields, rbins):

                this_slc = {'r1': r_in, 'r2': r_out}

                this_slc['field'] = field_slice

                this_slc['colour'] = clr

                massfunc[key].append(this_slc)

        return massfunc

    def _init_dNdm(self, model, rslice, equivs=None):
        '''Compute the dN/dm for this model in a given mass function slice.'''

        densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                    for j in range(model.nms)]

        with u.set_enabled_equivalencies(equivs):
            sample_radii = rslice['field']._prev_sample.to(u.pc)

            dNdm = np.empty(model.nms)

            for j in range(model.nms):
                Nj = rslice['field'].MC_integrate(densityj[j], sample_radii)
                widthj = (model.mj[j] * model.mbin_widths[j])
                dNdm[j] = (Nj / widthj).value

        return dNdm

    # ----------------------------------------------------------------------
    # Save and load confidence intervals to a file
    # ----------------------------------------------------------------------

    def save(self, filename, overwrite=False):
        '''Save model CI quantities to a file for quicker future initialization.

        Saves the computed confidence intervals on all model profiles, and the
        full chains of computed quantities (e.g. BH mass, number), so that
        the same CIModelVisualizer can be initialized later (through the
        `CIModelVisualizer.load` classmethod) without needing to recompute a
        large number of models again.

        The desired file must be an existing HDF5 file. All quantities will
        be stored under a "model_output" group, in the root group.
        The easiest file to use may be the run output file for the cluster fit
        that this CI model was based on.
        Note that an existing saved CI model within this file will be
        quietly overwritten if `overwrite=True`.

        The full set of models used to initilize this class *will not* be
        stored, only the quantities required to initialize this class.

        Parameters
        ----------
        filename : pathlib.Path or str
            Path to the output HDF5 file to save this CI model to.

        overwrite : bool, optional
            If True, will overwrite any existing saved model outputs in this
            file. By default raises a ValueError if model output exists already.
        '''

        with h5py.File(filename, 'a') as file:

            # --------------------------------------------------------------
            # Prep file
            # --------------------------------------------------------------

            if 'model_output' in file:
                mssg = f'Model output already exists in {filename}.'

                if overwrite is True:
                    logging.info(mssg + ' Overwriting.')
                    del file['model_output']
                else:
                    mssg += ' Set `overwrite=True` to overwrite.'
                    raise ValueError(mssg)

            modelgrp = file.create_group('model_output')

            # --------------------------------------------------------------
            # Store metadata
            # --------------------------------------------------------------

            meta_grp = modelgrp.create_group('metadata')

            meta_grp.create_dataset('r', data=self.r)
            meta_grp.create_dataset('star_bin', data=self.star_bin)
            meta_grp.create_dataset('mj', data=self.mj)
            meta_grp.attrs['rlims'] = self.rlims.to_value('pc')
            meta_grp.attrs['s2'] = self.s2
            meta_grp.attrs['F'] = self.F
            meta_grp.attrs['d'] = self.d.to_value('kpc')
            meta_grp.attrs['N'] = self.N
            meta_grp.attrs['cluster'] = self.obs.cluster

            # --------------------------------------------------------------
            # Store profiles
            # --------------------------------------------------------------

            prof_grp = modelgrp.create_group('profiles')

            profile_keys = (
                'rho_MS', 'rho_tot', 'rho_BH', 'rho_WD', 'rho_NS',
                'pm_T', 'pm_R', 'pm_tot', 'pm_ratio', 'LOS',
                'Sigma_MS', 'Sigma_tot', 'Sigma_BH', 'Sigma_WD',
                'Sigma_NS', 'cum_M_MS', 'cum_M_tot', 'cum_M_BH', 'cum_M_WD',
                'cum_M_NS', 'frac_M_MS', 'frac_M_rem', 'numdens'
            )

            for key in profile_keys:

                data = getattr(self, key)
                ds = prof_grp.create_dataset(key, data=data)
                ds.attrs['unit'] = data.unit.to_string()

            # --------------------------------------------------------------
            # Store quantities
            # --------------------------------------------------------------

            quant_grp = modelgrp.create_group('quantities')

            quant_keys = (
                'f_rem', 'f_BH', 'BH_mass', 'BH_num',
                'r0', 'rt', 'rh', 'rhp', 'ra', 'rv', 'mmean', 'volume',
                'BH_rh', 'spitzer_chi', 'trh', 'N_relax', 'K_scale'
            )

            for key in quant_keys:

                data = getattr(self, key)
                ds = quant_grp.create_dataset(key, data=data)
                ds.attrs['unit'] = data.unit.to_string()

            # --------------------------------------------------------------
            # Store mass function
            # --------------------------------------------------------------

            mf_grp = modelgrp.create_group('mass_func')

            for PI in self.mass_func:

                # remove the 'mass_function' tag, as the slash messes up h5py
                PI_grp = mf_grp.create_group(PI.split('/')[-1])

                for rind, rbin in enumerate(self.mass_func[PI]):
                    # can't save the field object here, must compute in load

                    slc_grp = PI_grp.create_group(str(rind))

                    ds = slc_grp.create_dataset('r1', data=rbin['r1'])
                    ds.attrs['unit'] = rbin['r1'].unit.to_string()

                    ds = slc_grp.create_dataset('r2', data=rbin['r2'])
                    ds.attrs['unit'] = rbin['r2'].unit.to_string()

                    ds = slc_grp.create_dataset('mj', data=rbin['mj'])
                    ds.attrs['unit'] = rbin['mj'].unit.to_string()

                    slc_grp.create_dataset('colour', data=rbin['colour'])

                    slc_grp.create_dataset('dNdm', data=rbin['dNdm'])

    @classmethod
    def load(cls, filename, observations=None):
        '''Initialize this class by loading already saved model outputs.

        Based on model outputs computed before (by initializing this class
        through the `CIModelVisualizer.from_chain` classmethod) and
        saved using the `CIModelVisualizer.save` method, loads in all required
        model profiles and quantities from the given file and initializes this
        class based on them.

        Required quantities are read from the given HDF5 file under the
        "/model_output" group.
        Note that if this group was not created by the `save` method, unintended
        errors may arise only later, during some plotting.

        Parameters
        ----------
        filename : pathlib.Path or str
            Path to the HDF5 file containing the saved "model_output".

        observations : None or gcfit.Observations, optional
            The `Observations` instance corresponding to this cluster.
            If not given, an attempt will be made to create this class based on
            the stored cluster name.

        Returns
        -------
        CIModelVisualizer
            The loaded CI model visualization object.
        '''

        with h5py.File(filename, 'r') as file:

            try:
                modelgrp = file['model_output']
            except KeyError as err:
                mssg = f'No saved model outputs in {filename}'
                raise RuntimeError(mssg) from err

            # init class
            if (obs := observations) is None:
                restrict = modelgrp['metadata'].attrs.get('restrict_to', None)
                obs = Observations(modelgrp['metadata'].attrs['cluster'],
                                   restrict_to=restrict)

            viz = cls(obs)

            # Get metadata
            viz.N = modelgrp['metadata'].attrs['N']
            viz.s2 = modelgrp['metadata'].attrs['s2']
            viz.F = modelgrp['metadata'].attrs['F']
            viz.d = modelgrp['metadata'].attrs['d'] << u.kpc
            viz.rlims = modelgrp['metadata'].attrs['rlims'] << u.pc

            viz.r = modelgrp['metadata']['r'][:] << u.pc
            viz.star_bin = modelgrp['metadata']['star_bin'][()]
            viz.mj = modelgrp['metadata']['mj'][:] << u.Msun

            # Get profile and quantity percentiles
            for grp in ('profiles', 'quantities'):

                for key in modelgrp[grp]:

                    value = modelgrp[grp][key][:]

                    try:
                        value *= u.Unit(modelgrp[grp][key].attrs['unit'])
                    except KeyError:
                        pass

                    setattr(viz, key, value)

            # get mass func percentiles and generate the fields

            viz.mass_func = {}

            cen = (viz.obs.mdata['RA'], viz.obs.mdata['DEC'])

            for PI in modelgrp['mass_func']:

                # add the mass_function tag back in
                viz.mass_func[f'mass_function/{PI}'] = []

                field = mass.Field.from_dataset(viz.obs[f'mass_function/{PI}'],
                                                cen=cen)

                for rind in modelgrp['mass_func'][PI]:

                    rbin = modelgrp['mass_func'][PI][rind]

                    slc = {
                        'r1': rbin['r1'][()] * u.Unit(rbin['r1'].attrs['unit']),
                        'r2': rbin['r2'][()] * u.Unit(rbin['r2'].attrs['unit']),
                        'mj': rbin['mj'][()] * u.Unit(rbin['mj'].attrs['unit']),
                        'colour': rbin['colour'][:],
                        'dNdm': rbin['dNdm'][:],
                    }

                    slc['field'] = field.slice_radially(slc['r1'], slc['r2'])

                    viz.mass_func[f'mass_function/{PI}'].append(slc)

        return viz


class ObservationsVisualizer(_ClusterVisualizer):
    '''Analysis and visualization of a observational data, alone.

    Provides a number of plotting methods useful for the analysis of a
    observational datasets, without the need for a corresponding model.
    This class provided the ability to plot (nearly) all of the usual profiles
    provided by `ModelVisualizer`, but only showing the relevant observational
    datasets in each. There are of course a number of model based quantities
    (with no corresponding observables) which cannot be plotted through this,
    and which will simply raise an error when called.

    Parameters
    ----------

    observations : gcfit.Observations
        The `Observations` instance corresponding to the cluster (data) to
        be analyzed.

    d : float or u.Quantity, optional
        The distance to this cluster (assumed in kpc), which is used to convert
        between on-sky angular and linear units in all plotting. Required
        explicitly as it cannot be read from a model. By default will attempt
        to use the stored initial value of "d".

    See Also
    --------
    _ClusterVisualizer : Base class providing all common plotting functions.
    '''

    @_ClusterVisualizer._support_units
    def _init_massfunc(self, observations):
        '''Initialize mass function quantities, as dict of PIs.'''

        self.mass_func = {}

        cen = (observations.mdata['RA'], observations.mdata['DEC'])

        PI_list = observations.filter_datasets('*mass_function*')

        if not PI_list:
            self.mass_func = None
            return

        for i, (key, mf) in enumerate(PI_list.items()):

            self.mass_func[key] = []

            clr = self.cmap(i / len(PI_list))

            field = mass.Field.from_dataset(mf, cen=cen)

            rbins = np.unique(np.c_[mf['r1'], mf['r2']], axis=0)
            rbins.sort(axis=0)

            for r_in, r_out in rbins:

                this_slc = {'r1': r_in, 'r2': r_out}

                field_slice = field.slice_radially(r_in, r_out)

                this_slc['field'] = field_slice

                this_slc['colour'] = clr

                # fake it till ya make it
                this_slc['dNdm'] = np.array([[]])

                this_slc['mj'] = np.array([])

                self.mass_func[key].append(this_slc)

    def __init__(self, observations, d=None):
        self.obs = observations
        self.name = observations.cluster

        self.rh = observations.initials['rh'] << u.pc

        self.star_bin = None
        self.mj = [] << u.Msun

        self.d = (d or observations.initials['d']) << u.kpc
        self.s2 = 0.
        self.F = 1.

        self.pm_T = None
        self.pm_R = None
        self.pm_tot = None
        self.pm_ratio = None
        self.LOS = None
        self.numdens = None

        self.K_scale = None

        self._init_massfunc(observations)


# --------------------------------------------------------------------------
# Sampled Model visualizers
# --------------------------------------------------------------------------

class SampledVisualizer:
    '''Some visualizations of a sampled model.

    Provides a few simple plotting methods useful for visualizing a sampled
    (multimass) model.
    This basic class provides only a few rudimentary plotting functions designed
    to allow for a quick examination of a sampled model.

    Parameters
    ----------
    sampledmodel : gcfit.SampledModel
        Sampled model instance to be plotted.

    thin : int, optional
        Optional thinning parameter which will reduce the amount of stars
        plotted by `[::thin]` each time. Can be useful at plotting models with
        very high `N`.
    '''

    _setup_artist = _ClusterVisualizer._setup_artist

    def _get_stars(self, type_='all'):
        '''Get a mask of stars to be plotted, based on `thin` and a star type'''

        mask = np.full(self.model.N, False)

        if type_ == 'all':
            mask[:] = True
        else:
            mask[self.model.star_types == type_] = True

        mask[::self.thin] = False

        return mask

    def __init__(self, sampledmodel, thin=1):

        self.model = sampledmodel
        self.d = sampledmodel.d
        self.thin = thin

    def plot_positions(self, fig=None, ax=None, type_='all',
                       galactic=False, projection=None, **kwargs):
        '''Plot the positions of each sampled star.

        Creates a scatter plot of the positions of all sampled stars in the
        model (after the `self.thin` thinning).
        Optionally show these in the cartesian cluster-centred frame, or in
        the projected galactic coordinates (if the model has been projected).

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_ClusterVisualizer._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot the positions. Should be a
            part of the given `fig`.

        type_ : {"all", "MS", "WD", "BH"}, optional
            Optionally restrict to plotting only the stars of a given star
            type. Defaults to all types.

        galactic : bool, optional
            If True, will plot the positions in the galactic frame
            (e.g. lon, lat, dist). If the sampled model was not projected at
            initialization, this will raise an error.

        projection : {None, "3d", "polar"}
            Optionally plot the positions in a 3-dimensional or polar frame.
            Defaults to simply a 2D cartesian frame. "polar" cannot be combined
            with the `galactic` frame.

        **kwargs : dict
            All other arguments are passed to `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        subplot_kw = dict(projection=projection)

        fig, ax = self._setup_artist(fig=fig, ax=ax, subplot_kw=subplot_kw)

        mask = self._get_stars(type_)

        if galactic:

            x = self.model.galactic.lon[mask]
            y = self.model.galactic.lat[mask]
            z = self.model.galactic.distance[mask]

        elif projection == 'polar':

            x = self.model.pos.r[mask]
            y = self.model.pos.theta[mask]
            z = None  # can't be polar and 3d

        else:

            x = self.model.pos.x[mask]
            y = self.model.pos.y[mask]
            z = self.model.pos.y[mask]

        if projection == '3d':
            ax.scatter(x, y, z, **kwargs)
        else:
            ax.scatter(x, y, **kwargs)

        return fig

    def plot_velocities(self, fig=None, ax=None, type_='all',
                        galactic=False, projection=None, **kwargs):
        '''Plot the velocities of each sampled star.

        Creates a scatter plot of the value of the velocities of all
        sampled stars in the model (after the `self.thin` thinning).
        Optionally show these in the cartesian cluster-centred frame, or in
        the projected galactic coordinates (if the model has been projected).

        Parameters
        ----------
        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_ClusterVisualizer._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to plot the velocities. Should be a
            part of the given `fig`.

        type_ : {"all", "MS", "WD", "BH"}, optional
            Optionally restrict to plotting only the stars of a given star
            type. Defaults to all types.

        galactic : bool, optional
            If True, will plot the velocities in the galactic frame
            (e.g. pm_l_cosb, pm_b, v_los). If the sampled model was not
            projected at initialization, this will raise an error.

        projection : {None, "3d", "polar"}
            Optionally plot the positions in a 3-dimensional or polar frame.
            Defaults to simply a 2D cartesian frame. "polar" cannot be combined
            with the `galactic` frame.

        **kwargs : dict
            All other arguments are passed to `ax.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        subplot_kw = dict(projection=projection)

        fig, ax = self._setup_artist(fig=fig, ax=ax, subplot_kw=subplot_kw)

        mask = self._get_stars(type_)

        if galactic:

            vx = self.model.galactic.pm_l_cosb[mask]
            vy = self.model.galactic.pm_b[mask]
            vz = self.model.galactic.v_los[mask]

        elif projection == 'polar':

            vx = self.model.vel.r[mask]
            vy = self.model.vel.theta[mask]
            vz = None  # can't be polar and 3d

        else:

            vx = self.model.vel.x[mask]
            vy = self.model.vel.y[mask]
            vz = self.model.vel.y[mask]

        if projection == '3d':
            ax.scatter(vx, vy, vz, **kwargs)
        else:
            ax.scatter(vx, vy, **kwargs)

        return fig

    @_ClusterVisualizer._support_units
    def plot_simulation(self, phot_system, r_band, g_band, b_band,
                        pixel_scale, FWHM,
                        fig=None, ax=None, *, source_kw=None,
                        ideal_imager=False, exptime=3600,
                        imager_kw=None, psf_kw=None,
                        observe_kw=None, rgb_kw=None, show_kw=None):
        '''Plot a simulated (RGB) photometric observation of the model.

        This method provides a barebones skeleton of the necessary steps
        required to create and plot a simulated observation (i.e. RGB image) of
        the sampled cluster, using the `artpop` package.
        This method relies on the `artpop.Source` object created by
        the `SampledModel.to_artpop` method.

        The source is passed through a PSF, artificial imager, and is observed
        in the three given "RGB" bands, each of which is combined using the
        astropy `make_lupton_rgb` function.

        This is a very simple method which follows the bare requirements set
        out by the `artpop` tutorial. If finer, more manual control is required,
        the `artpop.Source` object from `SampledModel.to_artpop` should be
        used directly in creating your own artpop-based imagery.

        `artpop` may require the isochrones for a given photometric system be
        already present, and may automatically start downloading them.

        Parameters
        ----------
        phot_system : str
            Name of the photometric system to simulate stellar magnitudes
            within. Must be supported by `artpop`.

        r_band, g_band, b_band : str
            The names of the red, green and blue filters to use. Will be
            passed to `artpop.ArtImager.observe`, and must be valid for the
            given `phot_system.

        pixel_scale : float or u.Quantity
            The pixel scale of the mock image. If a float is given,
            the units will be assumed to be `arcsec / pixels`.

        FWHM : float or u.Quantity
            Full width at half maximum of the psf. If a float is given,
            the units will be assumed to be arcsec. The units can be
            angular or in pixels. Passed to `artpop.moffat_psf`.

        fig : None or matplotlib.figure.Figure, optional
            Figure to place the ax on. If None (default), a new figure will
            be created, otherwise the given figure should be empty, or already
            have the correct number of axes.
            See `_ClusterVisualizer._setup_artist` for more details.

        ax : None or matplotlib.axes.Axes, optional
            An axes instance on which to place the simulated image. Should be a
            part of the given `fig`.

        source_kw : dict, optional
            Optional arguments passed to `SampledModel.to_artpop`.

        ideal_imager : bool, optional
            Whether to use the imager class `artpop.IdealImager` or
            `artpop.ArtImager` (default).

        exptime : float, optional
            Exposure time. If float is given, the units are assumed to be
            seconds. Passed to `artpop.ArtImager.observe`.

        imager_kw, psf_kw, observe_kw, rgb_kw, show_kw : dict, optional
            Optional arguments passed to `artpop.ArtImager`,
            `artpop.moffat_psf`, `artpop.ArtImager.observe`, `make_lupton_rgb`,
            and `artpop.show_image`, respectively.

        Returns
        -------
        matplotlib.figure.Figure
            The corresponding figure, containing all axes and plot artists.
        '''

        import artpop
        from astropy.visualization import make_lupton_rgb

        fig, ax = self._setup_artist(fig=fig, ax=ax)

        # ------------------------------------------------------------------
        # Get artpop Source object from sampled model directly
        # ------------------------------------------------------------------

        if source_kw is None:
            source_kw = {}

        src = self.model.to_artpop(phot_system, pixel_scale, **source_kw)

        # ------------------------------------------------------------------
        # Construct PSF
        # ------------------------------------------------------------------

        if psf_kw is None:
            psf_kw = {}

        psf = artpop.moffat_psf(fwhm=FWHM, **psf_kw)

        # ------------------------------------------------------------------
        # Construct imager
        # ------------------------------------------------------------------

        if imager_kw is None:
            imager_kw = {}

        if ideal_imager:
            imager = artpop.IdealImager(**imager_kw)

        else:
            imager = artpop.ArtImager(phot_system=phot_system, **imager_kw)

        # ------------------------------------------------------------------
        # Observe each colour
        # ------------------------------------------------------------------

        if observe_kw is None:
            observe_kw = {}

        if not ideal_imager:
            observe_kw.setdefault('exptime', exptime)

        obs_r = imager.observe(src, r_band, psf=psf, **observe_kw)
        obs_g = imager.observe(src, g_band, psf=psf, **observe_kw)
        obs_b = imager.observe(src, b_band, psf=psf, **observe_kw)

        # ------------------------------------------------------------------
        # Construct RGB image
        # ------------------------------------------------------------------

        if rgb_kw is None:
            rgb_kw = {}

        rgb = make_lupton_rgb(obs_r.image, obs_g.image, obs_b.image, **rgb_kw)

        # ------------------------------------------------------------------
        # Plot image
        # ------------------------------------------------------------------

        if show_kw is None:
            show_kw = {}

        artpop.show_image(rgb, subplots=(fig, ax), **show_kw)

        return fig


# --------------------------------------------------------------------------
# Collection of models
# --------------------------------------------------------------------------


class ModelCollection:
    '''Analysis and visualization of a collection of multiple models.

    Provides easy and uniform access to the analysis and visualization of a
    collection of different model visualizer instances, representing either
    different clusters, or different representations of the same cluster.

    This class is intimately tied to the `analysis.RunCollection` class,
    and provides it's access to the models corresponding to each of it's runs.
    The most likely way to initialize this class is therefore through the
    `RunCollection.get_models`/`RunCollection.get_CImodels` method.

    Parameters
    ----------
    visualizers : list of ModelVisualizer or CIModelVisualizer
        List of model visualizer objects which will make up this collection.
        Note that they must all be of the same type (`ModelVisualizer` or
        `CIModelVisualizer`), and the other classes provided here
        (e.g. `ObservationsVisualizer`) will not work.
    '''

    def __str__(self):
        return f"Collection of Models"

    def __iter__(self):
        '''Return an iterator over the individual model vizualizers.'''
        return iter(self.visualizers)

    def __getattr__(self, key):
        '''When accessing an attribute, fall back to get it from each model.'''
        return [getattr(mv, key) for mv in self.visualizers]

    def __init__(self, visualizers):
        self.visualizers = visualizers

        if all(isinstance(mv, ModelVisualizer) for mv in visualizers):
            self._ci = False
        elif all(isinstance(mv, CIModelVisualizer) for mv in visualizers):
            self._ci = True
        else:
            mssg = ('Invalid modelviz type. All visualizers must be either '
                    'ModelVisualizer or CIModelVisualizer')
            raise TypeError(mssg)

    @classmethod
    def load(cls, filenames):
        '''Load the model outputs previously saved in the a list of files.'''
        return cls([CIModelVisualizer.load(fn) for fn in filenames])

    def save(self, filenames, overwrite=False):
        '''Save the model outputs of all visualizers in the a list of files.'''
        for fn, mv in zip(filenames, self.visualizers):
            mv.save(fn, overwrite=overwrite)

    @classmethod
    def from_models(cls, models, obs_list=None):
        '''Initialize from a collection of models.

        Initializes a `ModelCollection` instance of single `ModelVisualizer`s
        based on a list of already computed models. A `ModelVisualizer`
        instance will be created for every given model.

        Parameters
        ----------
        models : list of gcfit.Model
            Collection of models to create a collection of visualizers for.

        obs_list : list of gcfit.Observations, optional
            List of observations corresponding to each model. Must be the same
            size as `models`. If None (default), all observations will attempt
            to be read from each model (as is default in `ModelVisualizer`).

        Returns
        -------
        ModelCollection
            The created model collection object.
        '''

        if obs_list is None:
            obs_list = [None, ] * len(models)

        return cls([ModelVisualizer(m, o) for m, o in zip(models, obs_list)])

    @classmethod
    def from_chains(cls, chains, obs_list, ci=True, *args, **kwargs):
        '''Initialize from a collection of parameter chains.

        Initializes a `ModelCollection` instance of either `ModelVisualizer` or
        `CIModelVisualizer`s based on a list of chains.

        If each chain is an array of shape (Nsamples, Nparams), each visualizer
        will be initialized using `from_chain`, otherwise if size (Nparams),
        will be initialized using `from_theta`.

        Parameters
        ----------
        chains : list of np.ndarray[Nsamples, Nparams] or np.ndarray[Nparams]
            List of parameter chains for each model, used to create each model.
            Either a single theta value or a full chain of values for each.
            Final axis of the chains must be of the size of the number of
            model parameters (13).

        obs_list : list of gcfit.Observations or None
            List of observations corresponding to each model.

        ci : bool, optional
            If True (default) creates collection of `CIModelVisualizer`,
            otherwise `ModelVisualizer`.

        *args, **kwargs
            Optional arguments passed to each individual visualizer init.

        Returns
        -------
        ModelCollection
            The created model collection object.
        '''

        viz = CIModelVisualizer if ci else ModelVisualizer

        if obs_list is None:
            obs_list = [None, ] * chains.shape[0]

        visualizers = []
        for ch, obs in zip(chains, obs_list):

            logging.info(f'Initializing {obs.cluster} for ModelCollection')

            init = viz.from_chain if ch.ndim == 2 else viz.from_theta

            visualizers.append(init(ch[...], obs, *args, **kwargs))

        return cls(visualizers)

    # ----------------------------------------------------------------------
    # Iterative plots
    # ----------------------------------------------------------------------

    def iter_plots(self, plot_func, yield_model=False, *args, **kwargs):
        '''Iterator yielding a call to `plot_func` for each visualizer.'''
        for mv in self.visualizers:
            fig = getattr(mv, plot_func)(*args, **kwargs)

            yield (fig, mv) if yield_model else fig

    def save_plots(self, plot_func, fn_pattern=None, save_kw=None, size=None,
                   *args, **kwargs):
        '''Iterate over calls to `plot_func` on each model and save the figures.

        Iterates over the `iter_plots` function and saves all individual
        figures to separate files, under a custom iterative file naming schema.

        Parameters
        ----------
        plot_func : str
            The name of the plotting function called on each run.

        fn_pattern : str, optional
            A format string, which is passed the argument "cluster"
            representing each model's name, and is used to create the filename
            each figure is saved under.
            Defaults to `f'./{cluster}_{plot_func[5:]}'`.

        save_kw : dict, optional
            Optional arguments are passed to the `fig.savefig` function.

        size : 2-tuple of float, optional
            Optional resizing of the figure, using `fig.set_size_inches`.

        **kwargs : dict
            All other arguments are passed to `iter_plots`.
        '''

        if fn_pattern is None:
            fn_pattern = f'./{{cluster}}_{plot_func[5:]}'

        if save_kw is None:
            save_kw = {}

        for fig, mv in self.iter_plots(plot_func, True, *args, **kwargs):

            if size is not None:
                fig.set_size_inches(size)

            save_kw['fname'] = fn_pattern.format(cluster=mv.name)

            logging.info(f'Saving {mv} to {save_kw["fname"]}')

            fig.savefig(**save_kw)

            plt.close(fig)
