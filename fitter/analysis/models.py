from .. import util
from ..probabilities import mass
from ..core.data import Observations, Model

import h5py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr
import matplotlib.ticker as mpl_tick
import astropy.visualization as astroviz

import logging


__all__ = ['ModelVisualizer', 'CIModelVisualizer', 'ObservationsVisualizer']


def _get_model(theta, observations):
    try:
        return Model(theta, observations=observations)
    except ValueError:
        logging.warning(f"Model did not converge with {theta=}")
        return None

# --------------------------------------------------------------------------
# Individual model visualizers
# --------------------------------------------------------------------------


class _ClusterVisualizer:

    _MARKERS = ('o', '^', 'D', '+', 'x', '*', 's', 'p', 'h', 'v', '1', '2')

    # Default xaxis limits for all profiles. Set by inits, can be reset by user
    rlims = None

    # -----------------------------------------------------------------------
    # Artist setups
    # -----------------------------------------------------------------------

    def _setup_artist(self, fig, ax, *, use_name=True):
        '''setup a plot (figure and ax) with one single ax'''

        if ax is None:
            if fig is None:
                # no figure or ax provided, make one here
                fig, ax = plt.subplots()

            else:
                # Figure provided, no ax provided. Try to grab it from the fig
                # if that doens't work, create it
                cur_axes = fig.axes

                if len(cur_axes) > 1:
                    raise ValueError(f"figure {fig} already has too many axes")

                elif len(cur_axes) == 1:
                    ax = cur_axes[0]

                else:
                    ax = fig.add_subplot()

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
        '''setup a subplot with multiple axes'''

        if subfig_kw is None:
            subfig_kw = {}

        def create_axes(base, shape):
            '''create the axes of `shape` on this base (fig)'''

            # make sure shape is a tuple of atleast 1d, at most 2d

            if not isinstance(shape, tuple):
                # TODO doesnt work on an int
                shape = tuple(shape)

            if len(shape) == 1:
                shape = (shape, 1)

            elif len(shape) > 2:
                mssg = f"Invalid `shape` for subplots {shape}, must be 2D"
                raise ValueError(mssg)

            # split into dict of nrows, ncols

            shape = dict(zip(("nrows", "ncols"), shape))

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

            # this fig has axes, check that they match shape
            if axarr := fig.axes:
                # TODO this won't actually work, cause fig.axes is just a list
                if axarr.shape != shape:
                    mssg = (f"figure {fig} already contains axes with "
                            f"mismatched shape ({axarr.shape} != {shape})")
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

    # -----------------------------------------------------------------------
    # Unit support
    # -----------------------------------------------------------------------

    def _support_units(method):
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
        '''from an array of data percentiles, return the median array'''
        return percs[percs.shape[0] // 2] if percs.ndim > 1 else percs

    def _get_err(self, dataset, key):
        '''gather the error variables corresponding to `key` from `dataset`'''
        try:
            return dataset[f'Δ{key}']
        except KeyError:
            try:
                return np.c_[dataset[f'Δ{key},down'], dataset[f'Δ{key},up']].T
            except KeyError:
                return None

    def _plot_model(self, ax, data, intervals=None, *,
                    x_data=None, x_unit='pc', y_unit=None,
                    CI_kwargs=None, **kwargs):

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

        x_domain = self.r if x_data is None else x_data

        if x_unit:
            x_domain = x_domain.to(x_unit)

        if y_unit:
            data = data.to(y_unit)

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
                   err_transform=None, **kwargs):

        # TODO need to handle colours better
        defaultcolour = None

        # ------------------------------------------------------------------
        # Get data and relevant errors for plotting
        # ------------------------------------------------------------------

        xdata = dataset[x_key]
        ydata = dataset[y_key]

        xerr = self._get_err(dataset, x_key)
        yerr = self._get_err(dataset, y_key)

        # ------------------------------------------------------------------
        # Convert any units desired
        # ------------------------------------------------------------------

        if x_unit is not None:
            xdata = xdata.to(x_unit)

        if y_unit is not None:
            ydata = ydata.to(y_unit)

        # ------------------------------------------------------------------
        # If given, transform errors based on `err_transform` function
        # ------------------------------------------------------------------

        if err_transform is not None:
            yerr = err_transform(yerr)

        # ------------------------------------------------------------------
        # Setup default plotting details, style, labels
        # ------------------------------------------------------------------

        kwargs.setdefault('marker', '.')
        kwargs.setdefault('linestyle', 'None')
        kwargs.setdefault('color', defaultcolour)

        label = dataset.cite()
        if 'm' in dataset.mdata:
            label += fr' ($m={dataset.mdata["m"]}\ M_\odot$)'

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------

        # TODO not sure if I like the mfc=none style,
        #   mostly due to https://github.com/matplotlib/matplotlib/issues/3400
        return ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, mfc='none',
                           label=label, **kwargs)

    def _plot_profile(self, ax, ds_pattern, y_key, model_data, *,
                      y_unit=None, residuals=False, err_transform=None,
                      res_kwargs=None, **kwargs):
        '''figure out what needs to be plotted and call model/data plotters
        all **kwargs passed to both _plot_model and _plot_data
        model_data dimensions *must* be (mass bins, intervals, r axis)
        '''

        # TODO we might still want to allow for specific model/data kwargs?

        ds_pattern = ds_pattern or ''

        strict = kwargs.pop('strict', False)

        # Restart marker styles each plotting call
        markers = iter(self._MARKERS)

        # TODO need to figure out how we handle passed kwargs better
        default_clr = kwargs.pop('color', None)

        if res_kwargs is None:
            res_kwargs = {}

        res_kwargs.setdefault('y_unit', y_unit)

        # ------------------------------------------------------------------
        # Determine the relevant datasets to the given pattern
        # ------------------------------------------------------------------

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

        for key, dset in datasets.items():

            mrk = next(markers)

            # get mass bin of this dataset, for later model plotting
            if 'm' in dset.mdata:
                m = dset.mdata['m'] * u.Msun
                mass_bin = np.where(self.mj == m)[0][0]
            else:
                mass_bin = self.star_bin

            if mass_bin in masses:
                clr = masses[mass_bin][0][0].get_color()
            else:
                clr = default_clr

            # plot the data
            try:
                line = self._plot_data(ax, dset, y_key, marker=mrk, color=clr,
                                       err_transform=err_transform,
                                       y_unit=y_unit, **kwargs)

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

        if model_data is not None:

            # ensure that the data is (mass bin, intervals, r domain)
            if len(model_data.shape) != 3:
                raise ValueError("invalid model data shape")

            # No data plotted, use the star_bin
            if not masses:
                if model_data.shape[0] > 1:
                    masses = {self.star_bin: None}
                else:
                    masses = {0: None}

            res_ax = None

            for mbin, errbars in masses.items():

                ymodel = model_data[mbin, :, :]

                # TODO having model/data be same color is kinda hard to read
                #   this is why I added mfc=none, but I dont like that either
                if errbars is not None:
                    clr = errbars[0][0].get_color()
                else:
                    clr = default_clr

                self._plot_model(ax, ymodel, color=clr, y_unit=y_unit, **kwargs)

                if residuals:
                    res_ax = self._add_residuals(ax, ymodel, errbars,
                                                 res_ax=res_ax, **res_kwargs)

        # Adjust x limits
        if self.rlims is not None:
            ax.set_xlim(*self.rlims)

        # label x axis
        if residuals:
            res_ax.set_xlabel(f'Distance from centre [{res_ax.get_xlabel()}]')
        else:
            ax.set_xlabel(f'Distance from centre [{ax.get_xlabel()}]')

    # -----------------------------------------------------------------------
    # Plot extras
    # -----------------------------------------------------------------------

    def _add_residuals(self, ax, ymodel, errorbars, percentage=False, *,
                       show_chi2=False, xmodel=None, y_unit=None, size="15%",
                       res_ax=None, divider_kwargs=None, **kwargs):
        '''
        errorbars : a list of outputs from calls to plt.errorbars
        '''
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if not errorbars:
            mssg = "Cannot compute residuals, no observables data provided"
            raise ValueError(mssg)

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

            clr = errbar[0].get_color()
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
                            color=clr, marker=mrk, linestyle='none')

            # --------------------------------------------------------------
            # Optionally compute chi-squared statistic
            # --------------------------------------------------------------

            if show_chi2:
                chi2 += np.sum((res / yerr)**2)

        if show_chi2:
            fake = plt.Line2D([], [], label=fr"$\chi^2={chi2:.2f}$")
            res_ax.legend(handles=[fake], handlelength=0, handletextpad=0)

        # ------------------------------------------------------------------
        # Label axes
        # ------------------------------------------------------------------

        if percentage:
            res_ax.set_ylabel(r'Residuals')
            res_ax.yaxis.set_major_formatter(mpl_tick.PercentFormatter())
        else:
            res_ax.set_ylabel(f'Residuals [{res_ax.get_ylabel()}]')

        return res_ax

    def _add_hyperparam(self, ax, ymodel, xdata, ydata, yerr):
        # TODO this is still a complete mess

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
                 x_unit='pc', y_unit='km/s', res_kwargs=None, **kwargs):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Line-of-Sight Velocity Dispersion')

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*velocity_dispersion*', 'σ'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        self._plot_profile(ax, pattern, var, self.LOS,
                           strict=strict, residuals=residuals,
                           x_unit=x_unit, y_unit=y_unit,
                           res_kwargs=res_kwargs, **kwargs)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_tot(self, fig=None, ax=None,
                    show_obs=True, residuals=False, *,
                    x_unit='pc', y_unit='mas/yr', res_kwargs=None, **kwargs):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Total Proper Motion")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_tot'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        self._plot_profile(ax, pattern, var, self.pm_tot,
                           strict=strict, residuals=residuals,
                           x_unit=x_unit, y_unit=y_unit,
                           res_kwargs=res_kwargs, **kwargs)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_ratio(self, fig=None, ax=None,
                      show_obs=True, residuals=False, *,
                      x_unit='pc', res_kwargs=None, **kwargs):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Proper Motion Anisotropy")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_ratio'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        self._plot_profile(ax, pattern, var, self.pm_ratio,
                           strict=strict, residuals=residuals,
                           x_unit=x_unit,
                           res_kwargs=res_kwargs, **kwargs)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_T(self, fig=None, ax=None,
                  show_obs=True, residuals=False, *,
                  x_unit='pc', y_unit='mas/yr', res_kwargs=None, **kwargs):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Tangential Proper Motion")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_T'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        # pm_T = self.pm_T.to('mas/yr')

        self._plot_profile(ax, pattern, var, self.pm_T,
                           strict=strict, residuals=residuals,
                           x_unit=x_unit, y_unit=y_unit,
                           res_kwargs=res_kwargs, **kwargs)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_R(self, fig=None, ax=None,
                  show_obs=True, residuals=False, *,
                  x_unit='pc', y_unit='mas/yr', res_kwargs=None, **kwargs):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Radial Proper Motion")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_R'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        # pm_R = self.pm_R.to('mas/yr')

        self._plot_profile(ax, pattern, var, self.pm_R,
                           strict=strict, residuals=residuals,
                           x_unit=x_unit, y_unit=y_unit,
                           res_kwargs=res_kwargs, **kwargs)

        ax.legend()

        return fig

    @_support_units
    def plot_number_density(self, fig=None, ax=None,
                            show_obs=True, residuals=False, *,
                            x_unit='pc', res_kwargs=None, **kwargs):

        def quad_nuisance(err):
            return np.sqrt(err**2 + (self.s2 << err.unit**2))

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Number Density')

        ax.loglog()

        if show_obs:
            pattern, var = '*number_density*', 'Σ'
            strict = show_obs == 'strict'
            kwargs.setdefault('err_transform', quad_nuisance)

        else:
            pattern = var = None
            strict = False

        self._plot_profile(ax, pattern, var, self.numdens,
                           strict=strict, residuals=residuals,
                           x_unit=x_unit,
                           res_kwargs=res_kwargs, **kwargs)

        # bit arbitrary, but probably fine for the most part
        ax.set_ylim(bottom=1e-4)

        ax.legend()

        return fig

    @_support_units
    def plot_all(self, fig=None, sharex=True, **kwargs):
        '''Plots all the primary profiles (numdens, LOS, PM)
        but *not* the mass function, pulsars, or any secondary profiles
        (cum-mass, remnants, etc)
        '''

        fig, axes = self._setup_multi_artist(fig, (3, 2), sharex=sharex)

        axes = axes.reshape((3, 2))

        res_kwargs = dict(size="25%", show_chi2=False, percentage=True)
        kwargs.setdefault('res_kwargs', res_kwargs)

        self.plot_number_density(fig=fig, ax=axes[0, 0], **kwargs)
        self.plot_LOS(fig=fig, ax=axes[1, 0], **kwargs)
        self.plot_pm_ratio(fig=fig, ax=axes[2, 0], **kwargs)

        self.plot_pm_tot(fig=fig, ax=axes[0, 1], **kwargs)
        self.plot_pm_T(fig=fig, ax=axes[1, 1], **kwargs)
        self.plot_pm_R(fig=fig, ax=axes[2, 1], **kwargs)

        for ax in axes.flatten():
            ax.set_xlabel('')

        return fig

    # ----------------------------------------------------------------------
    # Mass Function Plotting
    # ----------------------------------------------------------------------

    @_support_units
    def plot_mass_func(self, fig=None, show_obs=True, show_fields=True, *,
                       colours=None, PI_legend=False, logscaled=False,
                       field_kw=None):

        # ------------------------------------------------------------------
        # Setup axes, splitting into two columns if necessary and adding the
        # extra ax for the field plot if desired
        # ------------------------------------------------------------------

        N_rbins = sum([len(d) for d in self.mass_func.values()])
        shape = ((int(np.ceil(N_rbins / 2)), int(np.floor(N_rbins / 2))), 2)

        # If adding the fields, include an extra column on the left for it
        if show_fields:
            shape = ((1, *shape[0]), shape[1] + 1)

        fig, axes = self._setup_multi_artist(fig, shape, sharex=True)

        axes = axes.T.flatten()

        ax_ind = 0

        # ------------------------------------------------------------------
        # If desired, use the `plot_MF_fields` method to show the fields
        # ------------------------------------------------------------------

        if show_fields:

            ax = axes[ax_ind]

            if field_kw is None:
                field_kw = {}

            field_kw.setdefault('radii', [])

            self.plot_MF_fields(fig, ax, **field_kw)

            ax_ind += 1

        # ------------------------------------------------------------------
        # Iterate over each PI, gathering data to plot
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

                ax = axes[ax_ind]

                clr = rbin.get('colour', None)

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
                                       fmt='o', color=clr)

                    clr = pnts[0].get_color()

                # ----------------------------------------------------------
                # Plot model. Doesn't utilize the `_plot_profile` method, as
                # this is *not* a profile, but does use similar, but simpler,
                # logic
                # ----------------------------------------------------------

                # The mass domain is provided explicitly, to support visualizers
                # which don't store the entire mass range (e.g. CImodels)
                mj = rbin['mj']

                dNdm = rbin['dNdm']

                midpoint = dNdm.shape[0] // 2

                median = dNdm[midpoint]

                med_plot, = ax.plot(mj, median, '--', c=clr)

                alpha = 0.8 / (midpoint + 1)
                for sigma in range(1, midpoint + 1):

                    ax.fill_between(
                        mj,
                        dNdm[midpoint + sigma],
                        dNdm[midpoint - sigma],
                        alpha=1 - alpha, color=clr
                    )

                    alpha += alpha

                if logscaled:
                    ax.set_xscale('log')

                ax.set_xlabel(None)

                # ----------------------------------------------------------
                # "Label" each bin with it's radial bounds.
                # Uses fake text to allow for using loc='best' from `legend`.
                # Really this should be a part of plt (see matplotlib#17946)
                # ----------------------------------------------------------

                r1 = rbin['r1'].to_value('arcmin')
                r2 = rbin['r2'].to_value('arcmin')

                fake = plt.Line2D([], [], label=f"r = {r1:.2f}'-{r2:.2f}'")
                handles = [fake]

                leg_kw = {'handlelength': 0, 'handletextpad': 0}

                # If this is the first bin, also add a PI tag
                if PI_legend and not rind and not show_fields:
                    pi_fake = plt.Line2D([], [], label=PI)
                    handles.append(pi_fake)
                    leg_kw['labelcolor'] = ['k', clr]

                ax.legend(handles=handles, **leg_kw)

                ax_ind += 1

        # ------------------------------------------------------------------
        # Put labels on subfigs
        # ------------------------------------------------------------------

        for sf in fig.subfigs[show_fields:]:

            sf.supxlabel(r'Mass [$M_\odot$]')

        fig.subfigs[show_fields].supylabel('dN/dm')

        return fig

    @_support_units
    def plot_MF_fields(self, fig=None, ax=None, *, radii=("rh",),
                       cmap=None, grid=True):
        '''plot all mass function fields in this observation
        '''
        import shapely.geometry as geom

        fig, ax = self._setup_artist(fig, ax)

        # Centre dot
        ax.plot(0, 0, 'kx')

        # ------------------------------------------------------------------
        # Iterate over each PI and it's radial bins
        # ------------------------------------------------------------------

        for PI, bins in self.mass_func.items():

            for rbin in bins:

                # ----------------------------------------------------------
                # Plot the field using this `Field` slice's own plotting method
                # ----------------------------------------------------------

                clr = rbin.get("colour", None)

                rbin['field'].plot(ax, fc=clr, alpha=0.7, ec='k', label=PI)

                # make this label private so it's only added once to legend
                PI = f'_{PI}'

        # ------------------------------------------------------------------
        # If desired, add a "pseudo" grid in the polar projection, at 2
        # arcmin intervals, up to the rt
        # ------------------------------------------------------------------

        # Ensure the gridlines don't affect the axes scaling
        ax.autoscale(False)

        if grid:
            # TODO this should probably use distance to furthest field
            rt = self.rt if hasattr(self, 'rt') else (20 << u.arcmin)
            ticks = np.arange(2, rt.to_value('arcmin'), 2)

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

                ax.annotate(f'{gr:.0f}"', xy=(circle[0].max(), 0),
                            color=grid_kw['color'])

        # ------------------------------------------------------------------
        # Try to plot the various radii quantities from this model, if desired
        # ------------------------------------------------------------------

        # TODO for CI this could be a CI of rh, ra, rt actually (60)

        for r_type in radii:

            # This is to explicitly avoid very ugly exceptions from geom
            if r_type not in {'rh', 'ra', 'rt'}:
                mssg = f'radii must be one of {{rh, ra, rt}}, not `{r_type}`'
                raise TypeError(mssg)

            radius = getattr(self, r_type).to_value('arcmin')
            circle = np.array(geom.Point(0, 0).buffer(radius).exterior.coords).T
            ax.plot(*circle, ls='--')
            ax.text(0, circle[1].max(), r_type)

        # ------------------------------------------------------------------
        # Add plot labels and legends
        # ------------------------------------------------------------------

        ax.set_xlabel('RA [arcmin]')
        ax.set_ylabel('DEC [arcmin]')

        # TODO figure out a better way of handling this always using best? (75)
        ax.legend(loc='upper left' if grid else 'best')

        return fig

    # -----------------------------------------------------------------------
    # Model plotting
    # -----------------------------------------------------------------------

    @_support_units
    def plot_density(self, fig=None, ax=None, kind='all', *,
                     x_unit='pc'):

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Surface Mass Density')

        # Total density
        if 'tot' in kind:
            kw = {"label": "Total", "color": "tab:cyan"}
            self._plot_profile(ax, None, None, self.rho_tot,
                               x_unit=x_unit, **kw)

        # Total Remnant density
        if 'rem' in kind:
            kw = {"label": "Remnants", "color": "tab:purple"}
            self._plot_profile(ax, None, None, self.rho_rem,
                               x_unit=x_unit, **kw)

        # Main sequence density
        if 'MS' in kind:
            kw = {"label": "Main-sequence stars", "color": "tab:orange"}
            self._plot_profile(ax, None, None, self.rho_MS,
                               x_unit=x_unit, **kw)

        if 'WD' in kind:
            kw = {"label": "White Dwarfs", "color": "tab:green"}
            self._plot_profile(ax, None, None, self.rho_WD,
                               x_unit=x_unit, **kw)

        if 'NS' in kind:
            kw = {"label": "Neutron Stars", "color": "tab:red"}
            self._plot_profile(ax, None, None, self.rho_NS,
                               x_unit=x_unit, **kw)

        # Black hole density
        if 'BH' in kind:
            kw = {"label": "Black Holes", "color": "tab:gray"}
            self._plot_profile(ax, None, None, self.rho_BH,
                               x_unit=x_unit, **kw)

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_ylabel(rf'Surface Density $[M_\odot / pc^3]$')
        # ax.set_xlabel('arcsec')

        # ax.legend()
        fig.legend(loc='upper center', ncol=6,
                   bbox_to_anchor=(0.5, 1.), fancybox=True)

        return fig

    @_support_units
    def plot_surface_density(self, fig=None, ax=None, kind='all', *,
                             x_unit='pc'):

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Surface Mass Density')

        # Total density
        if 'tot' in kind:
            kw = {"label": "Total", "color": "tab:cyan"}
            self._plot_profile(ax, None, None, self.Sigma_tot,
                               x_unit=x_unit, **kw)

        # Total Remnant density
        if 'rem' in kind:
            kw = {"label": "Remnants", "color": "tab:purple"}
            self._plot_profile(ax, None, None, self.Sigma_rem,
                               x_unit=x_unit, **kw)

        # Main sequence density
        if 'MS' in kind:
            kw = {"label": "Main-sequence stars", "color": "tab:orange"}
            self._plot_profile(ax, None, None, self.Sigma_MS,
                               x_unit=x_unit, **kw)

        if 'WD' in kind:
            kw = {"label": "White Dwarfs", "color": "tab:green"}
            self._plot_profile(ax, None, None, self.Sigma_WD,
                               x_unit=x_unit, **kw)

        if 'NS' in kind:
            kw = {"label": "Neutron Stars", "color": "tab:red"}
            self._plot_profile(ax, None, None, self.Sigma_NS,
                               x_unit=x_unit, **kw)

        # Black hole density
        if 'BH' in kind:
            kw = {"label": "Black Holes", "color": "tab:gray"}
            self._plot_profile(ax, None, None, self.Sigma_BH,
                               x_unit=x_unit, **kw)

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_ylabel(rf'Surface Density $[M_\odot / pc^2]$')
        # ax.set_xlabel('arcsec')

        # ax.legend()
        fig.legend(loc='upper center', ncol=6,
                   bbox_to_anchor=(0.5, 1.), fancybox=True)

        return fig

    @_support_units
    def plot_cumulative_mass(self, fig=None, ax=None, kind='all', *,
                             x_unit='pc'):

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Cumulative Mass')

        # Total density
        if 'tot' in kind:
            kw = {"label": "Total", "color": "tab:cyan"}
            self._plot_profile(ax, None, None, self.cum_M_tot,
                               x_unit=x_unit, **kw)

        # Main sequence density
        if 'MS' in kind:
            kw = {"label": "Main-sequence stars", "color": "tab:orange"}
            self._plot_profile(ax, None, None, self.cum_M_MS,
                               x_unit=x_unit, **kw)

        if 'WD' in kind:
            kw = {"label": "White Dwarfs", "color": "tab:green"}
            self._plot_profile(ax, None, None, self.cum_M_WD,
                               x_unit=x_unit, **kw)

        if 'NS' in kind:
            kw = {"label": "Neutron Stars", "color": "tab:red"}
            self._plot_profile(ax, None, None, self.cum_M_NS,
                               x_unit=x_unit, **kw)

        # Black hole density
        if 'BH' in kind:
            kw = {"label": "Black Holes", "color": "tab:gray"}
            self._plot_profile(ax, None, None, self.cum_M_BH,
                               x_unit=x_unit, **kw)

        ax.set_yscale("log")
        ax.set_xscale("log")

        # ax.set_ylabel(rf'$M_{{enc}} ({self.cum_M_tot.unit})$')
        ax.set_ylabel(rf'$M_{{enc}}$ $[M_\odot]$')
        # ax.set_xlabel('arcsec')

        # ax.legend()
        fig.legend(loc='upper center', ncol=5,
                   bbox_to_anchor=(0.5, 1.), fancybox=True)

        return fig

    @_support_units
    def plot_remnant_fraction(self, fig=None, ax=None, *, x_unit='pc'):
        '''Fraction of mass in remnants vs MS stars, like in baumgardt'''

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Remnant Fraction")

        ax.set_xscale("log")

        self._plot_profile(ax, None, None, self.frac_M_MS,
                           x_unit=x_unit, label="Main-sequence stars")
        self._plot_profile(ax, None, None, self.frac_M_rem,
                           x_unit=x_unit, label="Remnants")

        ax.set_ylabel(r"Mass fraction $M_{MS}/M_{tot}$, $M_{remn.}/M_{tot}$")

        ax.set_ylim(0.0, 1.0)

        ax.legend()

        return fig

    # -----------------------------------------------------------------------
    # Goodness of fit statistics
    # -----------------------------------------------------------------------

    @_support_units
    def _compute_profile_chi2(self, ds_pattern, y_key, model_data, *,
                              x_key='r', err_transform=None, reduced=True):
        '''compute chi2 for this dataset (pattern)'''

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
        '''compute chi2 between median model and all datasets
        Be cognizant that this is only the median model chi2, and not
        necessarily useful for actual statistics
        '''
        # TODO seems to produce alot of infs?

        def numdens_nuisance(err):
            return np.sqrt(err**2 + (self.s2 << err.unit**2))

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
    '''
    class for making, showing, saving all the plots related to a single model
    '''

    @classmethod
    def from_chain(cls, chain, observations, method='median'):
        '''
        create a Visualizer instance based on a chain, y taking the median
        of the chain parameters
        '''
        reduc_methods = {'median': np.median, 'mean': np.mean}

        # if 3d (Niters, Nwalkers, Nparams)
        # if 2d (Nwalkers, Nparams)
        # if 1d (Nparams)
        chain = chain.reshape((-1, chain.shape[-1]))

        theta = reduc_methods[method](chain, axis=0)

        return cls(Model(theta, observations), observations)

    @classmethod
    def from_theta(cls, theta, observations):
        '''
        create a Visualizer instance based on a theta, see `Model` for allowed
        theta types
        '''
        return cls(Model(theta, observations), observations)

    def __init__(self, model, observations=None):
        self.model = model
        self.obs = observations if observations else model.observations
        self.name = observations.cluster

        self.rh = model.rh
        self.ra = model.ra
        self.rt = model.rt
        self.F = model.F
        self.s2 = model.s2
        self.d = model.d

        self.r = model.r

        self.rlims = (9e-3, self.r.max().value + 5) << self.r.unit

        self._2πr = 2 * np.pi * model.r

        self.star_bin = model.nms - 1
        self.mj = model.mj

        self.LOS = np.sqrt(self.model.v2pj)[:, np.newaxis, :]
        self.pm_T = np.sqrt(model.v2Tj)[:, np.newaxis, :]
        self.pm_R = np.sqrt(model.v2Rj)[:, np.newaxis, :]

        self.pm_tot = np.sqrt(0.5 * (self.pm_T**2 + self.pm_R**2))
        self.pm_ratio = self.pm_T / self.pm_R

        self._init_numdens(model, observations)
        self._init_massfunc(model, observations)

        self._init_surfdens(model, observations)
        self._init_dens(model, observations)

        self._init_mass_frac(model, observations)
        self._init_cum_mass(model, observations)

    # TODO alot of these init functions could be more homogenous
    @_ClusterVisualizer._support_units
    def _init_numdens(self, model, observations):

        model_nd = model.Sigmaj / model.mj[:, np.newaxis]

        nd = np.empty(model_nd.shape)[:, np.newaxis, :] << model_nd.unit

        # Check for observational numdens profiles, to compute scaling factor K
        if obs_nd := observations.filter_datasets('*number_density'):

            if len(obs_nd) > 1:
                mssg = ('Too many number density datasets, '
                        'computing scaling factor using only final dataset')
                logging.warning(mssg)

            obs_nd = list(obs_nd.values())[-1]
            obs_r = obs_nd['r'].to(model.r.unit)

            for mbin in range(model_nd.shape[0]):

                nd_interp = util.QuantitySpline(model.r, model_nd[mbin, :])

                K = (np.nansum(obs_nd['Σ'] * nd_interp(obs_r) / obs_nd['Σ']**2)
                     / np.nansum(nd_interp(obs_r)**2 / obs_nd['Σ']**2))

                nd[mbin, 0, :] = K * model_nd[mbin, :]

        else:
            mssg = 'No number density datasets found, setting K=1'
            logging.info(mssg)

            nd[:, 0, :] = model_nd[:, :]

        self.numdens = nd

    @_ClusterVisualizer._support_units
    def _init_massfunc(self, model, observations, *, cmap=None):
        '''
        sets self.mass_func as a dict of PI's, where each PI has a list of
        subdicts. Each subdict represents a single radial slice (within this PI)
        and contains the radii, the mass func values, and the field slice
        '''

        cmap = cmap or plt.cm.rainbow

        self.mass_func = {}

        cen = (observations.mdata['RA'], observations.mdata['DEC'])

        PI_list = observations.filter_datasets('*mass_function*')

        densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                    for j in range(model.nms)]

        for i, (key, mf) in enumerate(PI_list.items()):

            self.mass_func[key] = []

            clr = cmap(i / len(PI_list))

            field = mass.Field.from_dataset(mf, cen=cen)

            rbins = np.unique(np.c_[mf['r1'], mf['r2']], axis=0)
            rbins.sort(axis=0)

            for r_in, r_out in rbins:

                this_slc = {'r1': r_in, 'r2': r_out}

                field_slice = field.slice_radially(r_in, r_out)

                this_slc['field'] = field_slice

                this_slc['colour'] = clr

                this_slc['dNdm'] = np.empty((1, model.nms))

                this_slc['mj'] = model.mj[:model.nms]

                sample_radii = field_slice.MC_sample(300).to(u.pc)

                for j in range(model.nms):

                    Nj = field_slice.MC_integrate(densityj[j], sample_radii)
                    widthj = (model.mj[j] * model.mes_widths[j])

                    this_slc['dNdm'][0, j] = (Nj / widthj).value

                self.mass_func[key].append(this_slc)

    @_ClusterVisualizer._support_units
    def _init_dens(self, model, observations):

        shp = (np.newaxis, np.newaxis, slice(None))

        self.rho_tot = np.sum(model.rhoj, axis=0)[shp]
        self.rho_MS = np.sum(model.rhoj[model._star_bins], axis=0)[shp]
        self.rho_rem = np.sum(model.rhoj[model._remnant_bins], axis=0)[shp]
        self.rho_BH = np.sum(model.BH_rhoj, axis=0)[shp]
        self.rho_WD = np.sum(model.WD_rhoj, axis=0)[shp]
        self.rho_NS = np.sum(model.NS_rhoj, axis=0)[shp]

    @_ClusterVisualizer._support_units
    def _init_surfdens(self, model, observations):

        shp = (np.newaxis, np.newaxis, slice(None))

        self.Sigma_tot = np.sum(model.Sigmaj, axis=0)[shp]
        self.Sigma_MS = np.sum(model.Sigmaj[model._star_bins], axis=0)[shp]
        self.Sigma_rem = np.sum(model.Sigmaj[model._remnant_bins], axis=0)[shp]
        self.Sigma_BH = np.sum(model.BH_Sigmaj, axis=0)[shp]
        self.Sigma_WD = np.sum(model.WD_Sigmaj, axis=0)[shp]
        self.Sigma_NS = np.sum(model.NS_Sigmaj, axis=0)[shp]

    @_ClusterVisualizer._support_units
    def _init_mass_frac(self, model, observations):

        int_MS = util.QuantitySpline(self.r, self._2πr * self.Sigma_MS)
        int_rem = util.QuantitySpline(self.r, self._2πr * self.Sigma_rem)
        int_tot = util.QuantitySpline(self.r, self._2πr * self.Sigma_tot)

        mass_MS = np.empty((1, 1, self.r.size))
        mass_rem = np.empty((1, 1, self.r.size))
        mass_tot = np.empty((1, 1, self.r.size))

        # TODO the rbins at the end always mess up fractional stuff, drop to 0
        mass_MS[0, 0, 0] = mass_rem[0, 0, 0] = mass_tot[0, 0, 0] = np.nan

        for i in range(1, self.r.size - 2):
            mass_MS[0, 0, i] = int_MS.integral(self.r[i], self.r[i + 1]).value
            mass_rem[0, 0, i] = int_rem.integral(self.r[i], self.r[i + 1]).value
            mass_tot[0, 0, i] = int_tot.integral(self.r[i], self.r[i + 1]).value

        self.frac_M_MS = mass_MS / mass_tot
        self.frac_M_rem = mass_rem / mass_tot

    @_ClusterVisualizer._support_units
    def _init_cum_mass(self, model, observations):

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
    '''
    class for making, showing, saving all the plots related to a bunch of models
    in the form of confidence intervals
    '''

    @_ClusterVisualizer._support_units
    def plot_BH_mass(self, fig=None, ax=None, bins='auto', color='b'):

        fig, ax = self._setup_artist(fig, ax)

        color = mpl_clr.to_rgb(color)
        facecolor = color + (0.33, )

        ax.hist(self.BH_mass, histtype='stepfilled',
                bins=bins, ec=color, fc=facecolor, lw=2)

        return fig

    @_ClusterVisualizer._support_units
    def plot_BH_num(self, fig=None, ax=None, bins='auto', color='b'):

        fig, ax = self._setup_artist(fig, ax)

        color = mpl_clr.to_rgb(color)
        facecolor = color + (0.33, )

        ax.hist(self.BH_num, histtype='stepfilled',
                bins=bins, ec=color, fc=facecolor, lw=2)

        return fig

    def __init__(self, observations):
        self.obs = observations
        self.name = observations.cluster

    @classmethod
    def from_chain(cls, chain, observations, N=100, *,
                   verbose=False, pool=None):
        import functools

        viz = cls(observations)

        viz.N = N

        # ------------------------------------------------------------------
        # Get info about the chain and set of models
        # ------------------------------------------------------------------

        # Flatten walkers, if not already
        chain = chain.reshape((-1, chain.shape[-1]))[-N:]

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

        # TODO https://github.com/nmdickson/GCfit/issues/100
        huge_model = Model(chain[np.argmax(chain[:, 4])], viz.obs)

        viz.rt = huge_model.rt
        viz.r = np.r_[0, np.geomspace(1e-5, viz.rt.value, num=99)] << u.pc

        viz.rlims = (9e-3, viz.r.max().value + 5) << viz.r.unit

        # Assume that this example model has same nms bin as all models
        # This approximation isn't exactly correct (especially when Ndot != 0),
        # but close enough for plots
        viz.star_bin = 0

        # mj only contains nms and tracer bins (the only ones we plot anyways)
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

        # number density

        numdens = np.full((1, N, Nr), np.nan) << u.pc**-2

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

            # Black holes

            BH_mass[model_ind] = np.sum(model.BH_Mj)
            BH_num[model_ind] = np.sum(model.BH_Nj)

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

        viz.mass_func = massfunc

        for rbins in viz.mass_func.values():
            for rslice in rbins:

                rslice['dNdm'] = perc(rslice['dNdm'], q, axis=0)

        viz.frac_M_MS = perc(frac_M_MS, q, axis=1)
        viz.frac_M_rem = perc(frac_M_rem, q, axis=1)

        viz.BH_mass = BH_mass
        viz.BH_num = BH_num

        return viz

    def _init_velocities(self, model, mass_bin):

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

        rho_MS = np.sum(model.rhoj[:model.nms], axis=0)
        rho_MS_interp = util.QuantitySpline(model.r, rho_MS)
        rho_MS = rho_MS_interp(self.r)

        rho_tot = np.sum(model.rhoj, axis=0)
        rho_tot_interp = util.QuantitySpline(model.r, rho_tot)
        rho_tot = rho_tot_interp(self.r)

        rho_BH = np.sum(model.BH_rhoj, axis=0)
        rho_BH_interp = util.QuantitySpline(model.r, rho_BH)
        rho_BH = rho_BH_interp(self.r)

        rho_WD = np.sum(model.WD_rhoj, axis=0)
        rho_WD_interp = util.QuantitySpline(model.r, rho_WD)
        rho_WD = rho_WD_interp(self.r)

        rho_NS = np.sum(model.NS_rhoj, axis=0)
        rho_NS_interp = util.QuantitySpline(model.r, rho_NS)
        rho_NS = rho_NS_interp(self.r)

        return rho_MS, rho_tot, rho_BH, rho_WD, rho_NS

    def _init_surfdens(self, model):

        Sigma_MS = np.sum(model.Sigmaj[:model.nms], axis=0)
        Sigma_MS_interp = util.QuantitySpline(model.r, Sigma_MS)
        Sigma_MS = Sigma_MS_interp(self.r)

        Sigma_tot = np.sum(model.Sigmaj, axis=0)
        Sigma_tot_interp = util.QuantitySpline(model.r, Sigma_tot)
        Sigma_tot = Sigma_tot_interp(self.r)

        Sigma_BH = np.sum(model.BH_Sigmaj, axis=0)
        Sigma_BH_interp = util.QuantitySpline(model.r, Sigma_BH)
        Sigma_BH = Sigma_BH_interp(self.r)

        Sigma_WD = np.sum(model.WD_Sigmaj, axis=0)
        Sigma_WD_interp = util.QuantitySpline(model.r, Sigma_WD)
        Sigma_WD = Sigma_WD_interp(self.r)

        Sigma_NS = np.sum(model.NS_Sigmaj, axis=0)
        Sigma_NS_interp = util.QuantitySpline(model.r, Sigma_NS)
        Sigma_NS = Sigma_NS_interp(self.r)

        return Sigma_MS, Sigma_tot, Sigma_BH, Sigma_WD, Sigma_NS

    def _init_cum_mass(self, model):
        # TODO it seems like the integrated mass is a bit less than total Mj?

        _2πr = 2 * np.pi * model.r

        cum_M_MS = _2πr * np.sum(model.Sigmaj[:model.nms], axis=0)
        cum_M_MS_interp = util.QuantitySpline(model.r, cum_M_MS)
        cum_M_MS = [cum_M_MS_interp.integral(self.r[0], ri) for ri in self.r]

        cum_M_tot = _2πr * np.sum(model.Sigmaj, axis=0)
        cum_M_tot_interp = util.QuantitySpline(model.r, cum_M_tot)
        cum_M_tot = [cum_M_tot_interp.integral(self.r[0], ri) for ri in self.r]

        cum_M_BH = _2πr * np.sum(model.BH_Sigmaj, axis=0)
        cum_M_BH_interp = util.QuantitySpline(model.r, cum_M_BH)
        cum_M_BH = [cum_M_BH_interp.integral(self.r[0], ri) for ri in self.r]

        cum_M_WD = _2πr * np.sum(model.WD_Sigmaj, axis=0)
        cum_M_WD_interp = util.QuantitySpline(model.r, cum_M_WD)
        cum_M_WD = [cum_M_WD_interp.integral(self.r[0], ri) for ri in self.r]

        cum_M_NS = _2πr * np.sum(model.NS_Sigmaj, axis=0)
        cum_M_NS_interp = util.QuantitySpline(model.r, cum_M_NS)
        cum_M_NS = [cum_M_NS_interp.integral(self.r[0], ri) for ri in self.r]

        return cum_M_MS, cum_M_tot, cum_M_BH, cum_M_WD, cum_M_NS

    def _init_mass_frac(self, model):

        _2πr = 2 * np.pi * model.r

        dens_tot = _2πr * np.sum(model.Sigmaj, axis=0)
        int_tot = util.QuantitySpline(model.r, dens_tot)
        mass_MS = np.empty((1, self.r.size))

        dens_MS = _2πr * np.sum(model.Sigmaj[model._star_bins], axis=0)
        int_MS = util.QuantitySpline(model.r, dens_MS)
        mass_rem = np.empty((1, self.r.size))

        dens_rem = _2πr * np.sum(model.Sigmaj[model._remnant_bins], axis=0)
        int_rem = util.QuantitySpline(model.r, dens_rem)
        mass_tot = np.empty((1, self.r.size))

        mass_MS[0, 0] = mass_rem[0, 0] = mass_tot[0, 0] = np.nan

        for i in range(1, self.r.size - 2):
            mass_MS[0, i] = int_MS.integral(self.r[i], self.r[i + 1]).value
            mass_rem[0, i] = int_rem.integral(self.r[i], self.r[i + 1]).value
            mass_tot[0, i] = int_tot.integral(self.r[i], self.r[i + 1]).value

        return mass_MS / mass_tot, mass_rem / mass_tot

    def _init_numdens(self, model, equivs=None):

        model_nd = model.Sigmaj[model.nms - 1] / model.mj[model.nms - 1]

        nd_interp = util.QuantitySpline(model.r, model_nd)

        if obs_nd := self.obs.filter_datasets('*number_density'):

            if len(obs_nd) > 1:
                mssg = ('Too many number density datasets, '
                        'computing scaling factor using only final dataset')
                logging.warning(mssg)

            obs_nd = list(obs_nd.values())[-1]
            obs_r = obs_nd['r'].to(model.r.unit, equivs)

            K = (np.nansum(obs_nd['Σ'] * nd_interp(obs_r) / obs_nd['Σ']**2)
                 / np.nansum(nd_interp(obs_r)**2 / obs_nd['Σ']**2))

        else:
            mssg = 'No number density datasets found, setting K=1'
            logging.info(mssg)

            K = 1

        return (K * nd_interp(self.r)).to('pc-2', equivs)

    def _prep_massfunc(self, observations, *, cmap=None):

        cmap = cmap or plt.cm.rainbow

        massfunc = {}

        cen = (observations.mdata['RA'], observations.mdata['DEC'])

        PI_list = observations.filter_datasets('*mass_function*')

        for i, (key, mf) in enumerate(PI_list.items()):

            massfunc[key] = []

            clr = cmap(i / len(PI_list))

            field = mass.Field.from_dataset(mf, cen=cen)

            rbins = np.unique(np.c_[mf['r1'], mf['r2']], axis=0)
            rbins.sort(axis=0)

            for r_in, r_out in rbins:

                this_slc = {'r1': r_in, 'r2': r_out}

                field_slice = field.slice_radially(r_in, r_out)

                this_slc['field'] = field_slice

                this_slc['colour'] = clr

                massfunc[key].append(this_slc)

        return massfunc

    def _init_dNdm(self, model, rslice, equivs=None):
        '''returns dndm for this model in this slice'''

        densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                    for j in range(model.nms)]

        with u.set_enabled_equivalencies(equivs):
            sample_radii = rslice['field'].MC_sample(300).to(u.pc)

            dNdm = np.empty(model.nms)

            for j in range(model.nms):
                Nj = rslice['field'].MC_integrate(densityj[j], sample_radii)
                widthj = (model.mj[j] * model.mes_widths[j])
                dNdm[j] = (Nj / widthj).value

        return dNdm

    # ----------------------------------------------------------------------
    # Save and load confidence intervals to a file
    # ----------------------------------------------------------------------

    def save(self, filename, overwrite=False):
        '''save the confidence intervals to a file so we can load them more
        quickly next time. This should, in most cases, be the run output file
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

            ds = quant_grp.create_dataset('BH_mass', data=self.BH_mass)
            ds.attrs['unit'] = self.BH_mass.unit.to_string()

            ds = quant_grp.create_dataset('BH_num', data=self.BH_num)
            ds.attrs['unit'] = self.BH_num.unit.to_string()

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
    def load(cls, filename, validate=False):
        ''' load the CI from a file which was `save`d, to avoid rerunning models
        validate: check while loading that all datasets are there, error if not
        '''

        with h5py.File(filename, 'r') as file:

            try:
                modelgrp = file['model_output']
            except KeyError as err:
                mssg = f'No saved model outputs in {filename}'
                raise RuntimeError(mssg) from err

            # init class
            obs = Observations(modelgrp['metadata'].attrs['cluster'])
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
    '''
    class for making, showing, saving all the plots related to observables data,
    without any models at all
    '''

    @_ClusterVisualizer._support_units
    def _init_massfunc(self, observations, *, cmap=None):
        '''
        sets self.mass_func as a dict of PI's, where each PI has a list of
        subdicts. Each subdict represents a single radial slice (within this PI)
        and contains the radii, the mass func values, and the field slice
        '''

        cmap = cmap or plt.cm.rainbow

        self.mass_func = {}

        cen = (observations.mdata['RA'], observations.mdata['DEC'])

        PI_list = observations.filter_datasets('*mass_function*')

        for i, (key, mf) in enumerate(PI_list.items()):

            self.mass_func[key] = []

            clr = cmap(i / len(PI_list))

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

        self._init_massfunc(observations)


# --------------------------------------------------------------------------
# Collection of models
# --------------------------------------------------------------------------


class ModelCollection:
    '''A collection of models, allowing for overplotting multiple models
    with one another.
    intimately tied to RunCollection, so we'll need "posterior distributions"
    on some things like BH mass and num, etc
    '''

    def __str__(self):
        return f"Collection of Models"

    def __iter__(self):
        '''return an iterator over the individual model vizs'''
        return iter(self.visualizers)

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
    def load(cls, filenames, validate=False):
        '''Load the models stored in the results files'''

        return cls([CIModelVisualizer.load(fn, validate=validate)
                    for fn in filenames])

    def save(self, filenames, overwrite=False):
        '''save the models in the results files'''

        for fn, mv in zip(filenames, self.visualizers):
            mv.save(fn, overwrite=overwrite)

    @classmethod
    def from_models(cls, models, obs_list=None):
        '''init from a simple list of already computed of models'''

        if obs_list is None:
            obs_list = [None, ] * len(models)

        return cls([ModelVisualizer(m, o) for m, o in zip(models, obs_list)])

    @classmethod
    def from_chains(cls, chains, obs_list, ci=True, *args, **kwargs):
        '''init from an array of parameter chains for each run

        chains is a list of chains (N models long) with each chain being an
        array of either (N params,) for a from_theta init or
        (N samples, N params) for a from_chain init.
        if ci is True, must be (N samples, N params, otherwise makes no sense)
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
        '''calls each models's `plot_func`, yields a figure
        all args, kwargs passed to plot func
        '''
        for mv in self.visualizers:
            fig = getattr(mv, plot_func)(*args, **kwargs)

            yield (fig, mv) if yield_model else fig

    def save_plots(self, plot_func, fn_pattern=None, save_kw=None,
                   *args, **kwargs):
        '''
        fn_pattern is format string which will be passed the kw "cluster" name
            (i.e. `fn_pattern.format(cluster=run.name)`)
            if None, will be ./{cluster}_{plot_func[5:]}
            (Include the desired dir here too)
        '''

        if fn_pattern is None:
            fn_pattern = f'./{{cluster}}_{plot_func[5:]}'

        if save_kw is None:
            save_kw = {}

        for fig, mv in self.iter_plots(plot_func, True, *args, **kwargs):

            save_kw['fname'] = fn_pattern.format(cluster=mv.name)

            logging.info(f'Saving {mv} to {save_kw["fname"]}')

            fig.savefig(**save_kw)

    # ----------------------------------------------------------------------
    # Collection Attributes
    # ----------------------------------------------------------------------

    @property
    def BH_mass(self):

        if not self._ci:
            mssg = ("'ModelCollection' object has no attribute 'BH_mass'. "
                    "Must be constructed with CIModelVisualizer objects.")
            raise AttributeError(mssg)

        return [mv.BH_mass for mv in self.visualizers]

    @property
    def BH_num(self):

        if not self._ci:
            mssg = ("'ModelCollection' object has no attribute 'BH_num'. "
                    "Must be constructed with CIModelVisualizer objects.")
            raise AttributeError(mssg)

        return [mv.BH_num for mv in self.visualizers]

    @property
    def chi2(self):
        return [mv.chi2 for mv in self.visualizers]
