from .. import util
from ..core.data import Observations, Model
from ..probabilities import pulsars, mass

import fnmatch
import string

import h5py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr
import astropy.visualization as astroviz


__all__ = ['ModelVisualizer', 'CIModelVisualizer', 'ObservationsVisualizer']

# TODO fix all spacings
# TODO add "require_data" and "require_model" decorators that check that
#   the required attribute exists, so we dont plot any completely empty plots
#   (i.e. dont even try to plot cum_mass from an ObservationsVisualizer
#   but should error nicely)

class _ClusterVisualizer:

    _REDUC_METHODS = {'median': np.median, 'mean': np.mean}

    # -----------------------------------------------------------------------
    # Artist setups
    # -----------------------------------------------------------------------

    def _setup_artist(self, fig, ax, *, use_name=True):
        # TODO should maybe attempt to use gcf, gca first
        if ax is None:
            if fig is None:
                fig, ax = plt.subplots()
            else:
                # TODO how to handle the shapes of subplots here probs read fig
                ax = fig.add_subplot()
        else:
            if fig is None:
                fig = ax.get_figure()

        if hasattr(self, 'name') and use_name:
            fig.suptitle(self.name)

        return fig, ax

    # TODO this shsould handle an axes arg as well
    #   The whole point of these methods is so we can reuse the fig, but thats
    #   not currenlty possible with the multi artist like it is for singles
    def _setup_multi_artist(self, fig, shape, *, use_name=True, **subplot_kw):
        '''setup a subplot with multiple axes, don't supply ax cause it doesnt
        really make sense in this case, you would need to supply the same
        amount of axes as shape and everything, just don't deal with it'''

        if shape is None:

            axarr = []
            if fig is None:
                fig = plt.figure()

        else:
            # TODO axarr should always be an arr, even if shape=1

            if fig is None:
                fig, axarr = plt.subplots(*shape, **subplot_kw)

            elif not fig.axes:
                fig, axarr = plt.subplots(*shape, num=fig.number, **subplot_kw)

            else:
                # TODO make sure this is using the correct order
                axarr = np.array(fig.axes).reshape(shape)
                # we shouldn't add axes to a fig that already has some
                # maybe just warn or error if the shape doesn't match `shape`
                pass

        if hasattr(self, 'name') and use_name:
            fig.suptitle(self.name)

        return fig, axarr

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
                return (dataset[f'Δ{key},down'], dataset[f'Δ{key},up'])
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
        # Setup plotting details, style, labels
        # ------------------------------------------------------------------

        kwargs.setdefault('marker', '.')
        kwargs.setdefault('linestyle', 'None')
        kwargs.setdefault('color', defaultcolour)

        label = dataset.cite()
        if 'm' in dataset.mdata:
            label += fr' ($m={dataset.mdata["m"]}$)'

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------

        return ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr,
                           label=label, **kwargs)

    def _plot(self, ax, ds_pattern, y_key, model_data, *, residuals=False,
              **kwargs):
        '''figure out what needs to be plotted and call model/data plotters
        all **kwargs passed to both _plot_model and _plot_data
        model_data dimensions *must* be (mass bins, intervals, r axis)
        '''

        # TODO we might still want to allow for specific model/data kwargs?

        ds_pattern = ds_pattern or ''

        strict = kwargs.pop('strict', False)

        # ------------------------------------------------------------------
        # Determine the relevant datasets to the given pattern
        # ------------------------------------------------------------------

        datasets = self.obs.filter_datasets(ds_pattern)

        if strict and ds_pattern and not datasets:
            mssg = f"Dataset matching '{ds_pattern}' do not exist in {self.obs}"
            # raise DataError
            raise KeyError(mssg)

        # ------------------------------------------------------------------
        # Iterate over the datasets, keeping track of all relevant masses
        # and calling `_plot_data`
        # ------------------------------------------------------------------

        # TODO this new handling of masses (for residuals) is way fragile
        masses = {}

        for key, dset in datasets.items():

            # get mass bin of this dataset, for later model plotting
            if 'm' in dset.mdata:
                m = dset.mdata['m'] * u.Msun
                mass_bin = np.where(self.mj == m)[0][0]
            else:
                mass_bin = self.star_bin

            # plot the data
            try:
                line = self._plot_data(ax, dset, y_key, **kwargs)

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

            N_mbin = model_data.shape[0]

            if not masses:
                if N_mbin > 1:
                    masses = [self.star_bin]
                else:
                    masses = [0]

            # TODO make sure the colors between data and model match, if masses
            for mbin in masses:

                ymodel = model_data[mbin, :, :]

                self._plot_model(ax, ymodel, **kwargs)

                if residuals:

                    try:
                        errorbars = masses[mbin]

                    except TypeError:
                        mssg = (f"Cannot plot residuals for mass={mbin}, "
                                "no corresponding data has been plotted")
                        raise ValueError(mssg)

                    self._add_residuals(ax, ymodel, errorbars)

    # -----------------------------------------------------------------------
    # Plot extras
    # -----------------------------------------------------------------------

    def _add_residuals(self, ax, ymodel, errorbars, *, xmodel=None):
        '''
        errorbars : a list of outputs from calls to plt.errorbars
        '''
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # ------------------------------------------------------------------
        # Get model data and spline
        # ------------------------------------------------------------------

        if xmodel is None:
            xmodel = self.r

        ymedian = self._get_median(ymodel)

        yspline = util.QuantitySpline(xmodel, ymedian)

        # ------------------------------------------------------------------
        # Setup axes, adding a new smaller axe for the residual underneath
        # ------------------------------------------------------------------

        divider = make_axes_locatable(ax)
        res_ax = divider.append_axes('bottom', size="15%", pad=0, sharex=ax)

        res_ax.grid()

        res_ax.set_xscale(ax.get_xscale())

        # ------------------------------------------------------------------
        # Plot the model line, hopefully centred on zero
        # ------------------------------------------------------------------

        self._plot_model(res_ax, ymodel - ymedian, color='k')

        # ------------------------------------------------------------------
        # Get data from the plotted errorbars
        # ------------------------------------------------------------------

        # TODO also copy all formatting from each errorbars to this one

        for errbar in errorbars:

            xdata, ydata = errbar[0].get_data()
            xerr = yerr = None

            if errbar.has_xerr:
                xerr_lines = errbar[2][0]
                yerr_lines = errbar[2][1] if errbar.has_yerr else None
            elif errbar.has_yerr:
                xerr_lines, yerr_lines = None, errbar[2][0]
            else:
                xerr_lines = yerr_lines = None

            if xerr_lines:
                xerr = np.array([(np.diff(seg, axis=0) / 2)[..., -1]
                                 for seg in xerr_lines.get_segments()]).T[0]

                xerr <<= xdata.unit

            if yerr_lines:
                yerr = np.array([(np.diff(seg, axis=0) / 2)[..., -1]
                                 for seg in yerr_lines.get_segments()]).T[0]

                yerr <<= ydata.unit

            res = yspline(xdata) - ydata

            res_ax.errorbar(xdata, res, fmt='k.', xerr=xerr, yerr=yerr)

    def _add_hyperparam(self, ax, ymodel, xdata, ydata, yerr):
        # TODO this is still a bit of a mess

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
    def plot_LOS(self, fig=None, ax=None, show_obs=True, residuals=False,
                 x_unit='pc', y_unit='km/s'):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Line-of-Sight Velocity Dispersion')

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*velocity_dispersion*', 'σ'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        self._plot(ax, pattern, var, self.LOS,
                   strict=strict, residuals=residuals,
                   x_unit=x_unit, y_unit=y_unit)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_tot(self, fig=None, ax=None,
                    show_obs=True, residuals=False,
                    x_unit='pc', y_unit='mas/yr'):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Total Proper Motion")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_tot'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        self._plot(ax, pattern, var, self.pm_tot,
                   strict=strict, residuals=residuals,
                   x_unit=x_unit, y_unit=y_unit)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_ratio(self, fig=None, ax=None,
                      show_obs=True, residuals=False,
                      x_unit='pc'):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Proper Motion Anisotropy")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_ratio'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        self._plot(ax, pattern, var, self.pm_ratio,
                   strict=strict, residuals=residuals,
                   x_unit=x_unit)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_T(self, fig=None, ax=None,
                  show_obs=True, residuals=False,
                  x_unit='pc', y_unit='mas/yr'):

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

        self._plot(ax, pattern, var, self.pm_T,
                   strict=strict, residuals=residuals,
                   x_unit=x_unit, y_unit=y_unit)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_R(self, fig=None, ax=None,
                  show_obs=True, residuals=False,
                  x_unit='pc', y_unit='mas/yr'):

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

        self._plot(ax, pattern, var, self.pm_R,
                   strict=strict, residuals=residuals,
                   x_unit=x_unit, y_unit=y_unit)

        ax.legend()

        return fig

    @_support_units
    def plot_number_density(self, fig=None, ax=None,
                            show_obs=True, residuals=False,
                            x_unit='pc'):

        def quad_nuisance(err):
            return np.sqrt(err**2 + (self.s2 << err.unit**2))

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Number Density')

        ax.loglog()

        if show_obs:
            pattern, var = '*number_density*', 'Σ'
            strict = show_obs == 'strict'
            kwargs = {'err_transform': quad_nuisance}

        else:
            pattern = var = None
            strict = False
            kwargs = {}

        self._plot(ax, pattern, var, self.numdens,
                   strict=strict, residuals=residuals,
                   x_unit=x_unit, **kwargs)

        ax.legend()

        return fig

    @_support_units
    def plot_mass_func(self, fig=None, ax=None, show_obs=True):

        # TODO cause this doesn't use _plot_model_CI its less robust

        fig, ax = self._setup_artist(fig, ax)

        scale = 10
        # yticks = []
        # ylabels = []

        XTEXT = 0.81 * u.Msun
        # ax.annotate("Radial Bins", (XTEXT, 10**scale), fontsize=12)
        # scale -= 4

        PI_list = fnmatch.filter([k[0] for k in self.obs.valid_likelihoods],
                                 '*mass_function*')

        PI_list = sorted(PI_list, key=lambda k: self.obs[k]['r1'].min())

        rbin = 0

        for key in PI_list:
            mf = self.obs[key]

            rbins = np.c_[mf['r1'], mf['r2']]

            mbin_mean = (mf['m1'] + mf['m2']) / 2.
            mbin_width = mf['m2'] - mf['m1']

            N = mf['N'] / mbin_width
            ΔN = mf['ΔN'] / mbin_width

            for r_in, r_out in np.unique(rbins, axis=0):
                r_mask = ((mf['r1'] == r_in)
                          & (mf['r2'] == r_out))

                N_data = N[r_mask].value
                err_data = ΔN[r_mask].value

                err = self.F * err_data

                pnts = ax.errorbar(mbin_mean[r_mask], N_data * 10**scale,
                                   fmt='o', yerr=err * 10**scale)

                clr = pnts[0].get_color()

                # plot contours

                midpoint = self.mass_func.shape[0] // 2

                m_domain = self.model.mj[:self.mass_func.shape[-1]]
                median = self.mass_func[midpoint, rbin] * 10**scale

                med_plot, = ax.plot(m_domain, median, '--', c=clr,
                                    label=f"R={r_in:.1f}-{r_out:.1f}")

                alpha = 0.8 / (midpoint + 1)
                for sigma in range(1, midpoint + 1):

                    ax.fill_between(
                        m_domain,
                        self.mass_func[midpoint + sigma, rbin] * 10**scale,
                        self.mass_func[midpoint - sigma, rbin] * 10**scale,
                        alpha=1 - alpha, color=clr
                    )

                    alpha += alpha

                # yticks.append(med_plot.get_ydata()[0])
                # ylabels.append(f"{r_in.value:.2f}'-{r_out.value:.2f}'")

                xy_pnt = (med_plot.get_xdata()[-1], med_plot.get_ydata()[-1])
                xy_txt = (XTEXT, med_plot.get_ydata()[-1])
                text = f"{r_in.value:.2f}'-{r_out.value:.2f}'"

                ax.annotate(text, xy_pnt, xytext=xy_txt, fontsize=12, color=clr)

                scale -= 1
                rbin += 1

        ax.set_yscale("log")
        ax.set_xscale("log")

        # ax.set_yticks(yticks)
        # ax.set_yticklabels(ylabels)

        ax.set_ylabel('dN/dm')
        ax.set_xlabel(r'Mass [$M_\odot$]')

        fig.tight_layout()

        return fig

    @_support_units
    def plot_pulsar(self, fig=None, ax=None, show_obs=True):
        # TODO this is out of date with the new pulsar probability code
        # TODO I dont even think this is what we should use anymore, but the
        #   new convolved distributions peak

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Pulsar LOS Acceleration')
        ax.set_xlabel('R')
        ax.set_ylabel(r'$a_{los}$')

        maz = u.Quantity(np.empty(self.model.nstep - 1), '1/s')
        for i in range(self.model.nstep - 1):
            a_domain, Paz = pulsars.cluster_component(self.model, self.model.r[i], -1)
            maz[i] = a_domain[Paz.argmax()] << maz.unit

        maz = (self.obs['pulsar/P'] * maz).decompose()

        if show_obs:
            try:
                obs_pulsar = self.obs['pulsar']

                ax.errorbar(obs_pulsar['r'],
                            self.obs['pulsar/Pdot'],
                            yerr=self.obs['pulsar/ΔPdot'],
                            fmt='k.')

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_r = self.model.r.to(u.arcmin, util.angular_width(self.model.d))

        upper_az, = ax.plot(model_r[:-1], maz)
        ax.plot(model_r[:-1], -maz, c=upper_az.get_color())

        return fig

    @_support_units
    def plot_pulsar_spin_dist(self, fig=None, ax=None, pulsar_ind=0,
                              show_obs=True, show_conv=False):

        import scipy.interpolate as interp

        fig, ax = self._setup_artist(fig, ax)

        # pulsars = self.obs['pulsar']
        puls_obs = self.obs['pulsar/spin']

        id_ = puls_obs['id'][pulsar_ind].value.decode()
        ax.set_title(f'Pulsar "{id_}" Period Derivative Likelihood')

        ax.set_ylabel('Probability')
        ax.set_xlabel(r'$\dot{P}/P$ $\left[s^{-1}\right]$')

        mass_bin = -1

        kde = pulsars.field_Pdot_KDE()
        Pdot_min, Pdot_max = kde.dataset[1].min(), kde.dataset[1].max()

        R = puls_obs['r'][pulsar_ind].to(u.pc)

        P = puls_obs['P'][pulsar_ind].to('s')

        Pdot_meas = puls_obs['Pdot'][pulsar_ind]
        ΔPdot_meas = np.abs(puls_obs['ΔPdot'][pulsar_ind])

        PdotP_domain, PdotP_c_prob = pulsars.cluster_component(self.model,
                                                               R, mass_bin)
        Pdot_domain = (P * PdotP_domain).decompose()

        # linear to avoid effects around asymptote
        Pdot_c_spl = interp.UnivariateSpline(
            Pdot_domain, PdotP_c_prob, k=1, s=0, ext=1
        )

        err = util.gaussian(x=Pdot_domain, sigma=ΔPdot_meas, mu=0)

        err_spl = interp.UnivariateSpline(Pdot_domain, err, k=3, s=0, ext=1)

        lg_P = np.log10(P / P.unit)

        P_grid, Pdot_int_domain = np.mgrid[lg_P:lg_P:1j, Pdot_min:Pdot_max:200j]

        P_grid, Pdot_int_domain = P_grid.ravel(), Pdot_int_domain.ravel()

        Pdot_int_prob = kde(np.vstack([P_grid, Pdot_int_domain]))

        Pdot_int_spl = interp.UnivariateSpline(
            Pdot_int_domain, Pdot_int_prob, k=3, s=0, ext=1
        )

        Pdot_int_prob = util.RV_transform(
            domain=10**Pdot_int_domain, f_X=Pdot_int_spl,
            h=np.log10, h_prime=lambda y: (1 / (np.log(10) * y))
        )

        Pdot_int_spl = interp.UnivariateSpline(
            10**Pdot_int_domain, Pdot_int_prob, k=3, s=0, ext=1
        )

        lin_domain = np.linspace(0., 1e-18, 5_000 // 2)
        lin_domain = np.concatenate((np.flip(-lin_domain[1:]), lin_domain))

        conv1 = np.convolve(err_spl(lin_domain), Pdot_c_spl(lin_domain), 'same')

        conv2 = np.convolve(conv1, Pdot_int_spl(lin_domain), 'same')

        # Normalize
        conv2 /= interp.UnivariateSpline(
            lin_domain, conv2, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        cluster_μ = self.obs.mdata['μ'] << u.Unit("mas/yr")
        PdotP_pm = pulsars.shklovskii_component(cluster_μ, self.model.d)

        cluster_coords = (self.obs.mdata['b'], self.obs.mdata['l']) * u.deg
        PdotP_gal = pulsars.galactic_component(*cluster_coords, D=self.model.d)

        x_total = (lin_domain / P) + PdotP_pm + PdotP_gal
        ax.plot(x_total, conv2)

        if show_conv:
            # Will really mess the scaling up, usually
            ax.plot(x_total, Pdot_c_spl(lin_domain))
            ax.plot(x_total, conv1)

        if show_obs:
            ax.axvline((Pdot_meas / P).decompose(), c='r', ls=':')

        prob_dist = interp.interp1d(
            (lin_domain / P) + PdotP_pm + PdotP_gal, conv2,
            assume_sorted=True, bounds_error=False, fill_value=0.0
        )

        print('prob=', prob_dist((Pdot_meas / P).decompose()))

        return fig

    @_support_units
    def plot_pulsar_orbital_dist(self, fig=None, ax=None, pulsar_ind=0,
                                 show_obs=True, show_conv=False):

        import scipy.interpolate as interp

        fig, ax = self._setup_artist(fig, ax)

        # pulsars = self.obs['pulsar']
        puls_obs = self.obs['pulsar/orbital']

        id_ = puls_obs['id'][pulsar_ind].value.decode()
        ax.set_title(f'Pulsar "{id_}" Period Derivative Likelihood')

        ax.set_ylabel('Probability')
        ax.set_xlabel(r'$\dot{P}/P$ $\left[s^{-1}\right]$')

        mass_bin = -1

        R = puls_obs['r'][pulsar_ind].to(u.pc)

        P = puls_obs['Pb'][pulsar_ind].to('s')

        Pdot_meas = puls_obs['Pbdot'][pulsar_ind]
        ΔPdot_meas = np.abs(puls_obs['ΔPbdot'][pulsar_ind])

        PdotP_domain, PdotP_c_prob = pulsars.cluster_component(self.model,
                                                               R, mass_bin)
        Pdot_domain = (P * PdotP_domain).decompose()

        Pdot_c_spl = interp.UnivariateSpline(
            Pdot_domain, PdotP_c_prob, k=1, s=0, ext=1
        )

        err = util.gaussian(x=Pdot_domain, sigma=ΔPdot_meas, mu=0)

        err_spl = interp.UnivariateSpline(Pdot_domain, err, k=3, s=0, ext=1)

        lin_domain = np.linspace(0., 1e-11, 5_000 // 2)
        lin_domain = np.concatenate((np.flip(-lin_domain[1:]), lin_domain))

        conv = np.convolve(err_spl(lin_domain), Pdot_c_spl(lin_domain), 'same')
        # conv = np.convolve(err, PdotP_c_prob, 'same')

        # Normalize
        conv /= interp.UnivariateSpline(
            lin_domain, conv, k=3, s=0, ext=1
        ).integral(-np.inf, np.inf)

        cluster_μ = self.obs.mdata['μ'] << u.Unit("mas/yr")
        PdotP_pm = pulsars.shklovskii_component(cluster_μ, self.model.d)

        cluster_coords = (self.obs.mdata['b'], self.obs.mdata['l']) * u.deg
        PdotP_gal = pulsars.galactic_component(*cluster_coords, D=self.model.d)

        x_total = (lin_domain / P) + PdotP_pm + PdotP_gal
        ax.plot(x_total, conv)

        if show_conv:
            # Will really mess the scaling up, usually
            ax.plot(x_total, PdotP_c_prob)
            ax.plot(x_total, conv)

        if show_obs:
            ax.axvline((Pdot_meas / P).decompose(), c='r', ls=':')

        prob_dist = interp.interp1d(
            x_total, conv,
            assume_sorted=True, bounds_error=False, fill_value=0.0
        )

        print('prob=', prob_dist((Pdot_meas / P).decompose()))

        return fig

    @_support_units
    def plot_all(self, fig=None, axes=None, show_obs='attempt'):

        # TODO a better method for being able to overplot multiple show_alls?
        fig, axes = plt.subplots(4, 2)
        # if fig is None:
        #     fig, axes = plt.subplots(4, 2)
        #     axes = axes.flatten()
        # else:
        #     axes = fig.axes

        fig.suptitle(str(self.obs))

        # kw = {'show_obs': show_obs, 'residuals': False, 'hyperparam': True}
        kw = {}

        self.plot_number_density(fig=fig, ax=axes[0, 0], **kw)
        self.plot_LOS(fig=fig, ax=axes[1, 0], **kw)
        self.plot_pm_tot(fig=fig, ax=axes[0, 1], **kw)
        self.plot_pm_T(fig=fig, ax=axes[1, 1], **kw)
        self.plot_pm_R(fig=fig, ax=axes[2, 1], **kw)
        self.plot_pm_ratio(fig=fig, ax=axes[3, 1], **kw)

        gs = axes[2, 0].get_gridspec()

        for ax in axes[2:, 0]:
            ax.remove()

        # Is this what is messing up the spacing?
        axbig = fig.add_subplot(gs[2:, 0])

        self.plot_mass_func(fig=fig, ax=axbig, show_obs=show_obs,)
                            # residuals=False, hyperparam=True)

        for ax in axes.flatten():
            ax.set_xlabel('')

        # TODO maybe have some written info in one of the empty panels (ax6)
        # fig.tight_layout()

        return fig

    @_support_units
    def plot_MF_fields(self, fig=None, ax=None, cmap=None, radii=("rh",)):
        '''plot all mass function fields in this observation
        '''
        import shapely.geometry as geom

        # TODO some of this should probably be in an init, not here

        fig, ax = self._setup_artist(fig, ax)

        PI_list = self.obs.filter_datasets('*mass_function*')
        PI_list = sorted(PI_list, key=lambda k: self.obs[k]['r1'].min())

        cmap = cmap or plt.cm.rainbow
        fc = iter(cmap(np.linspace(0, 1, len(PI_list))))

        for key in PI_list:
            mf = self.obs[key]

            # TODO this function should go in obs or mass or something
            #   including the new single coords check
            cen = (self.obs.mdata['RA'], self.obs.mdata['DEC'])
            unit = mf.mdata['field_unit']
            coords = []
            for ch in string.ascii_letters:
                try:
                    coords.append(mf['fields'].mdata[f'{ch}'])
                except KeyError:
                    break

            if len(coords) == 1:
                coords = coords[0]

            field = mass.Field(coords, cen=cen, unit=unit)

            field.plot(ax, fc=next(fc), alpha=0.7, ec='k', label=key)

        ax.plot(0, 0, 'kx')

        # try to plot the various radii from this model
        try:
            # TODO for CI this could be a CI of rh, ra, rt actually

            for r_type in radii:
                radius = getattr(self, r_type).to_value('arcmin')
                circle = np.array(geom.Point(0, 0).buffer(radius).exterior).T
                ax.plot(*circle)
                ax.text(0, circle[1].max(), r_type)

        except AttributeError:
            pass

        ax.set_xlabel('RA [arcmin]')
        ax.set_ylabel('DEC [arcmin]')

        ax.legend()

        return fig

    # -----------------------------------------------------------------------
    # Model plotting
    # -----------------------------------------------------------------------

    @_support_units
    def plot_density(self, fig=None, ax=None, *, kind='all', x_unit='pc'):

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Surface Mass Density')

        # Total density
        if 'tot' in kind:
            kw = {"label": "Total", "c": "tab:cyan"}
            self._plot(ax, None, None, self.rho_tot, x_unit=x_unit, **kw)

        # Total Remnant density
        if 'rem' in kind:
            kw = {"label": "Total", "c": "tab:purple"}
            self._plot(ax, None, None, self.rho_rem, x_unit=x_unit, **kw)

        # Main sequence density
        if 'MS' in kind:
            kw = {"label": "Main-sequence stars", "c": "tab:orange"}
            self._plot(ax, None, None, self.rho_MS, x_unit=x_unit, **kw)

        if 'WD' in kind:
            kw = {"label": "White Dwarfs", "c": "tab:green"}
            self._plot(ax, None, None, self.rho_WD, x_unit=x_unit, **kw)

        if 'NS' in kind:
            kw = {"label": "Neutron Stars", "c": "tab:red"}
            self._plot(ax, None, None, self.rho_NS, x_unit=x_unit, **kw)

        # Black hole density
        if 'BH' in kind:
            kw = {"label": "Black Holes", "c": "tab:gray"}
            self._plot(ax, None, None, self.rho_BH, x_unit=x_unit, **kw)

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_ylabel(rf'Surface Density $[M_\odot / pc^3]$')
        # ax.set_xlabel('arcsec')

        # ax.legend()
        fig.legend(loc='upper center', ncol=6,
                   bbox_to_anchor=(0.5, 1.), fancybox=True)

        return fig

    @_support_units
    def plot_surface_density(self, fig=None, ax=None, *, kind='all',
                             x_unit='pc'):

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Surface Mass Density')

        # Total density
        if 'tot' in kind:
            kw = {"label": "Total", "c": "tab:cyan"}
            self._plot(ax, None, None, self.Sigma_tot, x_unit=x_unit, **kw)

        # Total Remnant density
        if 'rem' in kind:
            kw = {"label": "Total", "c": "tab:purple"}
            self._plot(ax, None, None, self.Sigma_rem, x_unit=x_unit, **kw)

        # Main sequence density
        if 'MS' in kind:
            kw = {"label": "Main-sequence stars", "c": "tab:orange"}
            self._plot(ax, None, None, self.Sigma_MS, x_unit=x_unit, **kw)

        if 'WD' in kind:
            kw = {"label": "White Dwarfs", "c": "tab:green"}
            self._plot(ax, None, None, self.Sigma_WD, x_unit=x_unit, **kw)

        if 'NS' in kind:
            kw = {"label": "Neutron Stars", "c": "tab:red"}
            self._plot(ax, None, None, self.Sigma_NS, x_unit=x_unit, **kw)

        # Black hole density
        if 'BH' in kind:
            kw = {"label": "Black Holes", "c": "tab:gray"}
            self._plot(ax, None, None, self.Sigma_BH, x_unit=x_unit, **kw)

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_ylabel(rf'Surface Density $[M_\odot / pc^2]$')
        # ax.set_xlabel('arcsec')

        # ax.legend()
        fig.legend(loc='upper center', ncol=6,
                   bbox_to_anchor=(0.5, 1.), fancybox=True)

        return fig

    @_support_units
    def plot_cumulative_mass(self, fig=None, ax=None, *,
                             kind='all', x_unit='pc'):

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        fig, ax = self._setup_artist(fig, ax)

        # ax.set_title('Cumulative Mass')

        # Total density
        if 'tot' in kind:
            kw = {"label": "Total", "c": "tab:cyan"}
            self._plot(ax, None, None, self.cum_M_tot, x_unit=x_unit, **kw)

        # Main sequence density
        if 'MS' in kind:
            kw = {"label": "Main-sequence stars", "c": "tab:orange"}
            self._plot(ax, None, None, self.cum_M_MS, x_unit=x_unit, **kw)

        if 'WD' in kind:
            kw = {"label": "White Dwarfs", "c": "tab:green"}
            self._plot(ax, None, None, self.cum_M_WD, x_unit=x_unit, **kw)

        if 'NS' in kind:
            kw = {"label": "Neutron Stars", "c": "tab:red"}
            self._plot(ax, None, None, self.cum_M_NS, x_unit=x_unit, **kw)

        # Black hole density
        if 'BH' in kind:
            kw = {"label": "Black Holes", "c": "tab:gray"}
            self._plot(ax, None, None, self.cum_M_BH, x_unit=x_unit, **kw)

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

        self._plot(ax, None, None, self.frac_M_MS,
                   x_unit=x_unit, label="Main-sequence stars")
        self._plot(ax, None, None, self.frac_M_rem,
                   x_unit=x_unit, label="Remnants")

        ax.set_ylabel(r"Mass fraction $M_{MS}/M_{tot}$, $M_{remn.}/M_{tot}$")

        ax.set_ylim(0.0, 1.0)

        ax.legend()

        return fig

# --------------------------------------------------------------------------
# Visualizers
# --------------------------------------------------------------------------


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

        reduc = cls._REDUC_METHODS[method]

        # if 3d (Niters, Nwalkers, Nparams)
        # if 2d (Nwalkers, Nparams)
        # if 1d (Nparams)
        chain = chain.reshape((-1, chain.shape[-1]))

        theta = reduc(chain, axis=0)

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

        self.rh = model.rh
        self.ra = model.ra
        self.rt = model.rt
        self.F = model.F
        self.s2 = model.s2
        self.d = model.d

        self.r = model.r
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

        obs_nd = observations['number_density']  # <- TODO not very general
        obs_r = obs_nd['r'].to(model.r.unit)

        model_nd = model.Sigmaj / model.mj[:, np.newaxis]

        nd = np.empty(model_nd.shape)[:, np.newaxis, :]

        for mbin in range(model_nd.shape[0]):
            nd_interp = util.QuantitySpline(model.r, model_nd[mbin, :])

            K = (np.nansum(obs_nd['Σ'] * nd_interp(obs_r) / obs_nd['Σ']**2)
                 / np.nansum(nd_interp(obs_r)**2 / obs_nd['Σ']**2))

            nd[mbin, 0, :] = K * model_nd[mbin, :]

        self.numdens = nd

    @_ClusterVisualizer._support_units
    def _init_massfunc(self, model, observations):

        # TODO don't treat any PI different than we would any subgroup
        #   might need to add an offset param to plotdata, or redo this logic

        PI_list = fnmatch.filter([k[0] for k in observations.valid_likelihoods],
                                 '*mass_function*')

        PI_list = sorted(PI_list, key=lambda k: observations[k]['r1'].min())

        N_rbins = sum([np.unique(observations[k]['r1']).size for k in PI_list])
        N_mbins = model.nms
        # mf_full = np.empty((N, N_rbins, N_mbins))
        mass_func = np.empty((1, N_rbins, N_mbins))

        densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                    for j in range(model.nms)]

        rbin_ind = -1

        for key in PI_list:
            mf = observations[key]

            cen = (observations.mdata['RA'], observations.mdata['DEC'])
            unit = mf.mdata['field_unit']
            coords = []
            for ch in string.ascii_letters:
                try:
                    coords.append(mf['fields'].mdata[f'{ch}'])
                except KeyError:
                    break

            field = mass.Field(coords, cen=cen, unit=unit)

            rbins = np.c_[mf['r1'], mf['r2']]

            for r_in, r_out in np.unique(rbins, axis=0):
                rbin_ind += 1

                field_slice = field.slice_radially(r_in, r_out)
                sample_radii = field_slice.MC_sample(300).to(u.pc)

                for j in range(model.nms):
                    Nj = field_slice.MC_integrate(densityj[j], sample_radii)
                    widthj = (model.mj[j] * model.mes_widths[j])
                    mass_func[0, rbin_ind, j] = (Nj / widthj).value

        self.mass_func = mass_func

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

    @classmethod
    def from_chain(cls, chain, observations, N=100, *, verbose=True):

        viz = cls()

        # ------------------------------------------------------------------
        # Get info about the chain and set of models
        # ------------------------------------------------------------------

        chain = chain.reshape((-1, chain.shape[-1]))

        viz.N = N

        viz.obs = observations

        median_chain = np.median(chain[-N:], axis=0)
        # median_model = Model(median_chain, viz.obs)  # Doesnt converge a lot

        viz.F = median_chain[7]
        viz.s2 = median_chain[6]
        viz.d = median_chain[12] << u.kpc

        if verbose:
            import tqdm
            chain_loader = tqdm.tqdm(chain[-N:])
        else:
            chain_loader = chain[-N:]

        # TODO uses up an inordinate amount of memory, should be generator (5)
        # model_sample = [Model(θ, viz.obs) for θ in chain_loader]
        model_sample = []
        for i, θ in enumerate(chain_loader):
            try:
                model_sample.append(Model(θ, viz.obs))
            except ValueError:
                print(f"{i} did not converge")
                continue

        N = len(model_sample)  # Account for unconverged models

        ex_model = model_sample[0]

        viz.star_bin = 0
        mj_MS = ex_model.mj[ex_model.nms - 1]
        mj_tracer = ex_model.mj[ex_model._tracer_bins]
        viz.mj = np.r_[mj_MS, mj_tracer]

        # Setup the radial domain to interpolate everything onto

        max_r = max(model_sample, key=lambda m: m.rt).rt
        viz.r = np.r_[0, np.geomspace(1e-5, max_r.value, num=99)] << u.pc

        # ------------------------------------------------------------------
        # Setup the final full parameters arrays with dims of
        # [mass bins, intervals (from percentile of models), radial bins]
        # ------------------------------------------------------------------

        # velocities

        vel_unit = np.sqrt(ex_model.v2Tj).unit

        Nm = 1 + len(ex_model.mj[ex_model._tracer_bins])

        vTj = np.empty((Nm, N, viz.r.size)) << vel_unit
        vRj = np.empty((Nm, N, viz.r.size)) << vel_unit
        vtotj = np.empty((Nm, N, viz.r.size)) << vel_unit
        vaj = np.empty((Nm, N, viz.r.size)) << u.dimensionless_unscaled
        vpj = np.empty((Nm, N, viz.r.size)) << vel_unit

        # mass density

        rho_unit = ex_model.rhoj.unit

        rho_MS = np.empty((1, N, viz.r.size)) << rho_unit
        rho_tot = np.empty((1, N, viz.r.size)) << rho_unit
        rho_BH = np.empty((1, N, viz.r.size)) << rho_unit
        rho_WD = np.empty((1, N, viz.r.size)) << rho_unit
        rho_NS = np.empty((1, N, viz.r.size)) << rho_unit

        # surface density

        Sigma_unit = ex_model.Sigmaj.unit

        Sigma_MS = np.empty((1, N, viz.r.size)) << Sigma_unit
        Sigma_tot = np.empty((1, N, viz.r.size)) << Sigma_unit
        Sigma_BH = np.empty((1, N, viz.r.size)) << Sigma_unit
        Sigma_WD = np.empty((1, N, viz.r.size)) << Sigma_unit
        Sigma_NS = np.empty((1, N, viz.r.size)) << Sigma_unit

        # Cumulative mass

        mass_unit = ex_model.M.unit

        cum_M_MS = np.empty((1, N, viz.r.size)) << mass_unit
        cum_M_tot = np.empty((1, N, viz.r.size)) << mass_unit
        cum_M_BH = np.empty((1, N, viz.r.size)) << mass_unit
        cum_M_WD = np.empty((1, N, viz.r.size)) << mass_unit
        cum_M_NS = np.empty((1, N, viz.r.size)) << mass_unit

        # Mass Fraction

        frac_M_MS = np.empty((1, N, viz.r.size)) << u.dimensionless_unscaled
        frac_M_rem = np.empty((1, N, viz.r.size)) << u.dimensionless_unscaled

        # number density

        numdens = np.empty((1, N, viz.r.size)) << u.arcmin**-2

        # mass function

        # TODO no clue how to set this up
        # Beggining to think its wrong to try and force the mf into the same
        # format as all these other profiles, as its not a profile at all

        # massfunc = np.empty((N, N_rbins, N_mbins))

        # BH mass (exclusive to CI models, so it's different)

        BH_mass = np.empty(N) << u.Msun
        BH_num = np.empty(N) << u.dimensionless_unscaled

        # ------------------------------------------------------------------
        # iterate over all models in the sample and compute/store their
        # relevant parameters
        # ------------------------------------------------------------------

        for model_ind, model in enumerate(model_sample):

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

            numdens[slc] = viz._init_numdens(model, viz.obs, equivs=equivs)

            # Mass Functions

            # massfunc[slc] = viz._init_massfunc(model, viz.obs, equivs=equivs)

            # Mass Fractions

            frac_M_MS[slc], frac_M_rem[slc] = viz._init_mass_frac(model)

            # Black holes

            BH_mass[model_ind] = np.sum(model.BH_Mj)
            BH_num[model_ind] = np.sum(model.BH_Nj)

        # ------------------------------------------------------------------
        # compute and store the percentiles and medians
        # ------------------------------------------------------------------

        # TODO get sigmas dynamically ased on an arg
        q = [97.72, 84.13, 50., 15.87, 2.28]

        axes = (1, 0, 2)  # `np.percentile` messes up the dimensions

        viz.pm_T = np.transpose(np.percentile(vTj, q, axis=1), axes)
        viz.pm_R = np.transpose(np.percentile(vRj, q, axis=1), axes)
        viz.pm_tot = np.transpose(np.percentile(vtotj, q, axis=1), axes)
        viz.pm_ratio = np.transpose(np.nanpercentile(vaj, q, axis=1), axes)
        viz.LOS = np.transpose(np.percentile(vpj, q, axis=1), axes)

        viz.rho_MS = np.transpose(np.percentile(rho_MS, q, axis=1), axes)
        viz.rho_tot = np.transpose(np.percentile(rho_tot, q, axis=1), axes)
        viz.rho_BH = np.transpose(np.percentile(rho_BH, q, axis=1), axes)
        viz.rho_WD = np.transpose(np.percentile(rho_WD, q, axis=1), axes)
        viz.rho_NS = np.transpose(np.percentile(rho_NS, q, axis=1), axes)

        viz.Sigma_MS = np.transpose(np.percentile(Sigma_MS, q, axis=1), axes)
        viz.Sigma_tot = np.transpose(np.percentile(Sigma_tot, q, axis=1), axes)
        viz.Sigma_BH = np.transpose(np.percentile(Sigma_BH, q, axis=1), axes)
        viz.Sigma_WD = np.transpose(np.percentile(Sigma_WD, q, axis=1), axes)
        viz.Sigma_NS = np.transpose(np.percentile(Sigma_NS, q, axis=1), axes)

        viz.cum_M_MS = np.transpose(np.percentile(cum_M_MS, q, axis=1), axes)
        viz.cum_M_tot = np.transpose(np.percentile(cum_M_tot, q, axis=1), axes)
        viz.cum_M_BH = np.transpose(np.percentile(cum_M_BH, q, axis=1), axes)
        viz.cum_M_WD = np.transpose(np.percentile(cum_M_WD, q, axis=1), axes)
        viz.cum_M_NS = np.transpose(np.percentile(cum_M_NS, q, axis=1), axes)

        viz.numdens = np.transpose(np.percentile(numdens, q, axis=1), axes)
        # viz.mass_func = np.percentile(massfunc, q, axis=1)
        viz.mass_func = np.empty((0, 0, 0))

        viz.frac_M_MS = np.percentile(frac_M_MS, q, axis=1)
        viz.frac_M_rem = np.percentile(frac_M_rem, q, axis=1)

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
        va_interp = util.QuantitySpline(model.r[~finite], va[~finite])
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

    def _init_numdens(self, model, observations, equivs=None):

        obs_nd = observations['number_density']
        obs_r = obs_nd['r'].to(model.r.unit, equivs)

        model_nd = model.Sigmaj[model.nms - 1] / model.mj[model.nms - 1]

        nd_interp = util.QuantitySpline(model.r, model_nd)

        K = (np.nansum(obs_nd['Σ'] * nd_interp(obs_r) / obs_nd['Σ']**2)
             / np.nansum(nd_interp(obs_r)**2 / obs_nd['Σ']**2))

        return K * nd_interp(self.r)

    def _init_massfunc(self, model, observations, equivs=None):
        # TODO I really don't think this will work at all at this point (25)

        densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                    for j in range(model.nms)]

        # TODO change these to filter_likelihoods obviously
        PI_list = fnmatch.filter([k[0] for k in observations.valid_likelihoods],
                                 '*mass_function*')

        PI_list = sorted(PI_list, key=lambda k: observations[k]['r1'].min())

        N_rbins = sum([np.unique(observations[k]['r1']).size for k in PI_list])
        N_mbins = max(model_sample, key=lambda m: m.nms).nms

        rbin_ind = -1

        # TODO units?
        mf = np.empty((N_rbins, N_mbins))

        for key in PI_list:
            mf = observations[key]

            cen = (observations.mdata['RA'], observations.mdata['DEC'])
            unit = mf.mdata['field_unit']
            coords = []
            for ch in string.ascii_letters:
                try:
                    coords.append(mf['fields'].mdata[f'{ch}'])
                except KeyError:
                    break

            field = mass.Field(coords, cen=cen, unit=unit)

            rbins = np.c_[mf['r1'], mf['r2']]

            for r_in, r_out in np.unique(rbins, axis=0):
                rbin_ind += 1

                with u.set_enabled_equivalencies(equivs):
                    field_slice = field.slice_radially(r_in, r_out)
                    sample_radii = field_slice.MC_sample(300).to(u.pc)

                    for j in range(model.nms):
                        Nj = field_slice.MC_integrate(densityj[j], sample_radii)
                        widthj = (model.mj[j] * model.mes_widths[j])
                        mf[rbin_ind, j] = (Nj / widthj).value

        return mf

    # ----------------------------------------------------------------------
    # Save and load confidence intervals to a file
    # ----------------------------------------------------------------------

    def save(self, filename):
        '''save the confidence intervals to a file so we can load them more
        quickly next time
        '''

        with h5py.File(filename, 'x') as file:

            meta_grp = file.create_group('metadata')

            meta_grp.create_dataset('r', data=self.r)
            meta_grp.create_dataset('star_bin', data=self.star_bin)
            meta_grp.create_dataset('mj', data=self.mj)
            meta_grp.attrs['s2'] = self.s2
            meta_grp.attrs['F'] = self.F
            meta_grp.attrs['d'] = self.d
            meta_grp.attrs['N'] = self.N
            meta_grp.attrs['cluster'] = self.obs.cluster

            perc_grp = file.create_group('percentiles')

            ds = perc_grp.create_dataset('rho_MS', data=self.rho_MS)
            ds.attrs['unit'] = self.rho_MS.unit.to_string()

            ds = perc_grp.create_dataset('rho_tot', data=self.rho_tot)
            ds.attrs['unit'] = self.rho_tot.unit.to_string()

            ds = perc_grp.create_dataset('rho_BH', data=self.rho_BH)
            ds.attrs['unit'] = self.rho_BH.unit.to_string()

            ds = perc_grp.create_dataset('rho_WD', data=self.rho_WD)
            ds.attrs['unit'] = self.rho_WD.unit.to_string()

            ds = perc_grp.create_dataset('rho_NS', data=self.rho_NS)
            ds.attrs['unit'] = self.rho_NS.unit.to_string()

            ds = perc_grp.create_dataset('pm_T', data=self.pm_T)
            ds.attrs['unit'] = self.pm_T.unit.to_string()

            ds = perc_grp.create_dataset('pm_R', data=self.pm_R)
            ds.attrs['unit'] = self.pm_R.unit.to_string()

            ds = perc_grp.create_dataset('pm_tot', data=self.pm_tot)
            ds.attrs['unit'] = self.pm_tot.unit.to_string()

            ds = perc_grp.create_dataset('pm_ratio', data=self.pm_ratio)
            ds.attrs['unit'] = self.pm_ratio.unit.to_string()

            ds = perc_grp.create_dataset('LOS', data=self.LOS)
            ds.attrs['unit'] = self.LOS.unit.to_string()

            ds = perc_grp.create_dataset('Sigma_MS', data=self.Sigma_MS)
            ds.attrs['unit'] = self.Sigma_MS.unit.to_string()

            ds = perc_grp.create_dataset('Sigma_tot', data=self.Sigma_tot)
            ds.attrs['unit'] = self.Sigma_tot.unit.to_string()

            ds = perc_grp.create_dataset('Sigma_BH', data=self.Sigma_BH)
            ds.attrs['unit'] = self.Sigma_BH.unit.to_string()

            ds = perc_grp.create_dataset('Sigma_WD', data=self.Sigma_WD)
            ds.attrs['unit'] = self.Sigma_WD.unit.to_string()

            ds = perc_grp.create_dataset('Sigma_NS', data=self.Sigma_NS)
            ds.attrs['unit'] = self.Sigma_NS.unit.to_string()

            ds = perc_grp.create_dataset('cum_M_MS', data=self.cum_M_MS)
            ds.attrs['unit'] = self.cum_M_MS.unit.to_string()

            ds = perc_grp.create_dataset('cum_M_tot', data=self.cum_M_tot)
            ds.attrs['unit'] = self.cum_M_tot.unit.to_string()

            ds = perc_grp.create_dataset('cum_M_BH', data=self.cum_M_BH)
            ds.attrs['unit'] = self.cum_M_BH.unit.to_string()

            ds = perc_grp.create_dataset('cum_M_WD', data=self.cum_M_WD)
            ds.attrs['unit'] = self.cum_M_WD.unit.to_string()

            ds = perc_grp.create_dataset('cum_M_NS', data=self.cum_M_NS)
            ds.attrs['unit'] = self.cum_M_NS.unit.to_string()

            ds = perc_grp.create_dataset('frac_M_MS', data=self.frac_M_MS)
            ds.attrs['unit'] = self.frac_M_MS.unit.to_string()

            ds = perc_grp.create_dataset('frac_M_rem', data=self.frac_M_rem)
            ds.attrs['unit'] = self.frac_M_rem.unit.to_string()

            ds = perc_grp.create_dataset('numdens', data=self.numdens)
            ds.attrs['unit'] = self.numdens.unit.to_string()

            ds = perc_grp.create_dataset('mass_func', data=self.mass_func)

            ds = perc_grp.create_dataset('BH_mass', data=self.BH_mass)
            ds.attrs['unit'] = self.BH_mass.unit.to_string()

            ds = perc_grp.create_dataset('BH_num', data=self.BH_num)
            ds.attrs['unit'] = self.BH_num.unit.to_string()

    @classmethod
    def load(cls, filename, validate=False):
        ''' load the CI from a file which was `save`d, to avoid rerunning models
        validate: check while loading that all datasets are there, error if not
        '''

        viz = cls()

        with h5py.File(filename, 'r') as file:

            viz.obs = Observations(file['metadata'].attrs['cluster'])
            viz.N = file['metadata'].attrs['N']
            viz.s2 = file['metadata'].attrs['s2']
            viz.F = file['metadata'].attrs['F']
            viz.d = file['metadata'].attrs['d'] << u.kpc

            viz.r = file['metadata']['r'][:] << u.pc
            viz.mj = file['metadata']['mj'][:] << u.Msun

            for key in file['percentiles']:
                value = file['percentiles'][key][:]

                try:
                    value *= u.Unit(file['percentiles'][key].attrs['unit'])
                except KeyError:
                    pass

                setattr(viz, key, value)

        return viz


class ObservationsVisualizer(_ClusterVisualizer):
    '''
    class for making, showing, saving all the plots related to observables data,
    without any models at all
    '''

    def __init__(self, observations, d=None):
        self.obs = observations

        self.star_bin = None

        self.d = (d or observations.initials['d']) << u.kpc
        self.s2 = 0.
        self.F = 1.

        self.pm_T = None
        self.pm_R = None
        self.pm_tot = None
        self.pm_ratio = None
        self.LOS = None
        self.numdens = None
        self.mass_func = None
        # These dont even plot data so they dont make sense, see way above todo
        # self.rho_MS
        # self.rho_tot
        # self.rho_BH
        # self.rho_WD
        # self.rho_NS
        # self.Sigma_MS
        # self.Sigma_tot
        # self.Sigma_BH
        # self.Sigma_WD
        # self.Sigma_NS
        # self.cum_M_MS
        # self.cum_M_tot
        # self.cum_M_BH
        # self.cum_M_WD
        # self.cum_M_NS
        # self.frac_M_MS
        # self.frac_M_rem
