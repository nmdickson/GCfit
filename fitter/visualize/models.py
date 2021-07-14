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
            eqvs = [util.angular_width(self.d)[0],
                    util.angular_speed(self.d)[0]]

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

    def _plot_model(self, ax, data, intervals=None, r_unit='pc', *,
                    CI_kwargs=None, **kwargs):

        CI_kwargs = dict() if CI_kwargs is None else CI_kwargs

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

        r_domain = self.r.to(r_unit)

        median = data[midpoint]

        med_plot, = ax.plot(r_domain, median, **kwargs)

        CI_kwargs.setdefault('color', med_plot.get_color())

        alpha = 0.8 / (intervals + 1)
        for sigma in range(1, intervals + 1):

            ax.fill_between(
                r_domain, data[midpoint + sigma], data[midpoint - sigma],
                alpha=(1 - alpha), **CI_kwargs
            )

            alpha += alpha

        return ax

    def _plot_data(self, ax, dataset, y_key, *, r_key='r', r_unit='pc',
                   err_transform=None, **kwargs):

        # TODO need to handle colours better
        defaultcolour = None

        xdata = dataset[r_key]
        ydata = dataset[y_key]

        xerr = self._get_err(dataset, r_key)
        yerr = self._get_err(dataset, y_key)

        if err_transform is not None:
            yerr = err_transform(yerr)

        kwargs.setdefault('linestyle', 'None')
        kwargs.setdefault('marker', '.')
        kwargs.setdefault('color', defaultcolour)

        label = dataset.cite()
        if 'm' in dataset.mdata:
            label += fr' ($m={dataset.mdata["m"]}$)'

        ax.errorbar(xdata.to(r_unit), ydata, xerr=xerr, yerr=yerr,
                    label=label, **kwargs)

        return ax

    def _plot(self, ax, ds_pattern, y_key, model_data, *, strict=False,
              data_kwargs=None, model_kwargs=None):
        '''figure out what needs to be plotted and call model/data plotters
        '''
        ds_pattern = ds_pattern or ''

        data_kwargs = data_kwargs or {}
        model_kwargs = model_kwargs or {}

        datasets = self.obs.filter_datasets(ds_pattern, True)

        if data_kwargs.get('strict', False) and ds_pattern and not datasets:
            mssg = f"Dataset matching '{ds_pattern}' do not exist in {self.obs}"
            # raise DataError
            raise KeyError(mssg)

        masses = []

        for key, dset in datasets.items():

            # get mass bin of this dataset, for later model plotting
            if 'm' in dset.mdata:
                m = dset.mdata['m'] * u.Msun
                mass_bin = np.where(self.model.mj == m)[0][0]
            else:
                mass_bin = self.model.nms - 1

            # plot the data
            try:
                self._plot_data(ax, dset, y_key, **data_kwargs)

            except KeyError as err:
                if strict:
                    raise err
                else:
                    # warnings.warn(err.args[0])
                    continue

            if mass_bin not in masses:
                masses.append(mass_bin)

        if model_data is not None:

            # TODO make sure the colors between data and model match, if masses
            for mbin in (masses or [self.model.nms - 1]):
                self._plot_model(ax, model_data[mbin], **model_kwargs)

    # -----------------------------------------------------------------------
    # Plot extras
    # -----------------------------------------------------------------------

    def _add_residuals(self, ax, ymodel, xdata, ydata, xerr=None, yerr=None):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ymedian = self._get_median(ymodel)

        yspline = util.QuantitySpline(self.r, ymedian)

        res = yspline(xdata) - ydata

        divider = make_axes_locatable(ax)
        res_ax = divider.append_axes('bottom', size="15%", pad=0, sharex=ax)

        res_ax.errorbar(xdata, res, fmt='k.', xerr=xerr, yerr=yerr)

        res_ax.grid()

        self._plot_model(res_ax, ymodel - ymedian, color='k')

        res_ax.set_xscale(ax.get_xscale())

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
    def plot_LOS(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Line-of-Sight Velocity Dispersion')

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*velocity_dispersion*', 'σ'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        self._plot(ax, pattern, var, self.LOS, strict=strict)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_tot(self, fig=None, ax=None, show_obs=True, x_unit='pc'):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Total Proper Motion")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_tot'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        pm_tot = self.pm_tot.to('mas/yr')

        self._plot(ax, pattern, var, pm_tot, strict=strict)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_ratio(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Proper Motion Anisotropy")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_ratio'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        self._plot(ax, pattern, var, self.pm_ratio, strict=strict)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_T(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Tangential Proper Motion")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_T'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        pm_T = self.pm_T.to('mas/yr')

        self._plot(ax, pattern, var, pm_T, strict=strict)

        ax.legend()

        return fig

    @_support_units
    def plot_pm_R(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Radial Proper Motion")

        ax.set_xscale("log")

        if show_obs:
            pattern, var = '*proper_motion*', 'PM_R'
            strict = show_obs == 'strict'

        else:
            pattern = var = None
            strict = False

        pm_R = self.pm_R.to('mas/yr')

        self._plot(ax, pattern, var, pm_R, strict=strict)
        ax.legend()

        return fig

    @_support_units
    def plot_number_density(self, fig=None, ax=None, show_obs=True):

        def quad_nuisance(err):
            return np.sqrt(err**2 + (self.s2 << err.unit**2))

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Number Density')

        ax.loglog()

        self._plot_model(ax, self.numdens)

        if show_obs:
            pattern, var = '*number_density*', 'Σ'
            strict = show_obs == 'strict'
            kwargs = {'err_transform': quad_nuisance}

        else:
            pattern = var = kwargs = None
            strict = False

        self._plot(ax, pattern, var, self.numdens, strict=strict,
                   data_kwargs=kwargs)

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

    # -----------------------------------------------------------------------
    # Model plotting
    # -----------------------------------------------------------------------
    # TODO plot_remnant_fraction (see baumgardt cat)

    @_support_units
    def plot_density(self, fig=None, ax=None, *, kind='all'):

        # TODO some z-order stuff is off, and alpha needs to be better

        if kind == 'all':
            kind = {'MS', 'tot', 'BH', 'WD', 'NS'}

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Mass Density')

        # Main sequence density
        if 'MS' in kind:
            self._plot_model_CI(ax, self.rho_MS, label='Main Sequence')

        # Total density
        if 'tot' in kind:
            self._plot_model_CI(ax, self.rho_tot, label='Total')

        # Black hole density
        if 'BH' in kind:
            self._plot_model_CI(ax, self.rho_BH, label='Black Hole')

        if 'WD' in kind:
            self._plot_model_CI(ax, self.rho_WD, label='White Dwarf')

        if 'NS' in kind:
            self._plot_model_CI(ax, self.rho_NS, label='Neutron Star')

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        return fig

    @_support_units
    def plot_surface_density(self, fig=None, ax=None, *, kind='all'):

        if kind == 'all':
            kind = {'MS', 'tot', 'BH'}

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Surface Mass Density')

        # Main sequence density
        if 'MS' in kind:
            self._plot_model_CI(ax, self.Sigma_MS, label='Main Sequence')

        # Total density
        if 'tot' in kind:
            self._plot_model_CI(ax, self.Sigma_tot, label='Total')

        # Black hole density
        if 'BH' in kind:
            self._plot_model_CI(ax, self.Sigma_BH, label='Black Hole')

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

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
            self._plot_model_CI(ax, self.cum_M_tot,
                                r_unit=x_unit, label='Total', c='tab:cyan')

        # Main sequence density
        if 'MS' in kind:
            self._plot_model_CI(ax, self.cum_M_MS,
                                r_unit=x_unit, label='Main Sequence', c='tab:orange')

        if 'WD' in kind:
            self._plot_model_CI(ax, self.cum_M_WD,
                                r_unit=x_unit, label='White Dwarf', c='tab:green')

        if 'NS' in kind:
            self._plot_model_CI(ax, self.cum_M_NS,
                                r_unit=x_unit, label='Neutron Star', c='tab:red')

        # Black hole density
        if 'BH' in kind:
            self._plot_model_CI(ax, self.cum_M_BH,
                                r_unit=x_unit, label='Black Hole', c='tab:gray')

        ax.set_yscale("log")
        ax.set_xscale("log")

        # ax.set_ylabel(rf'$M_{{enc}} ({self.cum_M_tot.unit})$')
        ax.set_ylabel(rf'$M_{{enc}}$ $[M_\odot]$')
        ax.set_xlabel('arcsec')

        # ax.legend()
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=5, fancybox=True)

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

        self.F = model.F
        self.s2 = model.s2
        self.d = model.d

        self.r = model.r

        self.LOS = np.sqrt(self.model.v2pj)
        self.pm_T = np.sqrt(model.v2Tj)
        self.pm_R = np.sqrt(model.v2Rj)
        self.pm_tot = np.sqrt(0.5 * (self.pm_T**2 + self.pm_R**2))
        self.pm_ratio = self.pm_T / self.pm_R
        self.numdens = self._init_numdens(model, observations)
        self.mass_func = self._init_massfunc(model, observations)

    @_ClusterVisualizer._support_units
    def _init_numdens(self, model, observations):

        obs_nd = observations['number_density']  # <- not very general
        obs_r = obs_nd['r'].to(model.r.unit)

        model_nd = model.Sigmaj[model.nms - 1] / model.mj[model.nms - 1]

        nd_interp = util.QuantitySpline(model.r, model_nd)

        K = (np.nansum(obs_nd['Σ'] * nd_interp(obs_r) / obs_nd['Σ']**2)
             / np.nansum(nd_interp(obs_r)**2 / obs_nd['Σ']**2))

        return K * model_nd

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

        return mass_func


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

        # Get info and models

        chain = chain.reshape((-1, chain.shape[-1]))

        viz.N = N

        viz.obs = observations

        median_chain = np.median(chain[-N:], axis=0)
        # median_model = Model(median_chain, viz.obs)

        viz.F = median_chain[7]
        viz.s2 = median_chain[6]
        viz.d = median_chain[12] << u.kpc
        # viz.median_mj = median_model.mj

        if verbose:
            import tqdm
            chain_loader = tqdm.tqdm(chain[-N:])
        else:
            chain_loader = chain[-N:]

        model_sample = []
        for i, θ in enumerate(chain_loader):
            try:
                model_sample.append(Model(θ, viz.obs))
            except ValueError:
                print(f"{i} did not converge")
                continue
        N = len(model_sample)
        # model_sample = [Model(θ, viz.obs) for θ in chain_loader]

        viz.median_mj = model_sample[0].mj

        # Setup the radial domain to interpolate everything onto

        max_r = max(model_sample, key=lambda m: m.rt).rt
        viz.r = np.r_[0, np.geomspace(1e-5, max_r.value, num=99)] << u.pc

        # Setup the final full parameters arrays

        # velocities

        vel_unit = np.sqrt(model_sample[0].v2Tj).unit

        vTj_full = np.empty((N, viz.r.size)) << vel_unit
        vRj_full = np.empty((N, viz.r.size)) << vel_unit

        vtotj_full = np.empty((N, viz.r.size)) << vel_unit
        vaj_full = np.empty((N, viz.r.size)) << u.dimensionless_unscaled

        vpj_full = np.empty((N, viz.r.size)) << vel_unit

        # mass density

        rho_unit = model_sample[0].rhoj.unit

        rho_MS_full = np.empty((N, viz.r.size)) << rho_unit
        rho_tot_full = np.empty((N, viz.r.size)) << rho_unit
        rho_BH_full = np.empty((N, viz.r.size)) << rho_unit
        rho_WD_full = np.empty((N, viz.r.size)) << rho_unit
        rho_NS_full = np.empty((N, viz.r.size)) << rho_unit

        # surface density

        Sigma_unit = model_sample[0].Sigmaj.unit

        Sigma_MS_full = np.empty((N, viz.r.size)) << Sigma_unit
        Sigma_tot_full = np.empty((N, viz.r.size)) << Sigma_unit
        Sigma_BH_full = np.empty((N, viz.r.size)) << Sigma_unit
        Sigma_WD_full = np.empty((N, viz.r.size)) << Sigma_unit
        Sigma_NS_full = np.empty((N, viz.r.size)) << Sigma_unit

        # Cumulative mass

        mass_unit = model_sample[0].M.unit

        cum_M_MS_full = np.empty((N, viz.r.size)) << mass_unit
        cum_M_tot_full = np.empty((N, viz.r.size)) << mass_unit
        cum_M_BH_full = np.empty((N, viz.r.size)) << mass_unit
        cum_M_WD_full = np.empty((N, viz.r.size)) << mass_unit
        cum_M_NS_full = np.empty((N, viz.r.size)) << mass_unit

        # number density

        nd_full = np.empty((N, viz.r.size)) << u.arcmin**-2

        # mass function

        PI_list = fnmatch.filter([k[0] for k in viz.obs.valid_likelihoods],
                                 '*mass_function*')

        PI_list = sorted(PI_list, key=lambda k: viz.obs[k]['r1'].min())

        N_rbins = sum([np.unique(viz.obs[k]['r1']).size for k in PI_list])
        N_mbins = max(model_sample, key=lambda m: m.nms).nms
        mf_full = np.empty((N, N_rbins, N_mbins))

        # BH mass

        BH_mass = np.empty(N) << u.Msun
        BH_num = np.empty(N) << u.dimensionless_unscaled

        # Get the interpolater and interpolate every parameter

        for ind, model in enumerate(model_sample):

            mass_bin = model.nms - 1
            equivs = util.angular_width(model.d)

            # Velocities

            vT = np.sqrt(model.v2Tj[mass_bin])
            vTj_interp = util.QuantitySpline(model.r, vT)
            vTj_full[ind, :] = vTj_interp(viz.r)

            vR = np.sqrt(model.v2Rj[mass_bin])
            vRj_interp = util.QuantitySpline(model.r, vR)
            vRj_full[ind, :] = vRj_interp(viz.r)

            vtot = np.sqrt(0.5 * (model.v2Tj[mass_bin] + model.v2Rj[mass_bin]))
            vtotj_interp = util.QuantitySpline(model.r, vtot)
            vtotj_full[ind, :] = vtotj_interp(viz.r)

            va = np.sqrt(model.v2Tj[mass_bin] / model.v2Rj[mass_bin])
            finite = np.isnan(va)
            vaj_interp = util.QuantitySpline(model.r[~finite], va[~finite])
            vaj_full[ind, :] = vaj_interp(viz.r)

            vp = np.sqrt(model.v2pj[mass_bin])
            vpj_interp = util.QuantitySpline(model.r, vp)
            vpj_full[ind, :] = vpj_interp(viz.r)

            # Mass Densities

            rho_MS = np.sum(model.rhoj[:model.nms], axis=0)
            rho_MS_interp = util.QuantitySpline(model.r, rho_MS)
            rho_MS_full[ind, :] = rho_MS_interp(viz.r)

            rho_tot = np.sum(model.rhoj, axis=0)
            rho_tot_interp = util.QuantitySpline(model.r, rho_tot)
            rho_tot_full[ind, :] = rho_tot_interp(viz.r)

            rho_BH = np.sum(model.BH_rhoj, axis=0)
            rho_BH_interp = util.QuantitySpline(model.r, rho_BH)
            rho_BH_full[ind, :] = rho_BH_interp(viz.r)

            rho_WD = np.sum(model.WD_rhoj, axis=0)
            rho_WD_interp = util.QuantitySpline(model.r, rho_WD)
            rho_WD_full[ind, :] = rho_WD_interp(viz.r)

            rho_NS = np.sum(model.NS_rhoj, axis=0)
            rho_NS_interp = util.QuantitySpline(model.r, rho_NS)
            rho_NS_full[ind, :] = rho_NS_interp(viz.r)

            # Surface Densities

            Sigma_MS = np.sum(model.Sigmaj[:model.nms], axis=0)
            Sigma_MS_interp = util.QuantitySpline(model.r, Sigma_MS)
            Sigma_MS_full[ind, :] = Sigma_MS_interp(viz.r)

            Sigma_tot = np.sum(model.Sigmaj, axis=0)
            Sigma_tot_interp = util.QuantitySpline(model.r, Sigma_tot)
            Sigma_tot_full[ind, :] = Sigma_tot_interp(viz.r)

            Sigma_BH = np.sum(model.BH_Sigmaj, axis=0)
            Sigma_BH_interp = util.QuantitySpline(model.r, Sigma_BH)
            Sigma_BH_full[ind, :] = Sigma_BH_interp(viz.r)

            Sigma_WD = np.sum(model.WD_Sigmaj, axis=0)
            Sigma_WD_interp = util.QuantitySpline(model.r, Sigma_WD)
            Sigma_WD_full[ind, :] = Sigma_WD_interp(viz.r)

            Sigma_NS = np.sum(model.NS_Sigmaj, axis=0)
            Sigma_NS_interp = util.QuantitySpline(model.r, Sigma_NS)
            Sigma_NS_full[ind, :] = Sigma_NS_interp(viz.r)

            # Cumulative Mass distribution
            # TODO it seems like the integrated mass is a bit less than total Mj

            cum_M_MS = 2 * np.pi * model.r * Sigma_MS
            cum_M_MS_interp = util.QuantitySpline(model.r, cum_M_MS)
            cum_M_MS_full[ind, :] = [cum_M_MS_interp.integral(viz.r[0], ri)
                                     for ri in viz.r]

            cum_M_tot = 2 * np.pi * model.r * Sigma_tot
            cum_M_tot_interp = util.QuantitySpline(model.r, cum_M_tot)
            cum_M_tot_full[ind, :] = [cum_M_tot_interp.integral(viz.r[0], ri)
                                      for ri in viz.r]

            cum_M_BH = 2 * np.pi * model.r * Sigma_BH
            cum_M_BH_interp = util.QuantitySpline(model.r, cum_M_BH)
            cum_M_BH_full[ind, :] = [cum_M_BH_interp.integral(viz.r[0], ri)
                                     for ri in viz.r]

            cum_M_WD = 2 * np.pi * model.r * Sigma_WD
            cum_M_WD_interp = util.QuantitySpline(model.r, cum_M_WD)
            cum_M_WD_full[ind, :] = [cum_M_WD_interp.integral(viz.r[0], ri)
                                     for ri in viz.r]

            cum_M_NS = 2 * np.pi * model.r * Sigma_NS
            cum_M_NS_interp = util.QuantitySpline(model.r, cum_M_NS)
            cum_M_NS_full[ind, :] = [cum_M_NS_interp.integral(viz.r[0], ri)
                                     for ri in viz.r]

            # Number Densities

            obs_nd = viz.obs['number_density']
            obs_r = obs_nd['r'].to(model.r.unit, equivs)
            # TODO maybe this should actually be a part of `Model`
            model_nd = model.Sigmaj[mass_bin] / model.mj[mass_bin]

            nd_interp = util.QuantitySpline(model.r, model_nd)

            K = (np.nansum(obs_nd['Σ'] * nd_interp(obs_r) / obs_nd['Σ']**2)
                 / np.nansum(nd_interp(obs_r)**2 / obs_nd['Σ']**2))

            nd_full[ind, :] = K * nd_interp(viz.r)

            # Mass Functions

            densityj = [util.QuantitySpline(model.r, model.Sigmaj[j])
                        for j in range(model.nms)]

            rbin_ind = -1

            for key in PI_list:
                mf = viz.obs[key]

                cen = (viz.obs.mdata['RA'], viz.obs.mdata['DEC'])
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
                        mf_full[ind, rbin_ind, j] = (Nj / widthj).value

            # Black holes
            BH_mass[ind] = np.sum(model.BH_Mj)
            BH_num[ind] = np.sum(model.BH_Nj)
            # TODO Some way to quantify mass of indiv BHs? like a 2dhist of N&m?

        # compute and store the percentiles and median
        # TODO get sigmas dynamically ased on an arg
        q = [97.72, 84.13, 50., 15.87, 2.28]

        viz.rho_MS = np.percentile(rho_MS_full, q, axis=0)
        viz.rho_tot = np.percentile(rho_tot_full, q, axis=0)
        viz.rho_BH = np.percentile(rho_BH_full, q, axis=0)
        viz.rho_WD = np.percentile(rho_WD_full, q, axis=0)
        viz.rho_NS = np.percentile(rho_NS_full, q, axis=0)

        viz.vTj = np.percentile(vTj_full, q, axis=0)
        viz.vRj = np.percentile(vRj_full, q, axis=0)
        viz.vtotj = np.percentile(vtotj_full, q, axis=0)
        # print(vaj_full)
        viz.vaj = np.nanpercentile(vaj_full, q, axis=0)
        # print(viz.vaj)
        viz.vpj = np.percentile(vpj_full, q, axis=0)

        viz.Sigma_MS = np.percentile(Sigma_MS_full, q, axis=0)
        viz.Sigma_tot = np.percentile(Sigma_tot_full, q, axis=0)
        viz.Sigma_BH = np.percentile(Sigma_BH_full, q, axis=0)
        viz.Sigma_WD = np.percentile(Sigma_WD_full, q, axis=0)
        viz.Sigma_NS = np.percentile(Sigma_NS_full, q, axis=0)

        viz.cum_M_MS = np.percentile(cum_M_MS_full, q, axis=0)
        viz.cum_M_tot = np.percentile(cum_M_tot_full, q, axis=0)
        viz.cum_M_BH = np.percentile(cum_M_BH_full, q, axis=0)
        viz.cum_M_WD = np.percentile(cum_M_WD_full, q, axis=0)
        viz.cum_M_NS = np.percentile(cum_M_NS_full, q, axis=0)

        viz.numdens = np.percentile(nd_full, q, axis=0)
        viz.mass_func = np.percentile(mf_full, q, axis=0)
        viz.BH_mass = BH_mass
        viz.BH_num = BH_num

        return viz

    def save(self, filename):
        '''save the confidence intervals to a file so we can load them more
        quickly next time
        '''

        with h5py.File(filename, 'x') as file:

            meta_grp = file.create_group('metadata')

            meta_grp.create_dataset('r', data=self.r)
            meta_grp.create_dataset('median_mj', data=self.median_mj)
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

            ds = perc_grp.create_dataset('vTj', data=self.vTj)
            ds.attrs['unit'] = self.vTj.unit.to_string()

            ds = perc_grp.create_dataset('vRj', data=self.vRj)
            ds.attrs['unit'] = self.vRj.unit.to_string()

            ds = perc_grp.create_dataset('vtotj', data=self.vtotj)
            ds.attrs['unit'] = self.vtotj.unit.to_string()

            ds = perc_grp.create_dataset('vaj', data=self.vaj)
            ds.attrs['unit'] = self.vaj.unit.to_string()

            ds = perc_grp.create_dataset('vpj', data=self.vpj)
            ds.attrs['unit'] = self.vpj.unit.to_string()

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
            viz.median_mj = file['metadata']['median_mj'][:] << u.Msun

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
    pass
