from .. import util
from ..core.data import Observations, Model
from ..probabilities import pulsars, mass

import h5py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr
import astropy.visualization as astroviz


__all__ = ['ModelVisualizer', 'CIModelVisualizer']

# TODO fix spacings

# TODO I thinkk this is somewhat out of date (20)


class _Visualizer:

    _REDUC_METHODS = {'median': np.median, 'mean': np.mean}

    def _setup_artist(self, fig, ax):
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

        return fig, ax

    # TODO this shsould handle an axes arg as well
    #   The whole point of these methods is so we can reuse the fig, but thats
    #   not currenlty possible with the multi artist like it is for singles
    def _setup_multi_artist(self, fig, shape, **subplot_kw):
        '''setup a subplot with multiple axes, don't supply ax cause it doesnt
        really make sense in this case, you would need to supply the same
        amount of axes as shape and everything, just don't deal with it'''

        if shape is None:

            axarr = []
            if fig is None:
                fig = plt.figure()

        else:

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

        return fig, axarr


class ModelVisualizer(_Visualizer):
    '''
    class for making, showing, saving all the plots related to a model
    '''

    def _support_units(method):
        import functools

        @functools.wraps(method)
        def _unit_decorator(self, *args, **kwargs):

            eqvs = [util.angular_width(self.model.d)[0],
                    util.angular_speed(self.model.d)[0]]

            with astroviz.quantity_support(), u.set_enabled_equivalencies(eqvs):
                return method(self, *args, **kwargs)

        return _unit_decorator

    def get_err(self, dataset, key):
        try:
            return dataset[f'Δ{key}']
        except KeyError:
            try:
                return (dataset[f'Δ{key},down'], dataset[f'Δ{key},up'])
            except KeyError:
                return None

    def _add_residuals(self, ax, xmodel, ymodel, xdata, ydata,
                       xerr=None, yerr=None):
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        res_ax = divider.append_axes('bottom', size="15%", pad=0, sharex=ax)

        yspline = util.QuantitySpline(xmodel, ymodel)

        res = yspline(xdata) - ydata

        res_ax.errorbar(xdata, res, fmt='k.', xerr=xerr, yerr=yerr)

        res_ax.grid()
        res_ax.axhline(0., c='k')

        res_ax.set_xscale(ax.get_xscale())

    # -----------------------------------------------------------------------
    # Plotting functions
    # -----------------------------------------------------------------------

    # TODO a better term than 'attempt' should be used
    # TODO change all these to check data for mass bin like in likelihoods

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
            a_domain, Paz = cluster_component(self.model, self.model.r[i], -1)
            maz[i] = a_domain[Paz.argmax()] << maz.unit

        maz = (self.obs['pulsar/P'] * maz).decompose()

        if show_obs:
            try:
                obs_pulsar = self.obs['pulsar']

                ax.errorbar(obs_pulsar['r'],
                            self.obs['pulsar/Pdot_meas'],
                            yerr=self.obs['pulsar/ΔPdot_meas'],
                            fmt='k.')

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_r = self.model.r.to(u.arcmin, util.angular_width(self.model.d))

        upper_az, = ax.plot(model_r[:-1], maz)
        ax.plot(model_r[:-1], -maz, c=upper_az.get_color())

        return fig

    # def plot_pulsar_distributions(self, fig=None, ax=None, show_obs=True):
        # '''plot the prob dists and the convolutions for all pulsars'''

        # N_pulsars = obs_r.size
        # prob_dist = np.array([
        #     cluster_component(self.model, A_SPACE, obs_r[i], i)
        #     for i in range(N_pulsars)
        # ])
        # max_probs = prob_dist.max(axis=1)

        # err = scipy.stats.norm.pdf(A_SPACE, 0, np.c_[obs_pulsar['Δa_los']])

        # prob_dist = likelihood_pulsars(self.model, obs_pulsar, err, True)
        # for ind in range(len(obs_pulsar['r'])):
        #     clr = f'C{ind + 1}'
        #     print(prob_dist[ind])
        #     # TO-DO lots of nans?
        #     plt.plot(A_SPACE, prob_dist[ind], c=clr)
        #     plt.axvline(obs_pulsar['r'][ind], c=clr)
        #     plt.axhline(obs_pulsar['a_los'][ind], c=clr)

        # return fig

    @_support_units
    def plot_LOS(self, fig=None, ax=None, show_obs=True, residuals=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Line of Sight Velocity Dispersion')

        ax.set_xscale("log")

        model_r = self.model.r.to(u.arcsec)
        model_LOS = np.sqrt(self.model.v2pj[mass_bin])

        if show_obs:
            try:
                veldisp = self.obs['velocity_dispersion']

                xerr = self.get_err(veldisp, 'r')
                yerr = self.get_err(veldisp, 'σ')

                ax.errorbar(veldisp['r'], veldisp['σ'], fmt='k.',
                            xerr=xerr, yerr=yerr)

                if residuals:
                    self._add_residuals(ax, model_r, model_LOS,
                                        veldisp['r'], veldisp['σ'], xerr, yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(model_r, model_LOS)

        return fig

    @_support_units
    def plot_pm_tot(self, fig=None, ax=None, show_obs=True, residuals=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Total Proper Motion")

        ax.set_xscale("log")

        model_t = np.sqrt(
            0.5 * (self.model.v2Tj[mass_bin] + self.model.v2Rj[mass_bin])
        )

        model_r = self.model.r.to(u.arcsec)
        model_t = model_t.to(u.mas / u.yr)

        if show_obs:
            try:
                pm = self.obs['proper_motion']

                xerr = self.get_err(pm, 'r')
                yerr = self.get_err(pm, 'PM_tot')

                ax.errorbar(pm['r'], pm['PM_tot'], fmt='k.',
                            xerr=xerr, yerr=yerr)

                if residuals:
                    self._add_residuals(ax, model_r, model_t,
                                        pm['r'], pm['PM_tot'], xerr, yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(model_r, model_t)

        return fig

    @_support_units
    def plot_pm_ratio(self, fig=None, ax=None, show_obs=True, residuals=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Proper Motion Ratio")

        ax.set_xscale("log")

        model_ratio = np.sqrt(self.model.v2Tj[mass_bin] / self.model.v2Rj[mass_bin])

        model_r = self.model.r.to(u.arcsec)

        if show_obs:
            try:

                pm = self.obs['proper_motion']

                xerr = self.get_err(pm, 'r')
                yerr = self.get_err(pm, 'PM_ratio')

                ax.errorbar(pm['r'], pm['PM_ratio'], fmt='k.',
                            xerr=xerr, yerr=yerr)

                if residuals:
                    self._add_residuals(ax, model_r, model_ratio,
                                        pm['r'], pm['PM_ratio'], xerr, yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(model_r, model_ratio)

        return fig

    @_support_units
    def plot_pm_T(self, fig=None, ax=None, show_obs=True, residuals=True):
        # TODO capture all mass bins which have data (high_mass, low_mass, etc)

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Tangential Proper Motion [High Mass]")

        ax.set_xscale("log")

        model_T = np.sqrt(self.model.v2Tj[mass_bin])

        model_r = self.model.r.to(u.arcsec)
        model_T = model_T.to(u.mas / u.yr)

        if show_obs:
            try:

                pm = self.obs['proper_motion/high_mass']

                xerr = self.get_err(pm, 'r')
                yerr = self.get_err(pm, 'PM_T')

                ax.errorbar(pm['r'], pm["PM_T"], fmt='k.',
                            xerr=xerr, yerr=yerr)

                if residuals:
                    self._add_residuals(ax, model_r, model_T,
                                        pm['r'], pm['PM_T'], xerr, yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(model_r, model_T)

        return fig

    @_support_units
    def plot_pm_R(self, fig=None, ax=None, show_obs=True, residuals=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Radial Proper Motion [High Mass]")

        ax.set_xscale("log")

        model_R = np.sqrt(self.model.v2Rj[mass_bin])

        model_r = self.model.r.to(u.arcsec)
        model_R = model_R.to(u.mas / u.yr)

        if show_obs:
            try:

                pm = self.obs['proper_motion/high_mass']

                xerr = self.get_err(pm, 'r')
                yerr = self.get_err(pm, 'PM_R')

                ax.errorbar(pm['r'], pm["PM_R"], fmt='k.',
                            xerr=xerr, yerr=yerr)

                if residuals:
                    self._add_residuals(ax, model_r, model_R,
                                        pm['r'], pm['PM_R'], xerr, yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(model_r, model_R)

        return fig

    @_support_units
    def plot_number_density(self, fig=None, ax=None, show_obs=True, residuals=True):
        # numdens is a bit different cause we want to compute K based on obs
        # whenever possible, even if we're not showing obs

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Number Density')

        ax.loglog()

        try:
            numdens = self.obs['number_density']

            # interpolate number density to the observed data points r
            interp_model = util.QuantitySpline(
                self.model.r,
                self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin]
            )(numdens['r'])

            K = (np.sum(numdens['Σ'] * interp_model / numdens["Σ"] ** 2)
                 / np.sum(interp_model ** 2 / numdens["Σ"] ** 2))

        except KeyError:
            K = 1.

        model_r = self.model.r.to(u.arcmin)
        model_Σ = K * self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin]

        if show_obs:
            try:

                numdens = self.obs['number_density']

                xerr = self.get_err(numdens, 'r')

                ΔΣ = self.get_err(numdens, "Σ")
                yerr = np.sqrt(ΔΣ**2 + (self.model.s2 << ΔΣ.unit**2))

                ax.errorbar(numdens['r'], numdens["Σ"], fmt='k.',
                            xerr=xerr, yerr=yerr)

                if residuals:
                    self._add_residuals(ax, model_r, model_Σ,
                                        numdens['r'], numdens['Σ'], xerr, yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(model_r, model_Σ)

        return fig

    @_support_units
    def plot_mass_func(self, fig=None, ax=None, show_obs=True, rbin_size=0.4):

        fig, ax = self._setup_artist(fig, ax)
        scale = 10
        yticks = []
        ylabels = []

        mf = self.obs['mass_function']

        # Have to use 'value' because str-based `Variable`s are still broken
        PI_arr = mf['fields'].astype(str).value

        M = 300

        # Generate the mass splines before the loops
        densityj = [util.QuantitySpline(self.model.r, self.model.Sigmaj[j])
                    for j in range(self.model.nms)]

        cen = (self.obs.mdata['RA'], self.obs.mdata['DEC'])
        fields = mass.initialize_fields(mf['fields'], cen)

        # sort based on the first r1 value of each PI, for better viz
        fields = {PI: fields[PI] for PI in
                  sorted(fields, key=lambda PI: mf['r1'][PI_arr == PI].min())}

        for PI, field in fields.items():

            PI_mask = (PI_arr == PI)

            rbins = np.c_[mf['r1'][PI_mask], mf['r2'][PI_mask]]

            mbin_mean = (mf['m1'][PI_mask] + mf['m2'][PI_mask]) / 2.
            mbin_width = mf['m2'][PI_mask] - mf['m1'][PI_mask]

            N = mf['N'][PI_mask] / mbin_width
            ΔN = mf['ΔN'][PI_mask] / mbin_width

            for r_in, r_out in np.unique(rbins, axis=0):
                r_mask = ((mf['r1'][PI_mask] == r_in)
                          & (mf['r2'][PI_mask] == r_out))

                field_slice = field.slice_radially(r_in, r_out)

                sample_radii = field_slice.MC_sample(M).to(u.pc)

                binned_N_model = np.empty(self.model.nms)
                for j in range(self.model.nms):
                    Nj = field_slice.MC_integrate(densityj[j], sample_radii)
                    widthj = (self.model.mj[j] * self.model.mes_widths[j])
                    binned_N_model[j] = (Nj / widthj).value

                N_data = N[r_mask].value
                err_data = ΔN[r_mask].value

                err = np.sqrt(err_data**2 + (self.model.F * N_data)**2)

                pnts = ax.errorbar(mbin_mean[r_mask], N_data * 10**scale, fmt='o', yerr=err * 10**scale)
                slp, = ax.plot(self.model.mj[:self.model.nms], binned_N_model * 10**scale, '-', c=pnts[0].get_color())

                yticks.append(slp.get_ydata()[0])
                ylabels.append(f"{r_in.value:.2f}'-{r_out.value:.2f}'")

                scale -= 1

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        ax.set_ylabel('dN/dm')
        ax.set_xlabel(r'Mass [$M_\odot$]')

        fig.tight_layout()

        return fig

    @_support_units
    def plot_density(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Mass Density')

        # Main sequence density
        rho_MS = np.sum(self.model.rhoj[:self.model.nms], axis=0)
        ax.plot(self.model.r, rho_MS, label='Main Sequence')

        # Total density
        rho_tot = np.sum(self.model.rhoj, axis=0)
        ax.plot(self.model.r, rho_tot, label='Total')

        # Black hole density
        rho_BH = np.sum(self.model.BH_rhoj, axis=0)
        ax.plot(self.model.r, rho_BH, label='Black Hole')

        # White Dwarf density
        rho_WD = np.sum(self.model.WD_rhoj, axis=0)
        ax.plot(self.model.r, rho_WD, label='White Dwarf')

        # Neutron Stars density
        rho_NS = np.sum(self.model.NS_rhoj, axis=0)
        ax.plot(self.model.r, rho_NS, label='Neutron Stars')

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        return fig

    @_support_units
    def plot_surface_density(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Surface Mass Density')

        # Main sequence density
        sig_MS = np.sum(self.model.Sigmaj[:self.model.nms], axis=0)
        ax.plot(self.model.r, sig_MS, label='Main Sequence')

        # Total density
        sig_MS = np.sum(self.model.Sigmaj, axis=0)
        ax.plot(self.model.r, sig_MS, label='Total')

        # Black hole density
        sig_MS = np.sum(self.model.BH_Sigmaj, axis=0)
        ax.plot(self.model.r, sig_MS, label='Black Hole')

        # White Dwarf density
        sig_WD = np.sum(self.model.WD_Sigmaj, axis=0)
        ax.plot(self.model.r, sig_WD, label='White Dwarf')

        # Neutron Stars density
        sig_NS = np.sum(self.model.NS_Sigmaj, axis=0)
        ax.plot(self.model.r, sig_NS, label='Neutron Stars')

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        return fig

    @_support_units
    def plot_cumulative_mass(self, fig=None, ax=None, x_unit='pc'):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Cumulative Mass')

        r0 = self.model.r[0]

        # for plotting in other units
        r_domain = self.model.r.to(x_unit)

        # Main sequence mass

        Sigma_MS = self.model.Sigmaj[:self.model.nms]
        sig_MS = 2 * np.pi * self.model.r * np.sum(Sigma_MS, axis=0)
        dM_MS = util.QuantitySpline(self.model.r, sig_MS)

        cum_M_MS = [dM_MS.integral(r0, ri) for ri in self.model.r]

        ax.plot(r_domain, cum_M_MS, label='Main Sequence')

        # Total mass

        sig_tot = 2 * np.pi * self.model.r * np.sum(self.model.Sigmaj, axis=0)
        dM_tot = util.QuantitySpline(self.model.r, sig_tot)

        cum_M_tot = [dM_tot.integral(r0, ri) for ri in self.model.r]

        ax.plot(r_domain, cum_M_tot, label='Total')

        # Black hole mass

        sig_BH = 2 * np.pi * self.model.r * np.sum(self.model.BH_Sigmaj, axis=0)
        dM_BH = util.QuantitySpline(self.model.r, sig_BH)

        cum_M_BH = [dM_BH.integral(r0, ri) for ri in self.model.r]

        ax.plot(r_domain, cum_M_BH, label='Black Hole')

        # White Dwarf mass

        sig_WD = 2 * np.pi * self.model.r * np.sum(self.model.WD_Sigmaj, axis=0)
        dM_WD = util.QuantitySpline(self.model.r, sig_WD)

        cum_M_WD = [dM_WD.integral(r0, ri) for ri in self.model.r]

        ax.plot(r_domain, cum_M_WD, label='White Dwarf')

        # Neutron Star mass

        sig_NS = 2 * np.pi * self.model.r * np.sum(self.model.NS_Sigmaj, axis=0)
        dM_NS = util.QuantitySpline(self.model.r, sig_NS)

        cum_M_NS = [dM_NS.integral(r0, ri) for ri in self.model.r]

        ax.plot(r_domain, cum_M_NS, label='Neutron Star')

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_ylabel(rf'$M_{{enc}} ({cum_M_tot.unit})$')

        ax.legend()

        return fig

    @_support_units
    def plot_all(self, fig=None, axes=None, show_obs='attempt'):

        # TODO have a require_obs option, where if required we change to try
        #   excepts and dont plot the ones that fail

        # TODO a better method for being able to overplot multiple show_alls
        if fig is None:
            fig, axes = plt.subplots(4, 2)
            axes = axes.flatten()
        else:
            axes = fig.axes

        fig.suptitle(str(self.obs))

        kw = {'show_obs': show_obs, 'residuals': False}

        # self.plot_pulsar(fig=fig, ax=axes[0], show_obs=show_obs)
        self.plot_number_density(fig=fig, ax=axes[1], **kw)
        self.plot_pm_tot(fig=fig, ax=axes[2], **kw)
        self.plot_pm_ratio(fig=fig, ax=axes[3], **kw)
        self.plot_pm_T(fig=fig, ax=axes[4], **kw)
        self.plot_pm_R(fig=fig, ax=axes[5], **kw)

        # self.plot_mass_func(fig=fig, ax=axes[6], show_obs=show_obs)

        self.plot_LOS(fig=fig, ax=axes[7], **kw)

        # TODO maybe have some written info in one of the empty panels (ax6)
        # fig.tight_layout()

        return fig

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

        # TODO somehow based on valid_likelihoods figure out dataset names
        # self.data_fields =


class CIModelVisualizer(_Visualizer):
    '''
    class for making, showing, saving all the plots related to a model
    but this time also including confidence intervals
    '''

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

    def _plot_model_CI(self, ax, percs, intervals=2, r_unit='pc', *,
                       CI_kwargs=None, **kwargs):

        CI_kwargs = dict() if CI_kwargs is None else CI_kwargs

        if not (percs.shape[0] % 2):
            mssg = 'Invalid `percs`, must have odd-numbered zeroth axis shape'
            raise ValueError(mssg)

        midpoint = percs.shape[0] // 2

        if intervals > midpoint:
            mssg = f'{intervals}σ is outside stored range of {midpoint}σ'
            raise ValueError(mssg)

        r_domain = self.r.to(r_unit)

        median_ = percs[midpoint]

        med_plot, = ax.plot(r_domain, median_, **kwargs)

        CI_kwargs.setdefault('color', med_plot.get_color())

        alpha = 0.8 / (intervals + 1)
        for sigma in range(1, intervals + 1):

            ax.fill_between(
                r_domain, percs[midpoint + sigma], percs[midpoint - sigma],
                alpha=(1 - alpha), **CI_kwargs
            )

            alpha += alpha

        return ax

    def get_err(self, dataset, key):
        try:
            return dataset[f'Δ{key}']
        except KeyError:
            try:
                return (dataset[f'Δ{key},down'], dataset[f'Δ{key},up'])
            except KeyError:
                return None

    # -----------------------------------------------------------------------
    # Plotting functions
    # -----------------------------------------------------------------------

    @_support_units
    def plot_LOS(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Line of Sight Velocity Dispersion')

        ax.set_xscale("log")

        if show_obs:
            try:
                veldisp = self.obs['velocity_dispersion']

                xerr = self.get_err(veldisp, 'r')
                yerr = self.get_err(veldisp, 'σ')

                ax.errorbar(veldisp['r'], veldisp['σ'], fmt='k.',
                            xerr=xerr, yerr=yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        self._plot_model_CI(ax, np.sqrt(self.v2pj))

        return fig

    @_support_units
    def plot_pm_tot(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Total Proper Motion")

        ax.set_xscale("log")

        if show_obs:
            try:
                pm = self.obs['proper_motion']

                xerr = self.get_err(pm, 'r')
                yerr = self.get_err(pm, 'PM_tot')

                ax.errorbar(pm['r'], pm['PM_tot'], fmt='k.',
                            xerr=xerr, yerr=yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_t = np.sqrt(
            0.5 * (self.v2Tj + self.v2Rj)
        )

        self._plot_model_CI(ax, model_t.to(u.mas / u.yr))

        return fig

    @_support_units
    def plot_pm_ratio(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Proper Motion Ratio")

        ax.set_xscale("log")

        if show_obs:
            try:

                pm = self.obs['proper_motion']

                xerr = self.get_err(pm, 'r')
                yerr = self.get_err(pm, 'PM_ratio')

                ax.errorbar(pm['r'], pm['PM_ratio'], fmt='k.',
                            xerr=xerr, yerr=yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_ratio2 = self.v2Tj / self.v2Rj

        self._plot_model_CI(ax, np.sqrt(model_ratio2))

        return fig

    @_support_units
    def plot_pm_T(self, fig=None, ax=None, show_obs=True):
        # TODO capture all mass bins which have data (high_mass, low_mass, etc)

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Tangential Proper Motion [High Mass]")

        ax.set_xscale("log")

        if show_obs:
            try:

                pm = self.obs['proper_motion/high_mass']

                xerr = self.get_err(pm, 'r')
                yerr = self.get_err(pm, 'PM_T')

                ax.errorbar(pm['r'], pm["PM_T"], fmt='k.',
                            xerr=xerr, yerr=yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_T = np.sqrt(self.v2Tj)

        self._plot_model_CI(ax, model_T.to(u.mas / u.yr))

        return fig

    @_support_units
    def plot_pm_R(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Radial Proper Motion [High Mass]")

        ax.set_xscale("log")

        if show_obs:
            try:

                pm = self.obs['proper_motion/high_mass']

                xerr = self.get_err(pm, 'r')
                yerr = self.get_err(pm, 'PM_R')

                ax.errorbar(pm['r'], pm["PM_R"], fmt='k.',
                            xerr=xerr, yerr=yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_R = np.sqrt(self.v2Rj)

        self._plot_model_CI(ax, model_R.to(u.mas / u.yr))

        return fig

    @_support_units
    def plot_number_density(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Number Density')

        ax.loglog()

        if show_obs:
            try:

                numdens = self.obs['number_density']

                xerr = self.get_err(numdens, 'r')

                ΔΣ = self.get_err(numdens, "Σ")
                yerr = np.sqrt(ΔΣ**2 + (self.s2 << ΔΣ.unit**2))

                ax.errorbar(numdens['r'], numdens["Σ"], fmt='k.',
                            xerr=xerr, yerr=yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        self._plot_model_CI(ax, self.numdens)

        return fig

    @_support_units
    def plot_mass_func(self, fig=None, ax=None, show_obs=True, rbin_size=0.4):

        # TODO cause this doesn't use _plot_model_CI its less robust

        fig, ax = self._setup_artist(fig, ax)

        mf = self.obs['mass_function']

        # TODO rbin size should actually come from the data,
        #   sollima is always 0.4' I think but still it should be an attr/smthn
        rbin_size <<= u.arcmin

        for annulus_ind in np.unique(mf['bin']):

            scale = 10**annulus_ind

            # Convert the radial bin baounds from arcmin to model units
            r1 = (rbin_size * annulus_ind).to(u.parsec)
            r2 = (rbin_size * (annulus_ind + 1)).to(u.parsec)

            # we only want to use the obs data for this r bin
            r_mask = (mf['bin'] == annulus_ind)

            # Grab the N_data (adjusted by width to get an average
            #                   dr of a bin (like average-interpolating almost))
            N_data = (mf['N'][r_mask] / mf['mbin_width'][r_mask]).value
            err_data = (mf['Δmbin'][r_mask] / mf['mbin_width'][r_mask]).value

            # Compute δN_model from poisson error, and nuisance factor
            err = np.sqrt(err_data**2 + (self.F * N_data)**2)

            pnts = ax.errorbar(mf['mbin_mean'][r_mask], N_data * scale,
                               fmt='o', yerr=err * scale)

            # plot contours

            midpoint = self.mass_func.shape[0] // 2

            m_domain = self.median_mj[:self.mass_func.shape[-1]]
            median_ = self.mass_func[midpoint, int(annulus_ind)] * scale

            med_plot, = ax.plot(m_domain, median_, 'x--', c=pnts[0].get_color(),
                                label=f"R={r1:.1f}-{r2:.1f}")

            alpha = 0.8 / (midpoint + 1)
            for sigma in range(1, midpoint + 1):

                ax.fill_between(
                    m_domain,
                    self.mass_func[midpoint + sigma, int(annulus_ind)] * scale,
                    self.mass_func[midpoint - sigma, int(annulus_ind)] * scale,
                    alpha=1 - alpha, color=med_plot.get_color()
                )

                alpha += alpha

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        return fig

    @_support_units
    def plot_BH_mass(self, fig=None, ax=None, bins='auto', color='b'):

        fig, ax = self._setup_artist(fig, ax)

        color = mpl_clr.to_rgb(color)
        facecolor = color + (0.33, )

        ax.hist(self.BH_mass, histtype='stepfilled',
                bins=bins, ec=color, fc=facecolor, lw=2)

        return fig

    @_support_units
    def plot_BH_num(self, fig=None, ax=None, bins='auto', color='b'):

        fig, ax = self._setup_artist(fig, ax)

        color = mpl_clr.to_rgb(color)
        facecolor = color + (0.33, )

        ax.hist(self.BH_num, histtype='stepfilled',
                bins=bins, ec=color, fc=facecolor, lw=2)

        return fig

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

        ax.set_title('Cumulative Mass')

        # Main sequence density
        if 'MS' in kind:
            self._plot_model_CI(ax, self.cum_M_MS,
                                r_unit=x_unit, label='Main Sequence')

        # Total density
        if 'tot' in kind:
            self._plot_model_CI(ax, self.cum_M_tot,
                                r_unit=x_unit, label='Total')

        # Black hole density
        if 'BH' in kind:
            self._plot_model_CI(ax, self.cum_M_BH,
                                r_unit=x_unit, label='Black Hole')

        if 'WD' in kind:
            self._plot_model_CI(ax, self.cum_M_WD,
                                r_unit=x_unit, label='White Dwarf')

        if 'NS' in kind:
            self._plot_model_CI(ax, self.cum_M_NS,
                                r_unit=x_unit, label='Neutron Star')

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.set_ylabel(rf'$M_{{enc}} ({self.cum_M_tot.unit})$')

        ax.legend()

        return fig

    @_support_units
    def plot_all(self, fig=None, axes=None, show_obs=False):

        # TODO a better method for being able to overplot multiple show_alls
        if fig is None:
            fig, axes = plt.subplots(4, 2)
            axes = axes.flatten()
        else:
            axes = fig.axes

        fig.suptitle(str(self.obs))

        # self.plot_pulsar(fig=fig, ax=axes[0], show_obs=show_obs)
        self.plot_number_density(fig=fig, ax=axes[1], show_obs=show_obs)
        self.plot_pm_tot(fig=fig, ax=axes[2], show_obs=show_obs)
        self.plot_pm_ratio(fig=fig, ax=axes[3], show_obs=show_obs)
        self.plot_pm_T(fig=fig, ax=axes[4], show_obs=show_obs)
        self.plot_pm_R(fig=fig, ax=axes[5], show_obs=show_obs)

        self.plot_LOS(fig=fig, ax=axes[7], show_obs=show_obs)

        return fig

    @classmethod
    def from_chain(cls, chain, observations, N=100, *, verbose=True):

        viz = cls()

        # Get info and models

        chain = chain.reshape((-1, chain.shape[-1]))

        viz.N = N

        viz.obs = observations

        median_chain = np.median(chain[-N:], axis=0)
        median_model = Model(median_chain, viz.obs)

        viz.F = median_chain[7]
        viz.s2 = median_chain[6]
        viz.d = median_chain[12] << u.kpc
        viz.median_mj = median_model.mj

        if verbose:
            import tqdm
            chain_loader = tqdm.tqdm(chain[-N:])
        else:
            chain_loader = chain[-N:]

        model_sample = [Model(θ, viz.obs) for θ in chain_loader]

        # Setup the radial domain to interpolate everything onto

        max_r = max(model_sample, key=lambda m: m.rt).rt
        viz.r = np.r_[0, np.geomspace(1e-5, max_r.value, num=99)] << u.pc

        # Setup the final full parameters arrays
        # TODO composite stuff like ratio prob need to be computed here,
        #   not in the plotting. seems to be creating errors

        vel_unit = model_sample[0].v2Tj.unit

        v2Tj_full = np.empty((N, viz.r.size)) << vel_unit
        v2Rj_full = np.empty((N, viz.r.size)) << vel_unit
        v2pj_full = np.empty((N, viz.r.size)) << vel_unit

        rho_unit = model_sample[0].rhoj.unit

        rho_MS_full = np.empty((N, viz.r.size)) << rho_unit
        rho_tot_full = np.empty((N, viz.r.size)) << rho_unit
        rho_BH_full = np.empty((N, viz.r.size)) << rho_unit
        rho_WD_full = np.empty((N, viz.r.size)) << rho_unit
        rho_NS_full = np.empty((N, viz.r.size)) << rho_unit

        Sigma_unit = model_sample[0].Sigmaj.unit

        Sigma_MS_full = np.empty((N, viz.r.size)) << Sigma_unit
        Sigma_tot_full = np.empty((N, viz.r.size)) << Sigma_unit
        Sigma_BH_full = np.empty((N, viz.r.size)) << Sigma_unit
        Sigma_WD_full = np.empty((N, viz.r.size)) << Sigma_unit
        Sigma_NS_full = np.empty((N, viz.r.size)) << Sigma_unit

        mass_unit = model_sample[0].M.unit

        cum_M_MS_full = np.empty((N, viz.r.size)) << mass_unit
        cum_M_tot_full = np.empty((N, viz.r.size)) << mass_unit
        cum_M_BH_full = np.empty((N, viz.r.size)) << mass_unit
        cum_M_WD_full = np.empty((N, viz.r.size)) << mass_unit
        cum_M_NS_full = np.empty((N, viz.r.size)) << mass_unit

        nd_full = np.empty((N, viz.r.size)) << u.arcmin**-2

        N_rbins = np.unique(viz.obs['mass_function/bin']).size
        N_mbins = max(model_sample, key=lambda m: m.nms).nms
        mf_full = np.empty((N, N_rbins, N_mbins))

        BH_mass = np.empty(N) << u.Msun
        BH_num = np.empty(N) << u.dimensionless_unscaled

        # Get the interpolater and interpolate every parameter

        for ind, model in enumerate(model_sample):

            mass_bin = model.nms - 1

            # Velocities

            v2Tj_interp = util.QuantitySpline(model.r, model.v2Tj[mass_bin])
            v2Tj_full[ind, :] = v2Tj_interp(viz.r)

            v2Rj_interp = util.QuantitySpline(model.r, model.v2Rj[mass_bin])
            v2Rj_full[ind, :] = v2Rj_interp(viz.r)

            v2pj_interp = util.QuantitySpline(model.r, model.v2pj[mass_bin])
            v2pj_full[ind, :] = v2pj_interp(viz.r)

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
            obs_r = obs_nd['r'].to(model.r.unit, util.angular_width(model.d))
            # TODO maybe this should actually be a part of `Model`
            model_nd = model.Sigmaj[mass_bin] / model.mj[mass_bin]

            nd_interp = util.QuantitySpline(model.r, model_nd)

            K = (np.nansum(obs_nd['Σ'] * nd_interp(obs_r) / obs_nd['Σ']**2)
                 / np.nansum(nd_interp(obs_r)**2 / obs_nd['Σ']**2))

            nd_full[ind, :] = K * nd_interp(viz.r)

            # Mass Functions

            rbin_size = 0.4 * u.arcmin

            for rbin_ind in np.unique(viz.obs['mass_function/bin']):

                # Convert the radial bin baounds from arcmin to model units
                with u.set_enabled_equivalencies(util.angular_width(model.d)):
                    r1 = (rbin_size * rbin_ind).to(u.parsec)
                    r2 = (rbin_size * (rbin_ind + 1)).to(u.parsec)

                # Get a binned version of N_model (an Nstars for each mbin)
                for mbin_ind in range(model.nms):

                    # Interpolate the viz.model density at the data locations

                    density = util.QuantitySpline(
                        model.r,
                        2 * np.pi * model.r * model.Sigmaj[mbin_ind]
                    )

                    # Convert density spline into Nstars
                    mper = model.mj[mbin_ind] * model.mes_widths[mbin_ind]

                    mf_full[ind, int(rbin_ind), mbin_ind] = (
                        density.integral(r1, r2) / mper
                    )

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

        viz.v2Tj = np.percentile(v2Tj_full, q, axis=0)
        viz.v2Rj = np.percentile(v2Rj_full, q, axis=0)
        viz.v2pj = np.percentile(v2pj_full, q, axis=0)

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

            ds = perc_grp.create_dataset('v2Tj', data=self.v2Tj)
            ds.attrs['unit'] = self.v2Tj.unit.to_string()

            ds = perc_grp.create_dataset('v2Rj', data=self.v2Rj)
            ds.attrs['unit'] = self.v2Rj.unit.to_string()

            ds = perc_grp.create_dataset('v2pj', data=self.v2pj)
            ds.attrs['unit'] = self.v2pj.unit.to_string()

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
