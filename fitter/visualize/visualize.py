from .. import util
from ..data import Observations, Model
from ..probabilities.new_Paz import vec_Paz

import sys

import h5py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr
import scipy.integrate as integ
import astropy.visualization as astroviz


__all__ = ['RunVisualizer', 'ModelVisualizer', 'CIModelVisualizer']


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

        from astropy.constants import c

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Pulsar LOS Acceleration')
        ax.set_xlabel('R')
        ax.set_ylabel(r'$a_{los}$')

        maz = u.Quantity(np.empty(self.model.nstep - 1), 'm/s^2')
        for i in range(self.model.nstep - 1):
            a_domain, Paz = vec_Paz(self.model, self.model.r[i], -1)
            maz[i] = a_domain[Paz.argmax()] << maz.unit

        maz = (self.obs['pulsar/P'] * maz / c).decompose()

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
        #     vec_Paz(self.model, A_SPACE, obs_r[i], i)
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
    def plot_LOS(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

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

        model_r = self.model.r.to(u.arcsec)

        ax.plot(model_r, np.sqrt(self.model.v2pj[mass_bin]))

        return fig

    @_support_units
    def plot_pm_tot(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

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
            0.5 * (self.model.v2Tj[mass_bin] + self.model.v2Rj[mass_bin])
        )

        model_r = self.model.r.to(u.arcsec)
        model_t = model_t.to(u.mas / u.yr)

        ax.plot(model_r, model_t)

        return fig

    @_support_units
    def plot_pm_ratio(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

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

        model_ratio2 = self.model.v2Tj[mass_bin] / self.model.v2Rj[mass_bin]

        model_r = self.model.r.to(u.arcsec)

        ax.plot(model_r, np.sqrt(model_ratio2))

        return fig

    @_support_units
    def plot_pm_T(self, fig=None, ax=None, show_obs=True):
        # TODO capture all mass bins which have data (high_mass, low_mass, etc)

        mass_bin = self.model.nms - 1

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

        model_T = np.sqrt(self.model.v2Tj[mass_bin])

        model_r = self.model.r.to(u.arcsec)
        model_T = model_T.to(u.mas / u.yr)

        ax.plot(model_r, model_T)

        return fig

    @_support_units
    def plot_pm_R(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

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

        model_R = np.sqrt(self.model.v2Rj[mass_bin])

        model_r = self.model.r.to(u.arcsec)
        model_R = model_R.to(u.mas / u.yr)

        ax.plot(model_r, model_R)

        return fig

    @_support_units
    def plot_number_density(self, fig=None, ax=None, show_obs=True):
        # numdens is a bit different cause we want to compute K based on obs
        # whenever possible, even if we're not showing obs

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Number Density')

        ax.loglog()

        try:
            numdens = self.obs['number_density']

            # interpolate number density to the observed data points r
            interp_model = np.interp(
                numdens['r'].to(u.arcmin), self.model.r.to(u.arcmin),
                self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin]
            )

            K = (np.sum(numdens['Σ'] * interp_model / numdens["Σ"] ** 2)
                 / np.sum(interp_model ** 2 / numdens["Σ"] ** 2))

        except KeyError:
            K = 1.

        if show_obs:
            try:

                numdens = self.obs['number_density']

                xerr = self.get_err(numdens, 'r')

                ΔΣ = self.get_err(numdens, "Σ")
                yerr = np.sqrt(ΔΣ**2 + (self.model.s2 << ΔΣ.unit**2))

                ax.errorbar(numdens['r'], numdens["Σ"], fmt='k.',
                            xerr=xerr, yerr=yerr)

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_r = self.model.r.to(u.arcmin)
        model_Σ = K * self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin]

        ax.plot(model_r, model_Σ)

        return fig

    @_support_units
    def plot_mass_func(self, fig=None, ax=None, show_obs=True, rbin_size=0.4):

        import scipy.interpolate as interp

        fig, ax = self._setup_artist(fig, ax)

        mf = self.obs['mass_function']

        # TODO rbin size should actually come from the data,
        #   sollima is always 0.4' I think but still it should be an attr/smthn
        rbin_size <<= u.arcmin

        # TODO I have tho think about how this all works, should it depend so
        #   much on the obs?
        for annulus_ind in np.unique(mf['bin']):

            scale = 10**annulus_ind

            # we only want to use the obs data for this r bin
            r_mask = (mf['bin'] == annulus_ind)

            # Convert the radial bin baounds from arcmin to model units
            r1 = (rbin_size * annulus_ind).to(u.parsec)
            r2 = (rbin_size * (annulus_ind + 1)).to(u.parsec)

            # Get a binned version of N_model (an Nstars for each mbin)
            binned_N_model = np.empty(self.model.nms)
            for mbin_ind in range(self.model.nms):

                # Interpolate the self.model density at the data locations
                density = interp.interp1d(
                    self.model.r,
                    2 * np.pi * self.model.r * self.model.Sigmaj[mbin_ind],
                    kind="cubic"
                )

                # Convert density spline into Nstars
                mper = self.model.mj[mbin_ind] * self.model.mes_widths[mbin_ind]

                binned_N_model[mbin_ind] = (
                    integ.quad(density, r1.value, r2.value)[0] / mper.value
                )

            # Grab the N_data (adjusted by width to get an average
            #                   dr of a bin (like average-interpolating almost))
            N_data = (mf['N'][r_mask] / mf['mbin_width'][r_mask]).value
            err_data = (mf['Δmbin'][r_mask] / mf['mbin_width'][r_mask]).value

            # Compute δN_model from poisson error, and nuisance factor
            err = np.sqrt(err_data**2 + (self.model.F * N_data)**2)

            pnts = ax.errorbar(mf['mbin_mean'][r_mask], N_data * scale,
                               fmt='o', yerr=err * scale)

            ax.plot(self.model.mj[:self.model.nms], binned_N_model * scale,
                    'x--', c=pnts[0].get_color(), label=f"R={r1:.1f}-{r2:.1f}")

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

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

        ax.set_yscale("log")
        ax.set_xscale("log")

        ax.legend()

        return fig

    @_support_units
    def plot_surface_density(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Surface Mass Density')

        # Main sequence density
        rho_MS = np.sum(self.model.Sigmaj[:self.model.nms], axis=0)
        ax.plot(self.model.r, rho_MS, label='Main Sequence')

        # Total density
        rho_MS = np.sum(self.model.Sigmaj, axis=0)
        ax.plot(self.model.r, rho_MS, label='Total')

        # Black hole density
        rho_MS = np.sum(self.model.BH_Sigmaj, axis=0)
        ax.plot(self.model.r, rho_MS, label='Black Hole')

        ax.set_yscale("log")
        ax.set_xscale("log")

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

        self.plot_pulsar(fig=fig, ax=axes[0], show_obs=show_obs)
        self.plot_number_density(fig=fig, ax=axes[1], show_obs='attempt')
        self.plot_pm_tot(fig=fig, ax=axes[2], show_obs=show_obs)
        self.plot_pm_ratio(fig=fig, ax=axes[3], show_obs=show_obs)
        self.plot_pm_T(fig=fig, ax=axes[4], show_obs=show_obs)
        self.plot_pm_R(fig=fig, ax=axes[5], show_obs=show_obs)

        self.plot_LOS(fig=fig, ax=axes[7], show_obs=show_obs)

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

    def _plot_model_CI(self, ax, percs, intervals=2, CI_kwargs=None, **kwargs):

        CI_kwargs = dict() if CI_kwargs is None else CI_kwargs

        if not (percs.shape[0] % 2):
            mssg = 'Invalid `percs`, must have odd-numbered zeroth axis shape'
            raise ValueError(mssg)

        midpoint = percs.shape[0] // 2

        if intervals > midpoint:
            mssg = f'{intervals}σ is outside stored range of {midpoint}σ'
            raise ValueError(mssg)

        median_ = percs[midpoint]

        med_plot, = ax.plot(self.r, median_, **kwargs)

        CI_kwargs.setdefault('color', med_plot.get_color())

        alpha = 0.8 / (intervals + 1)
        for sigma in range(1, intervals + 1):

            ax.fill_between(
                self.r, percs[midpoint + sigma], percs[midpoint - sigma],
                alpha=(1 - alpha), **CI_kwargs
            )

            alpha += alpha

        return ax

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

            # TODO THIS WONT WORK< HAVENT STORED A mj

            m_domain = self.mj[:self.mass_func.shape[-1]]
            median_ = self.mass_func[midpoint] * scale

            med_plot, = ax.plot(m_domain, median_, 'x--', c=pnts[0].get_color(),
                                label=f"R={r1:.1f}-{r2:.1f}")

            alpha = 0.8 / (midpoint + 1)
            for sigma in range(1, midpoint + 1):

                ax.fill_between(
                    m_domain,
                    self.mass_func[midpoint + sigma] * scale,
                    self.mass_func[midpoint - sigma] * scale,
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
    def plot_density(self, fig=None, ax=None, *, kind='all'):

        if kind == 'all':
            kind = {'MS', 'tot', 'BH'}

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
    def from_chain(cls, chain, observations, M=100, *, verbose=True):

        viz = cls()

        # Get info and models

        chain = chain.reshape((-1, chain.shape[-1]))

        viz.M = M

        viz.obs = observations

        viz.d = np.median(chain[-M:][:, -1], axis=0) << u.kpc

        if verbose:
            import tqdm
            chain_loader = tqdm.tqdm(chain[-M:])
        else:
            chain_loader = chain[-M:]

        model_sample = [Model(θ, viz.obs) for θ in chain_loader]

        # Setup the radial domain to interpolate everything onto

        max_r = max(model_sample, key=lambda m: m.rt).rt
        viz.r = np.r_[0, np.geomspace(1e-5, max_r.value, num=99)] << u.pc

        # Setup the final full parameters arrays

        vel_unit = model_sample[0].v2Tj.unit

        v2Tj_full = np.empty((M, viz.r.size)) << vel_unit
        v2Rj_full = np.empty((M, viz.r.size)) << vel_unit
        v2pj_full = np.empty((M, viz.r.size)) << vel_unit

        rho_unit = model_sample[0].rhoj.unit

        rho_MS_full = np.empty((M, viz.r.size)) << rho_unit
        rho_tot_full = np.empty((M, viz.r.size)) << rho_unit
        rho_BH_full = np.empty((M, viz.r.size)) << rho_unit
        # rho_WD_full = np.empty((M, viz.r.size)) << rho_unit

        Sigma_unit = model_sample[0].Sigmaj.unit

        Sigma_MS_full = np.empty((M, viz.r.size)) << Sigma_unit
        Sigma_tot_full = np.empty((M, viz.r.size)) << Sigma_unit
        Sigma_BH_full = np.empty((M, viz.r.size)) << Sigma_unit

        nd_full = np.empty((M, viz.r.size)) << u.arcmin**-2

        N_rbins = np.unique(viz.obs['mass_function/bin']).size
        N_mbins = max(model_sample, key=lambda m: m.nms).nms
        mf_full = np.empty((M, N_rbins, N_mbins))

        BH_mass = np.empty(M) << u.Msun

        # Get the interpolater and interpolate every parameter

        for ind, model in enumerate(model_sample):

            # Do this for BH and stuff too??
            # Not really sure how to handle mass bin
            mass_bin = model.nms - 1

            # Velocities

            v2Tj_interp = util.interpQuantity(model.r, model.v2Tj[mass_bin])
            v2Tj_full[ind, :] = v2Tj_interp(viz.r)

            v2Rj_interp = util.interpQuantity(model.r, model.v2Rj[mass_bin])
            v2Rj_full[ind, :] = v2Rj_interp(viz.r)

            v2pj_interp = util.interpQuantity(model.r, model.v2pj[mass_bin])
            v2pj_full[ind, :] = v2pj_interp(viz.r)

            # Mass Densities

            rho_MS = np.sum(model.rhoj[:model.nms], axis=0)
            rho_MS_interp = util.interpQuantity(model.r, rho_MS)
            rho_MS_full[ind, :] = rho_MS_interp(viz.r)

            rho_tot = np.sum(model.rhoj, axis=0)
            rho_tot_interp = util.interpQuantity(model.r, rho_tot)
            rho_tot_full[ind, :] = rho_tot_interp(viz.r)

            rho_BH = np.sum(model.BH_rhoj, axis=0)
            rho_BH_interp = util.interpQuantity(model.r, rho_BH)
            rho_BH_full[ind, :] = rho_BH_interp(viz.r)

            # rho_WD = np.sum(model.WD_rhoj, axis=0)
            # rho_WD_interp = util.interpQuantity(model.r, rho_WD)
            # rho_WD_full[ind, :] = rho_WD_interp(viz.r)

            # Surface Densities

            Sigma_MS = np.sum(model.Sigmaj[:model.nms], axis=0)
            Sigma_MS_interp = util.interpQuantity(model.r, Sigma_MS)
            Sigma_MS_full[ind, :] = Sigma_MS_interp(viz.r)

            Sigma_tot = np.sum(model.Sigmaj, axis=0)
            Sigma_tot_interp = util.interpQuantity(model.r, Sigma_tot)
            Sigma_tot_full[ind, :] = Sigma_tot_interp(viz.r)

            Sigma_BH = np.sum(model.BH_Sigmaj, axis=0)
            Sigma_BH_interp = util.interpQuantity(model.r, Sigma_BH)
            Sigma_BH_full[ind, :] = Sigma_BH_interp(viz.r)

            # Number Densities

            obs_nd = viz.obs['number_density']
            obs_r = obs_nd['r'].to(model.r.unit, util.angular_width(model.d))
            # TODO maybe this should actually be a part of `Model`
            model_nd = model.Sigmaj[mass_bin] / model.mj[mass_bin]

            nd_interp = util.interpQuantity(model.r, model_nd)

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
                    # TODO cant use interpQuantity here until the integ works
                    import scipy.interpolate
                    density = scipy.interpolate.interp1d(
                        model.r,
                        2 * np.pi * model.r * model.Sigmaj[mbin_ind],
                        kind="cubic"
                    )

                    # Convert density spline into Nstars
                    mper = model.mj[mbin_ind] * model.mes_widths[mbin_ind]

                    mf_full[ind, int(rbin_ind), mbin_ind] = (
                        integ.quad(density, r1.value, r2.value)[0] / mper.value
                    )

            # Black hole mass
            BH_mass[ind] = np.sum(model.BH_Mj)

        # compute and store the percentiles and median
        # TODO get sigmas dynamically ased on an arg
        q = [97.72, 84.13, 50., 15.87, 2.28]

        viz.rho_MS = np.percentile(rho_MS_full, q, axis=0)
        viz.rho_tot = np.percentile(rho_tot_full, q, axis=0)
        viz.rho_BH = np.percentile(rho_BH_full, q, axis=0)

        viz.v2Tj = np.percentile(v2Tj_full, q, axis=0)
        viz.v2Rj = np.percentile(v2Rj_full, q, axis=0)
        viz.v2pj = np.percentile(v2pj_full, q, axis=0)

        viz.Sigma_MS = np.percentile(Sigma_MS_full, q, axis=0)
        viz.Sigma_tot = np.percentile(Sigma_tot_full, q, axis=0)
        viz.Sigma_BH = np.percentile(Sigma_BH_full, q, axis=0)

        viz.numdens = np.percentile(nd_full, q, axis=0)
        viz.mass_func = np.percentile(mf_full, q, axis=0)
        viz.BH_mass = BH_mass

        return viz

    def save(self, filename):
        '''save the confidence intervals to a file so we can load them more
        quickly next time
        '''

        with h5py.File(filename, 'x') as file:

            meta_grp = file.create_group('metadata')

            meta_grp.create_dataset('r', data=self.r)
            meta_grp.attrs['d'] = self.d
            meta_grp.attrs['M'] = self.M
            meta_grp.attrs['cluster'] = self.obs.cluster

            perc_grp = file.create_group('percentiles')

            ds = perc_grp.create_dataset('rho_MS', data=self.rho_MS)
            ds.attrs['unit'] = self.rho_MS.unit.to_string()

            ds = perc_grp.create_dataset('rho_tot', data=self.rho_tot)
            ds.attrs['unit'] = self.rho_tot.unit.to_string()

            ds = perc_grp.create_dataset('rho_BH', data=self.rho_BH)
            ds.attrs['unit'] = self.rho_BH.unit.to_string()

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

            ds = perc_grp.create_dataset('numdens', data=self.numdens)
            ds.attrs['unit'] = self.numdens.unit.to_string()

            ds = perc_grp.create_dataset('mass_func', data=self.mass_func)

            ds = perc_grp.create_dataset('BH_mass', data=self.BH_mass)
            ds.attrs['unit'] = self.BH_mass.unit.to_string()

    @classmethod
    def load(cls, filename, validate=False):
        ''' load the CI from a file which was `save`d, to avoid rerunning models
        validate: check while loading that all datasets are there, error if not
        '''

        viz = cls()

        with h5py.File(filename, 'r') as file:

            viz.obs = Observations(file['metadata'].attrs['cluster'])
            viz.d = file['metadata'].attrs['d'] << u.kpc
            viz.M = file['metadata'].attrs['M']

            viz.r = file['metadata']['r'][:] << u.pc

            for key in file['percentiles']:
                value = file['percentiles'][key][:]

                try:
                    value *= u.Unit(file['percentiles'][key].attrs['unit'])
                except KeyError:
                    pass

                setattr(viz, key, value)

        return viz


class RunVisualizer(_Visualizer):
    '''All the plots based on a model run, like the chains and likelihoods
    and marginals corner plots and etc

    based on an output file I guess?
    '''

    def __str__(self):
        return f'{self.file.filename} - Run Results'

    def __init__(self, file, observations, group='mcmc'):

        # TODO this needs to be closed properly, probably
        if isinstance(file, str):
            self.file = h5py.File(file, 'r')
        else:
            self.file = file

        self._gname = group

        self.obs = observations

        self.has_indiv = 'blobs' in self.file[self._gname]
        self.has_stats = 'statistics' in self.file
        self.has_meta = 'metadata' in self.file

        # Ensure the dimensions are initialized correctly
        self.iterations = slice(None)
        self.walkers = slice(None)

    # ----------------------------------------------------------------------
    # Dimensions - Walkers
    # ----------------------------------------------------------------------

    @property
    def walkers(self):
        '''Walkers must be a slice, or a reduction method name, like "median"'''
        return self._walkers

    @walkers.setter
    def walkers(self, value):
        if not isinstance(value, slice) and value not in self._REDUC_METHODS:
            mssg = (f"`walkers` must be slice or one of "
                    f"{set(self._REDUC_METHODS)}, not {type(value)}")
            raise TypeError(mssg)

        self._walkers = value

    # ----------------------------------------------------------------------
    # Dimensions - Iterations
    # ----------------------------------------------------------------------

    # cut the ending zeroed iterations, if a run was cut short
    cut_incomplete = True

    @property
    def iterations(self):
        '''Iterations must be a slice. if cut_incomplete is True, will default
        to cutting the final empty iterations from everything
        '''
        return self._iterations

    @iterations.setter
    def iterations(self, value):
        if not isinstance(value, slice):
            mssg = f"`iteration` must be a slice, not {type(value)}"
            raise TypeError(mssg)

        if value.stop is None and self.cut_incomplete:
            stop = self.file[self._gname].attrs['iteration']
            value = slice(value.start, stop, value.step)

        self._iterations = value

    @property
    def _iteration_domain(self):

        if (start := self.iterations.start) is None:
            start = 0

        if (stop := self.iterations.stop) is None:
            stop = self.file[self._gname]['chain'].shape[0]

        step = self.iterations.step

        return np.arange(start + 1, stop + 1, step)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _get_chains(self, iterations=None, walkers=None):
        '''get the chains, properly using the iterations and walkers set or
        given, and accounting for fixed params'''

        iterations = self.iterations if iterations is None else iterations
        walkers = self.walkers if walkers is None else walkers

        labels = list(self.obs.initials)

        chain = self.file[self._gname]['chain'][iterations]

        if isinstance(walkers, slice):
            chain = chain[:, walkers]
        else:
            reduc = self._REDUC_METHODS[walkers]
            chain = reduc(chain, axis=1)

        # Hanlde fixed parameters
        if self.has_meta:

            fixed = sorted(
                ((k, v, labels.index(k)) for k, v in
                 self.file['metadata']['fixed_params'].attrs.items()),
                key=lambda item: labels.index(item[0])
            )

            for k, v, i in fixed:
                labels[i] += ' (fixed)'
                chain = np.insert(chain, i, v, axis=-1)

        return labels, chain

    def get_model(self, iterations=None, walkers=None, method='median'):
        '''
        if iterations, walkers is None, will use self.iterations, self.walkers
        '''
        # TODO there should be a method for comparing models w/ diff chain inds
        #   i.e. seeing how a model progresses over iterations

        labels, chain = self._get_chains(iterations, walkers)

        return ModelVisualizer.from_chain(chain, self.obs, method)

    def get_CImodel(self, iterations=None, walkers=None, method='median'):

        labels, chain = self._get_chains(iterations, walkers)

        return CIModelVisualizer.from_chain(chain, self.obs)

    # ----------------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------------

    def plot_chains(self, fig=None):

        labels, chain = self._get_chains()

        fig, axes = self._setup_multi_artist(fig, (len(labels), ), sharex=True)

        for ind, ax in enumerate(axes.flatten()):

            try:
                ax.plot(self._iteration_domain, chain[..., ind])
            except IndexError as err:
                mssg = 'reduced parameters, but no explanatory metadata stored'
                raise err(mssg)

            ax.set_ylabel(labels[ind])

        axes[-1].set_xlabel('Iterations')

        return fig

    def plot_indiv(self, fig=None):

        if not self.has_indiv:
            raise AttributeError("No blobs stored in file")

        probs = self.file[self._gname]['blobs'][self.iterations]

        if isinstance(self.walkers, slice):
            reduc = None
        else:
            reduc = self._REDUC_METHODS[self.walkers]

        fig, axes = self._setup_multi_artist(fig, (len(probs.dtype), ),
                                             sharex=True)

        for ind, ax in enumerate(axes.flatten()):

            label = probs.dtype.names[ind]

            indiv = probs[:][label]

            if reduc:
                indiv = reduc(indiv, axis=1)

            ax.plot(self._iteration_domain, indiv)

            ax.set_title(label)

        axes[-1].set_xlabel('Iterations')

        return fig

    def plot_marginals(self, fig=None, **corner_kw):
        import corner

        fig, ax = self._setup_multi_artist(fig, shape=None)

        labels, chain = self._get_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        # ugly
        ranges = [1. if 'fixed' not in lbl
                  else (chain[0, i] - 1, chain[0, i] + 1)
                  for i, lbl in enumerate(labels)]

        return corner.corner(chain, labels=labels, fig=fig,
                             range=ranges, plot_datapoints=False, **corner_kw)

    def plot_params(self, params, fig=None, *, colors=None, math_labels=None):
        # TODO handle colors in more plots, and handle iterator based colors

        fig, ax = self._setup_multi_artist(fig, shape=(1, len(params)))

        labels, chain = self._get_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        if colors is None:
            colors = ['b'] * len(params)

        for ind, key in enumerate(params):
            vals = chain[..., labels.index(key)]

            ax[ind].hist(vals, histtype='stepfilled', alpha=0.33,
                         color=colors[ind])  # , ec=colors[ind], lw=1.2)

            # TODO this is to make a clearer hist border, but should switch to
            #   explicitly creating a color with alpha for facecolor only
            ax[ind].hist(vals, histtype='step', color=colors[ind])

            # TODO optionally plot a line for the median

            ax[ind].set_xlabel(key if math_labels is None else math_labels[ind])

        return fig

    def plot_acceptance(self, fig=None, ax=None):

        if not self.has_stats:
            raise AttributeError("No statistics stored in file")
        else:
            stat_grp = self.file['statistics']

        fig, ax = self._setup_artist(fig, ax)

        acc = stat_grp['acceptance_rate'][self.iterations]

        if isinstance(self.walkers, slice):
            acc = acc[:, self.walkers]
        else:
            reduc = self._REDUC_METHODS[self.walkers]
            acc = reduc(acc, axis=1)

        ax.plot(self._iteration_domain, acc)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Acceptance Rate')

        return fig

    def plot_probability(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        prob = self.file[self._gname]['log_prob'][self.iterations]

        if isinstance(self.walkers, slice):
            prob = prob[:, self.walkers]
        else:
            reduc = self._REDUC_METHODS[self.walkers]
            prob = reduc(prob, axis=1)

        ax.plot(self._iteration_domain, prob)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Total Log Posterior Probability')

        return fig

    # ----------------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------------

    # TODO this very much does not work currently
    # def plot_boxplots(self, fig=None, ax=None):

    #     fig, axes = self._setup_artist(fig, ax)

    #     labels, chain = self._get_chains()

    #     chain = chain.reshape((-1, chain.shape[-1]))

    #     gridspec to hspace, wspace = 0
    #     subplot spacing to use more of grid
    #     replace bottom ticks with labels

    #     for i in range(chain.shape[-1]):
    #         axes[i].boxplot(chain[..., i])
    #         axes[i].tick_params(axis='y', direction='in', right=True)
    #         pad=-18, labelrotation=90??

    def print_summary(self, out=None, results_only=False, mathtext=False):
        '''write a summary of the run results, to a `out` file-like or stdout'''
        # TODO add more 2nd level results, like comments on BH masses, etc

        if out is None:
            out = sys.stdout

        mssg = f'{self}'
        mssg += f'\n{"=" * len(mssg)}\n'

        # RESULTS

        # median and 16, 84 percentiles of all params
        labels, chain = self._get_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        p16, p50, p84 = np.percentile(chain, [16, 50, 84], axis=0)

        uncert_minus, uncert_plus = p50 - p16, p84 - p50

        for ind, param in enumerate(labels):

            if 'fixed' in param:
                mssg += (f'{param[:-8]:>5} = {p50[ind]:.3f} ({"fixed":^14})\n')
            else:
                mssg += (f'{param:>5} = {p50[ind]:.3f} '
                         f'(+{uncert_plus[ind]:.3f}, '
                         f'-{uncert_minus[ind]:.3f})\n')

        if not results_only:

            # INFO OF RUN
            mssg += f'\nRun Metadata'
            mssg += f'\n{"=" * 12}\n'

            # number of iterations
            Niter = self.file[self._gname].attrs['iteration']
            mssg += f'Iterations = {Niter}\n'

            # dimensions ndim, nwalkers
            Ndim = self.file[self._gname].attrs['ndim']
            Nwalkers = self.file[self._gname].attrs['nwalkers']
            mssg += f'Dimensions = ({Nwalkers}, {Ndim})\n'

            # has stats? if so ... idk
            mssg += f'Has statistics = {self.has_stats}\n'

            # has metadata? if so fixed and excluded
            mssg += f'Has metadata = {self.has_meta}\n'
            if self.has_meta:
                mdata = self.file['metadata']

                mssg += 'Fixed parameters:\n'
                fixed = mdata['fixed_params'].attrs
                if fixed:
                    for k, v in fixed.items():
                        mssg += f'    {k} = {v}\n'
                else:
                    mssg += '    None\n'

                mssg += 'Excluded components:\n'
                exc = mdata['excluded_likelihoods'].attrs
                if exc:
                    for i, v in exc.items():
                        mssg += f'    ({i}) {v}\n'
                else:
                    mssg += '    None\n'

        out.write(mssg)

    # TODO We're gonna want to start viewing the distributions of some derived
    #   model parameters soon (for dist and contours and etc), which means
    #   getting the model for the entire chain, which is a monstrous task.
    #   Need to figure out the best way to do so.


# TODO a sort of "run details" method I can run on a glob pattern to print
#   a summary, so I can see which run is which, maybe should go in ./bin
