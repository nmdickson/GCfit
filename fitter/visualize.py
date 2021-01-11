import h5py
import numpy as np
import matplotlib.pyplot as plt

from .likelihoods import pc2arcsec, kms2masyr, as2pc, create_model

# TODO add confidence intervals to plots
# TODO fix spacings


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

            else:
                fig, axarr = plt.subplots(*shape, num=fig.number, **subplot_kw)

        return fig, axarr


class ModelVisualizer(_Visualizer):
    '''
    class for making, showing, saving all the plots related to a model
    '''

    # -----------------------------------------------------------------------
    # Plotting functions
    # -----------------------------------------------------------------------

    # Pulsar max az vs measured az
    def plot_pulsar(self, fig=None, ax=None, show_obs=True):
        # TODO this is out of date with the new pulsar probability code

        fig, ax = self._setup_artist(fig, ax)

        obs_pulsar = self.obs['pulsar']
        d = self.model.d

        # N_pulsars = obs_r.size
        # prob_dist = np.array([
        #     vec_Paz(self.model, A_SPACE, obs_r[i], i)
        #     for i in range(N_pulsars)
        # ])
        # max_probs = prob_dist.max(axis=1)

        maz = []
        for r in self.model.r:
            self.model.get_Paz(0, r, -1)
            maz.append(self.model.azmax)

        maz = np.array(maz)

        ax.set_title('Pulsar LOS Acceleration')
        ax.set_xlabel('R')
        ax.set_ylabel(r'$a_{los}$')

        if show_obs:
            ax.errorbar(obs_pulsar['r'], self.obs['pulsar/a_los'],
                        yerr=self.obs['pulsar/Δa_los'], fmt='k.')

        upper_az, = ax.plot(pc2arcsec(self.model.r, d) / 60., maz)
        ax.plot(pc2arcsec(self.model.r, d) / 60., -maz, c=upper_az.get_color())

        # err = scipy.stats.norm.pdf(A_SPACE, 0, np.c_[obs_pulsar['Δa_los']])

        # prob_dist = likelihood_pulsars(self.model, obs_pulsar, err, True)
        # for ind in range(len(obs_pulsar['r'])):
        #     clr = f'C{ind + 1}'
        #     print(prob_dist[ind])
        #     # TODO lots of nans?
        #     plt.plot(A_SPACE, prob_dist[ind], c=clr)
        #     plt.axvline(obs_pulsar['r'][ind], c=clr)
        #     plt.axhline(obs_pulsar['a_los'][ind], c=clr)

        return fig

    # line of sight dispersion
    def plot_LOS(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        mass_bin = self.model.nms - 1
        vel_disp = self.obs['velocity_dispersion']
        # TODO this should be more general
        try:
            obs_err = (vel_disp['Δσ,down'], vel_disp['Δσ,up'])
        except KeyError:
            obs_err = vel_disp['Δσ']

        ax.set_title('Line of Sight Velocity')
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{los} \ [km \ s^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            ax.errorbar(vel_disp['r'], vel_disp['σ'], yerr=obs_err, fmt='k.')

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                np.sqrt(self.model.v2pj[mass_bin]))

        return fig

    # total proper motion
    def plot_pm_tot(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        # TODO change all these to check data for mass bin like in likelihoods
        mass_bin = self.model.nms - 1
        pm = self.obs['proper_motion']
        model_t2 = 0.5 * (self.model.v2Tj[mass_bin] + self.model.v2Rj[mass_bin])

        ax.set_title("Total Proper Motion")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm, total} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            ax.errorbar(pm['r'], pm['PM_tot'],
                        xerr=pm['Δr'], yerr=pm['ΔPM_tot'], fmt='k.')

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                kms2masyr(np.sqrt(model_t2), self.model.d))

        return fig

    # proper motion anisotropy (ratio)
    def plot_pm_ratio(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        mass_bin = self.model.nms - 1
        pm = self.obs['proper_motion']
        model_ratio2 = self.model.v2Tj[mass_bin] / self.model.v2Rj[mass_bin]

        ax.set_title("Proper Motion Ratio")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm,T} / \sigma_{pm,R} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            ax.errorbar(pm['r'], pm['PM_ratio'],
                        xerr=pm['Δr'], yerr=pm['ΔPM_ratio'], fmt='k.')

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                np.sqrt(model_ratio2))

        return fig

    # number density
    def plot_number_density(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        mass_bin = self.model.nms - 1
        numdens = self.obs['number_density']

        # interpolate number density to the observed data points r
        interp_model = np.interp(
            numdens['r'], pc2arcsec(self.model.r, self.model.d) / 60.,
            self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin]
        )

        K = (np.sum(numdens['Σ'] * interp_model / numdens["Σ"] ** 2)
             / np.sum(interp_model ** 2 / numdens["Σ"] ** 2))

        ax.set_title('Number Density')
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\Sigma \  {[arcmin}^{-2}]$")

        ax.loglog()

        if show_obs:
            ax.errorbar(numdens['r'] * 60, numdens["Σ"],
                        yerr=np.sqrt(numdens["ΔΣ"]**2 + self.model.s2), fmt='k.')

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                K * self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin])

        return fig

    # tangential proper motion
    def plot_pm_T(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        # TODO capture all mass bins which have data (high_mass, low_mass, etc)
        mass_bin = self.model.nms - 1
        pm = self.obs['proper_motion/high_mass']
        model_pm = self.model.v2Tj

        ax.set_title("Tangential Proper Motion [High Mass]")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm, T} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            ax.errorbar(pm['r'], pm["PM_T"],
                        xerr=pm["Δr"], yerr=pm["ΔPM_T"], fmt='k.')

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                kms2masyr(np.sqrt(model_pm[mass_bin]), self.model.d))

        return fig

    # radial proper motion
    def plot_pm_R(self, fig=None, ax=None, show_obs=True):

        fig, ax = self._setup_artist(fig, ax)

        # TODO capture all mass bins which have data (high_mass, low_mass, etc)
        mass_bin = self.model.nms - 1
        pm = self.obs['proper_motion/high_mass']
        model_pm = self.model.v2Rj

        ax.set_title("Radial Proper Motion [High Mass]")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm, R} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            ax.errorbar(pm['r'], pm["PM_R"],
                        xerr=pm["Δr"], yerr=pm["ΔPM_R"], fmt='k.')

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                kms2masyr(np.sqrt(model_pm[mass_bin]), self.model.d))

        return fig

    # mass function
    # TODO add a "mass fucntion" plot, maybe like peters, or like limepy example
    def plot_mf_tot(self, fig=None, ax=None, show_obs=True):

        import scipy.integrate as integ
        import scipy.interpolate as interp

        fig, ax = self._setup_artist(fig, ax)

        mf = self.obs['mass_function']

        scale = [10, 0.5, 0.05, 0.01]

        for annulus_ind in np.unique(mf['bin']):

            # we only want to use the obs data for this r bin
            r_mask = (mf['bin'] == annulus_ind)

            r1 = as2pc(0.4 * 60 * annulus_ind, self.model.d)
            r2 = as2pc(0.4 * 60 * (annulus_ind + 1), self.model.d)

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
                binned_N_model[mbin_ind] = (
                    integ.quad(density, r1, r2)[0]
                    / (self.model.mj[mbin_ind] * self.model.mes_widths[mbin_ind])
                )

            # Grab the N_data (adjusted by width to get an average
            #                   dr of a bin (like average-interpolating almost))
            N_data = (mf['N'][r_mask] / mf['mbin_width'][r_mask])

            # Compute δN_model from poisson error, and nuisance factor
            err = np.sqrt(mf['Δmbin'][r_mask]**2 + (self.model.F * N_data)**2)

            ax.errorbar(mf['mbin_mean'][r_mask],
                        N_data * scale[annulus_ind], yerr=err, fmt='o')

            ax.plot(self.model.mj[:self.model.nms],
                    binned_N_model * scale[annulus_ind], 'x--')

        ax.set_yscale("log")
        ax.set_xscale("log")

        return fig

    def plot_all(self):

        # TODO base this on something like determine_components probably,
        #   that is, if we only want stuff we can compare with obs, might need
        #   a 'require obs' option

        fig, axes = plt.subplots(3, 2)

        fig.suptitle("cluster name")

        self.plot_pulsar(fig=fig, ax=axes[0, 0])
        self.plot_number_density(fig=fig, ax=axes[1, 0])
        self.plot_LOS(fig=fig, ax=axes[0, 1])
        self.plot_pm_tot(fig=fig, ax=axes[1, 1])
        self.plot_pm_ratio(fig=fig, ax=axes[2, 1])

        return fig

    @classmethod
    def from_chain(cls, chain, observations, method='median'):
        '''
        create a Visualizer instance based on a chain, y taking the median
        of the chain parameters

        # TODO this supports 1-d chain arrays (theta) but not the same dicts
        '''

        reduc = cls._REDUC_METHODS[method]

        # if 3d (Niters, Nwalkers, Nparams)
        # if 2d (Nwalkers, Nparams)
        # if 1d (Nparams)
        chain = chain.reshape((-1, chain.shape[-1]))

        theta = reduc(chain, axis=0)

        return cls(create_model(theta, strict=True), observations)

    def __init__(self, model, observations):
        self.obs = observations
        self.model = model


class RunVisualizer(_Visualizer):
    '''All the plots based on a model run, like the chains and likelihoods
    and marginals corner plots and etc

    based on an output file I guess?
    '''
    # Change these to change what iters/walkers to display
    # terations must be a slice
    iterations = slice(None)

    # walkers can also be a reduc method, like 'median', to reduce to one line
    walkers = slice(None)

    @property
    def _iteration_domain(self):
        try:
            domain = np.arange(self.iterations.start,
                               self.iterations.stop,
                               self.iterations.step)
        except TypeError:
            domain = np.arange(1, self.file[self._gname].attrs['iteration'] + 1)

        return domain

    def plot_chains(self, fig=None):

        labels = tuple(self.obs.initials)

        fig, axes = self._setup_multi_artist(fig, (len(labels), ), sharex=True)

        chain = self.file[self._gname]['chain'][self.iterations]

        if isinstance(self.walkers, slice):
            chain = chain[:, self.walkers]
        else:
            reduc = self._REDUC_METHODS[self.walkers]
            chain = reduc(chain, axis=1)

        for ind, ax in enumerate(axes.flatten()):

            ax.plot(self._iteration_domain, chain[..., ind])

            ax.set_xlabel('Iterations')
            ax.set_ylabel(labels[ind])

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

        labels = tuple(self.obs.initials)

        chain = self.file[self._gname]['chain'][self.iterations]

        if isinstance(self.walkers, slice):
            chain = chain[:, self.walkers]
        else:
            reduc = self._REDUC_METHODS[self.walkers]
            chain = reduc(chain, axis=1)

        chain = chain.reshape((-1, chain.shape[-1]))

        return corner.corner(chain, labels=labels, fig=fig, **corner_kw)

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

    def __init__(self, file, observations, group='mcmc'):

        # TODO handle fixed params, when creating models from chain

        # TODO this needs to be closed properly, probably
        if isinstance(file, str):
            self.file = h5py.File(file, 'r')
        else:
            self.file = file

        self._gname = group

        self.obs = observations

        self.has_indiv = 'blobs' in self.file[self._gname]
        self.has_stats = 'statistics' in self.file

    def get_model(self, iterations=None, walkers=None, method='median'):
        '''
        if iterations, walkers is None, will use self.iterations, self.walkers
        '''

        iterations = self.iterations if iterations is None else iterations
        walkers = self.walkers if walkers is None else walkers

        chain = self.file[self._gname]['chain'][iterations]

        if isinstance(walkers, slice):
            chain = chain[:, walkers]
        else:
            reduc = self._REDUC_METHODS[walkers]
            chain = reduc(chain, axis=1)

        return ModelVisualizer.from_chain(chain, self.obs, method)


def compare_models(*models, observations, labels=None):
    '''
    # TODO probably should make this a part of RunVisualizer
    create viz objects for all the models and plot them all on the same axes
    '''

    # TODO obs data is being plotted every time unfortunately
    fig, axes = plt.subplots(3, 2)

    visuals = [ModelVisualizer(mod, observations) for mod in models]

    for ind, viz in enumerate(visuals):
        viz.plot_pulsar(fig=fig, ax=axes[0, 0])
        viz.plot_number_density(fig=fig, ax=axes[1, 0])
        viz.plot_LOS(fig=fig, ax=axes[0, 1])
        viz.plot_pm_tot(fig=fig, ax=axes[1, 1])
        viz.plot_pm_ratio(fig=fig, ax=axes[2, 1])

    if labels:
        # getting handles is a bit fragile, requires obs being plotted first
        fig.legend(plt.gca().lines[1::2], labels)

    return fig
