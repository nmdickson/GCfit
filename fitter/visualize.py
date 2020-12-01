import numpy as np
import matplotlib.pyplot as plt

from .likelihoods import pc2arcsec, kms2masyr, create_model


class Visualizer:
    '''
    class for making, showing, saving all the plots related to a model
    '''

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

    # -----------------------------------------------------------------------
    # Plotting functions
    # -----------------------------------------------------------------------

    # Pulsar max az vs measured az
    def plot_pulsar(self, fig=None, ax=None):

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
    def plot_LOS(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        mass_bin = self.model.nms - 1
        vel_disp = self.obs['velocity_dispersion']
        obs_err = (vel_disp['Δσ_down'], vel_disp['Δσ_up'])

        ax.set_title('Line of Sight Velocity')
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{los} \ [km \ s^{-1}]$")

        ax.set_xscale("log")

        ax.errorbar(vel_disp['r'], vel_disp['σ'], yerr=obs_err, fmt='k.')
        ax.plot(pc2arcsec(self.model.r, self.model.d),
                np.sqrt(self.model.v2pj[mass_bin]))

        return fig

    # total proper motion
    def plot_pm_tot(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        mass_bin = self.model.nms - 1
        pm = self.obs['proper_motion']
        model_t2 = 0.5 * (self.model.v2Tj[mass_bin] + self.model.v2Rj[mass_bin])

        ax.set_title("Total Proper Motion")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm, total} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        ax.errorbar(pm['r'], pm['PM_tot'],
                    xerr=pm['Δr'], yerr=pm['ΔPM_tot'], fmt='k.')

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                kms2masyr(np.sqrt(model_t2), self.model.d))

        return fig

    # proper motion anisotropy (ratio)
    def plot_pm_ratio(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        mass_bin = self.model.nms - 1
        pm = self.obs['proper_motion']
        model_ratio2 = self.model.v2Tj[mass_bin] / self.model.v2Rj[mass_bin]

        ax.set_title("Proper Motion Ratio")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm,T} / \sigma_{pm,R} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        ax.errorbar(pm['r'], pm['PM_ratio'],
                    xerr=pm['Δr'], yerr=pm['ΔPM_ratio'], fmt='k.')

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                np.sqrt(model_ratio2))

        return fig

    # number density
    def plot_number_density(self, fig=None, ax=None):

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

        ax.errorbar(numdens['r'] * 60, numdens["Σ"],
                    yerr=np.sqrt(numdens["ΔΣ"]**2 + self.model.s2), fmt='k.')
        ax.plot(pc2arcsec(self.model.r, self.model.d),
                K * self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin])

        # TODO what to return from these methods? fig?, ax?, both?
        return fig

    # mass function
    def plot_mf_tot(self):
        pass

    # tangential proper motion
    # def plot_pm_T(self, fig=None, ax=None):

    #     plt.xscale("log")
    #     plt.xlabel("R [arcsec]")
    #     plt.ylabel(r"$\sigma_{pm, T} \ [mas \ yr^{-1}]$")

    #     plt.errorbar(sigmat["R[arcmin]"]*60,sigmat["sigmaR[mas/yr]"],
    #                  yerr=(sigmat["eminus"],sigmat["eplus"]),fmt='.')
    #     plt.errorbar(pm["Rproj[arcsec]"],pm["PM_T[mas/yr]"], fmt='.',
    #                  yerr=pm["dPM_T[mas/yr]"],xerr=pm["dRproj[arcsec]"])
    #     plt.plot(pc2arcsec(m.r),kms2masyr(np.sqrt(m.v2Tj[nms-1])),label="nms")
    #     plt.plot(pc2arcsec(m.r),kms2masyr(np.sqrt(m.v2Tj[-2])),label=0.38)
    #     plt.legend()

    # radial proper motion
    # def plot_pm_R(self, fig=None, ax=None):

    #     plt.xscale("log")
    #     plt.xlabel("R [arcsec]")
    #     plt.ylabel(r"$\sigma_{pm, R} \ [mas \ yr^{-1}]$")

    #     plt.errorbar(sigmar["R[arcmin]"]*60,sigmar["sigmaR[mas/yr]"],
    #                  yerr=(sigmar["eminus"],sigmar["eplus"]),fmt='.')
    #     plt.errorbar(pm["Rproj[arcsec]"],pm["PM_R[mas/yr]"], fmt='.',
    #                  yerr=pm["dPM_R[mas/yr]"],xerr=pm["dRproj[arcsec]"])
    #     plt.plot(pc2arcsec(m.r),kms2masyr(np.sqrt(m.v2Rj[nms-1])),label="nms")
    #     plt.plot(pc2arcsec(m.r),kms2masyr(np.sqrt(m.v2Rj[-2])),label=0.38)
    #     plt.legend()

    def plot_all(self):

        fig, axes = plt.subplots(3, 2)

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

        if method == 'median':
            reduc = np.median
        elif method == 'mean':
            reduc = np.mean

        # if 3d (Niters, Nwalkers, Nparams)
        # if 2d (Nwalkers, Nparams)
        # if 1d (Nparams)
        chain = chain.reshape((-1, chain.shape[-1]))

        theta = reduc(chain, axis=0)

        return cls(create_model(theta, strict=True), observations)

    def __init__(self, model, observations):
        self.obs = observations
        self.model = model


def compare_models(*models, observations, labels=None):
    '''
    create viz objects for all the models and plot them all on the same axes
    '''

    # TODO data is being plotted every time unfortunately
    fig, axes = plt.subplots(3, 2)

    visuals = [Visualizer(mod, observations) for mod in models]

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


# TODO corner plot should go in here too, and probably not bother being created
#   after each run
