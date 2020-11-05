import numpy as np
import matplotlib.pyplot as plt


class visualizer:
    '''
    class for making, showing, saving all the plots related to a model
    idk if a class is really necessary, but sue me
    '''

    # -----------------------------------------------------------------------
    # Plotting functions
    # -----------------------------------------------------------------------

    # TODO figure out figs and axes and shared plots and stuff
    #   right now it just plots and shows stuff immediately, with no subplots

    # Pulsar max az vs measured az
    def plot_pulsar(self):
        obs_pulsar = self.obs['pulsar']

        # N_pulsars = obs_r.size
        # prob_dist = np.array([
        #     vec_Paz(self.model, A_SPACE, obs_r[i], i) for i in range(N_pulsars)
        # ])
        # max_probs = prob_dist.max(axis=1)

        # better way to get this?
        d = self.obs.priors['d']
        mod_r = pc2arcsec(self.model.r, d) / 60.

        maz = []
        for r in self.model.r:
            self.model.get_Paz(0, r, -1)
            maz.append(self.model.azmax)

        maz = np.array(maz)

        plt.plot(mod_r, maz, c='b')
        plt.plot(mod_r, -maz, c='b')

        plt.scatter(obs_pulsar['r'], self.obs['pulsar/a_los'])
        plt.show()

        # err = scipy.stats.norm.pdf(A_SPACE, 0, np.c_[obs_pulsar['Δa_los']])

        # prob_dist = likelihood_pulsars(self.model, obs_pulsar, err, True)
        # for ind in range(len(obs_pulsar['r'])):
        #     clr = f'C{ind + 1}'
        #     print(prob_dist[ind])
        #     # TODO lots of nans?
        #     plt.plot(A_SPACE, prob_dist[ind], c=clr)
        #     plt.axvline(obs_pulsar['r'][ind], c=clr)
        #     plt.axhline(obs_pulsar['a_los'][ind], c=clr)

        #     plt.show()

    # line of sight dispersion
    def plot_LOS(self):

        mass_bin = self.model.nms - 1
        vel_disp = self.obs['velocity_dispersion']
        obs_err = (vel_disp['Δσ_down'], vel_disp['Δσ_up'])

        plt.xscale("log")
        plt.xlabel("R [arcsec]")
        plt.ylabel(r"$\sigma_{los} \ [km \ s^{-1}]$")
        plt.errorbar(vel_disp['r'], vel_disp['σ'], yerr=obs_err, fmt='.')
        plt.plot(pc2arcsec(self.model.r, self.model.d),
                 np.sqrt(self.model.v2pj[mass_bin]))

        plt.show()

    # total proper motion
    def plot_pm_tot(self):

        mass_bin = self.model.nms - 1
        pm = self.obs['proper_motion']

        plt.xscale("log")
        plt.xlabel("R [arcsec]")
        plt.ylabel(r"$\sigma_{pm, total} \ [mas \ yr^{-1}]$")

        plt.errorbar(pm['r'], pm['PM_tot'],
                     xerr=pm['Δr'], yerr=pm['ΔPM_tot'], fmt='.')

        plt.plot(pc2arcsec(self.model.r, self.model.d),
                 kms2masyr(np.sqrt(self.model.v2Tj[mass_bin]), self.model.d))

        plt.show()

    # proper motion anisotropy (ratio)
    def plot_pm_ratio(self):

        mass_bin = self.model.nms - 1
        pm = self.obs['proper_motion']
        model_ratio = self.model.v2Tj[mass_bin] / self.model.v2Rj[mass_bin]

        plt.xscale("log")
        plt.xlabel("R [arcsec]")
        plt.ylabel(r"$\sigma_{pm, ratio} \ [mas \ yr^{-1}]$")

        plt.errorbar(pm['r'], pm['PM_ratio'],
                     xerr=pm['Δr'], yerr=pm['ΔPM_ratio'], fmt='.')

        plt.plot(pc2arcsec(self.model.r, self.model.d),
                 model_ratio)

        plt.show()

    # tangential proper motion
    # def plot_pm_T(self):

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
    # def plot_pm_R(self):

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

    # number density
    def plot_number_density(self):

        mass_bin = self.model.nms - 1
        numdens = self.obs['number_density']

        # interpolate number density to the observed data points r
        interp_model = np.interp(
            numdens['r'], pc2arcsec(self.model.r, self.model.d) / 60.,
            self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin]
        )

        K = (np.sum(numdens['Σ'] * interp_model / numdens["Σ"] ** 2)
             / np.sum(interp_model ** 2 / numdens["Σ"] ** 2))

        plt.loglog()
        plt.xlabel("R [arcsec]")
        plt.ylabel(r"$\Sigma \  {[arcmin}^{-2}]$")

        plt.errorbar(numdens['r'] * 60, numdens["Σ"],
                     yerr=np.sqrt(numdens["ΔΣ"]**2 + self.model.s2), fmt=".")

        plt.plot(pc2arcsec(self.model.r, self.model.d),
                 K * self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin])

        plt.show()

    # mass function
    def plot_mf_tot(self):
        pass

    def __init__(self, model, observations, nms=None, s2=None, d=None):
        self.obs = observations
        self.model = model

        # We'll expect these to either be in the model already or in the args
        if nms is not None:
            self.model.nms = nms
        if s2 is not None:
            self.model.s2 = s2
        if d is not None:
            self.model.d = d


def compare_models(*models):
    '''
    create viz objects for all the models and plot them all on the same axes
    '''

    pass
