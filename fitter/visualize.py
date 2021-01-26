import sys

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

    # -----------------------------------------------------------------------
    # Plotting functions
    # -----------------------------------------------------------------------

    # TODO a better term than 'attempt' should be used
    # TODO change all these to check data for mass bin like in likelihoods
    # TODO the generation of obs_err should be generalized in all functions

    # Pulsar max az vs measured az
    def plot_pulsar(self, fig=None, ax=None, show_obs=True):
        # TODO this is out of date with the new pulsar probability code

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Pulsar LOS Acceleration')
        ax.set_xlabel('R')
        ax.set_ylabel(r'$a_{los}$')

        d = self.model.d

        maz = []
        for r in self.model.r:
            self.model.get_Paz(0, r, -1)
            maz.append(self.model.azmax)

        maz = np.array(maz)

        if show_obs:
            try:
                obs_pulsar = self.obs['pulsar']

                ax.errorbar(obs_pulsar['r'], self.obs['pulsar/a_los'],
                            yerr=self.obs['pulsar/Δa_los'], fmt='k.')

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        upper_az, = ax.plot(pc2arcsec(self.model.r, d) / 60., maz)
        ax.plot(pc2arcsec(self.model.r, d) / 60., -maz, c=upper_az.get_color())

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

        return fig

    # line of sight dispersion
    def plot_LOS(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Line of Sight Velocity')
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{los} \ [km \ s^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            try:
                veldisp = self.obs['velocity_dispersion']

                try:
                    obs_err = (veldisp['Δσ,down'], veldisp['Δσ,up'])
                except KeyError:
                    obs_err = veldisp['Δσ']

                ax.errorbar(veldisp['r'], veldisp['σ'], yerr=obs_err, fmt='k.')

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                np.sqrt(self.model.v2pj[mass_bin]))

        return fig

    # total proper motion
    def plot_pm_tot(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Total Proper Motion")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm, total} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            try:
                pm = self.obs['proper_motion']

                ax.errorbar(pm['r'], pm['PM_tot'],
                            xerr=pm['Δr'], yerr=pm['ΔPM_tot'], fmt='k.')

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_t2 = 0.5 * (self.model.v2Tj[mass_bin] + self.model.v2Rj[mass_bin])

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                kms2masyr(np.sqrt(model_t2), self.model.d))

        return fig

    # proper motion anisotropy (ratio)
    def plot_pm_ratio(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Proper Motion Ratio")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm,T} / \sigma_{pm,R} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            try:

                pm = self.obs['proper_motion']

                ax.errorbar(pm['r'], pm['PM_ratio'],
                            xerr=pm['Δr'], yerr=pm['ΔPM_ratio'], fmt='k.')

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_ratio2 = self.model.v2Tj[mass_bin] / self.model.v2Rj[mass_bin]

        ax.plot(pc2arcsec(self.model.r, self.model.d), np.sqrt(model_ratio2))

        return fig

    # tangential proper motion
    def plot_pm_T(self, fig=None, ax=None, show_obs=True):
        # TODO capture all mass bins which have data (high_mass, low_mass, etc)

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Tangential Proper Motion [High Mass]")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm, T} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            try:

                pm = self.obs['proper_motion/high_mass']

                ax.errorbar(pm['r'], pm["PM_T"],
                            xerr=pm["Δr"], yerr=pm["ΔPM_T"], fmt='k.')

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                kms2masyr(np.sqrt(self.model.v2Tj[mass_bin]), self.model.d))

        return fig

    # radial proper motion
    def plot_pm_R(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title("Radial Proper Motion [High Mass]")
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\sigma_{pm, R} \ [mas \ yr^{-1}]$")

        ax.set_xscale("log")

        if show_obs:
            try:

                pm = self.obs['proper_motion/high_mass']

                ax.errorbar(pm['r'], pm["PM_R"],
                            xerr=pm["Δr"], yerr=pm["ΔPM_R"], fmt='k.')

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        model_pm = self.model.v2Rj

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                kms2masyr(np.sqrt(model_pm[mass_bin]), self.model.d))

        return fig

    # number density
    def plot_number_density(self, fig=None, ax=None, show_obs=True):

        mass_bin = self.model.nms - 1

        fig, ax = self._setup_artist(fig, ax)

        ax.set_title('Number Density')
        ax.set_xlabel("R [arcsec]")
        ax.set_ylabel(r"$\Sigma \  {[arcmin}^{-2}]$")

        ax.loglog()

        # numdens is a bit different cause we want to compute K whenever
        #   possible, even if we're not showing obs

        try:
            numdens = self.obs['number_density']

            # interpolate number density to the observed data points r
            interp_model = np.interp(
                numdens['r'], pc2arcsec(self.model.r, self.model.d) / 60.,
                self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin]
            )

            K = (np.sum(numdens['Σ'] * interp_model / numdens["Σ"] ** 2)
                 / np.sum(interp_model ** 2 / numdens["Σ"] ** 2))

        except KeyError:
            K = 1.

        if show_obs:
            try:

                numdens = self.obs['number_density']

                ax.errorbar(numdens['r'] * 60, numdens["Σ"], fmt='k.',
                            yerr=np.sqrt(numdens["ΔΣ"]**2 + self.model.s2))

            except KeyError as err:
                if show_obs != 'attempt':
                    raise err

        ax.plot(pc2arcsec(self.model.r, self.model.d),
                K * self.model.Sigmaj[mass_bin] / self.model.mj[mass_bin])

        return fig

    # mass function
    # TODO add a "mass fucntion" plot, maybe like peters, or like limepy example
    def plot_mf_tot(self, fig=None, ax=None, show_obs=True):

        import scipy.integrate as integ
        import scipy.interpolate as interp

        fig, ax = self._setup_artist(fig, ax)

        mf = self.obs['mass_function']

        scale = [1000., 10., 0.1, 0.001]

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

            # TODO in Peters notebooks this is actually computed differently
            # Compute δN_model from poisson error, and nuisance factor
            err = np.sqrt(mf['Δmbin'][r_mask]**2 + (self.model.F * N_data)**2)

            ax.errorbar(mf['mbin_mean'][r_mask],
                        N_data * scale[annulus_ind], yerr=err, fmt='o')

            ax.plot(self.model.mj[:self.model.nms],
                    binned_N_model * scale[annulus_ind], 'x--')

        ax.set_yscale("log")
        ax.set_xscale("log")

        return fig

    # def plot_imf
    # def plot_bhcontent

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

        return cls(create_model(theta, observations), observations)

    def __init__(self, model, observations):
        self.obs = observations
        self.model = model


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


def compare_runs(output_files, observations):

    RV_list = [RunVisualizer(file, observations) for file in output_files]
    MV_list = [RV.get_model() for RV in RV_list]

    # compare run stats, chains, etc

    # plot median chains for all runs in same figure

    # fig1 = plt.figure()
    fig1 = None

    # plot indivs with full walkers for all runs in separate columns of same fig
    # TODO having trouble because you cant really comine figs in plt
    fig2 = plt.figure()
    # nrows=max number of likelihoods in the RVs, ncols=len(RV_list)

    for ind, RV in enumerate(RV_list):
        RV.walkers = 'median'
        fig1 = RV.plot_chains(fig1)

        # RV.walkers = slice(None)
        # indiv_axes = RV.plot_indiv().axes

        # fig2.add_subplot()

   # compare model outputs

    # do a plot all, with obs only being plots once, but all runs in same fig 

    # fig3 = plt.figure()
    fig3 = None

    for ind, MV in enumerate(MV_list):
        fig3 = MV.plot_all(fig3, show_obs=(not ind))
