from .models import _Visualizer, CIModelVisualizer, ModelVisualizer

import sys

import h5py
import numpy as np


__all__ = ['RunVisualizer']


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

    def get_CImodel(self, N=100, iterations=None, walkers=None):

        labels, chain = self._get_chains(iterations, walkers)

        return CIModelVisualizer.from_chain(chain, self.obs, N)

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

    # TODO this is missing alot of formatting needs
    def plot_summary(self, fig=None, *, box=True, violin=True):

        if not (box or violin):
            raise ValueError("Must plot atleast one of `box` or `violin`")

        labels, chain = self._get_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        fig, axes = self._setup_multi_artist(fig, shape=(1, chain.shape[-1]))

        # gridspec to hspace, wspace = 0
        # subplot spacing to use more of grid
        # replace bottom ticks with labels
        # Maybe set ylims ased on prior bounds? if they're not too large

        for i in range(chain.shape[-1]):

            if box:
                axes[i].boxplot(chain[..., i])

            if violin:
                axes[i].violinplot(chain[..., i])

            axes[i].tick_params(axis='y', direction='in', right=True)
            # pad=-18, labelrotation=90??

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


# TODO a sort of "run details" method I can run on a glob pattern to print
#   a summary, so I can see which run is which, maybe should go in ./bin