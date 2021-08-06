from .models import CIModelVisualizer, ModelVisualizer

import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr


__all__ = ['MCMCVisualizer']


class _RunVisualizer:
    '''base class for all visualizers of all run types'''

    _REDUC_METHODS = {'median': np.median, 'mean': np.mean}

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


class MCMCVisualizer(_RunVisualizer):
    '''All the plots based on a model run, like the chains and likelihoods
    and marginals corner plots and etc

    based on an output file I guess?
    '''
    # TODO a way to find the converged iteration automatcially (even possible?)
    # TODO a nice way to print the sources (accounting for excluded likelihoods)

    def __str__(self):
        return f'{self.file.filename} - Run Results'

    def __init__(self, file, observations, group='mcmc', name=None):

        # TODO this needs to be closed properly, probably
        if isinstance(file, h5py.File):
            self.file = file
        else:
            self.file = h5py.File(file, 'r')

        self._gname = group

        if name is not None:
            self.name = name

        # TODO could also try to get obs automatically from cluster name
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
    # TODO also support an array of indices, or like a condition.

    @property
    def walkers(self):
        '''Walkers must be a slice, or a reduction method name, like "median"'''
        return self._walkers

    @walkers.setter
    def walkers(self, value):
        # if not isinstance(value, slice) and value not in self._REDUC_METHODS:
        #     mssg = (f"`walkers` must be slice or one of "
        #             f"{set(self._REDUC_METHODS)}, not {type(value)}")
        #     raise TypeError(mssg)

        if value is None:
            self._walkers = slice(None)

        else:
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
        # TODO if using an `iterations` keyword, these checks aren't done
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

        if isinstance(walkers, str):
            reduc = self._REDUC_METHODS[walkers]
            chain = reduc(chain, axis=1)
        else:
            chain = chain[:, walkers]

        # Handle fixed parameters
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

    # ----------------------------------------------------------------------
    # Model Visualizers
    # ----------------------------------------------------------------------

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

        fig, axes = self._setup_multi_artist(fig, (len(probs.dtype), ),
                                             sharex=True)

        for ind, ax in enumerate(axes.flatten()):

            label = probs.dtype.names[ind]

            indiv = probs[:][label]

            if isinstance(self.walkers, str):
                reduc = self._REDUC_METHODS[self.walkers]
                indiv = reduc(indiv, axis=1)
            else:
                indiv = indiv[:, self.walkers]

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

    def plot_params(self, params, quants=None, fig=None, *,
                    colors=None, math_labels=None, bins=None):
        # TODO handle colors in more plots, and handle iterator based colors

        fig, ax = self._setup_multi_artist(fig, shape=(1, len(params)))

        # this shouldn't be necessary
        if len(params) == 1:
            ax = [ax]

        labels, chain = self._get_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        if colors is None:
            colors = ['b'] * len(params)

        for ind, key in enumerate(params):
            vals = chain[..., labels.index(key)]

            edgecolor = mpl_clr.to_rgb(colors[ind])
            facecolor = edgecolor + (0.33, )

            ax[ind].hist(vals, histtype='stepfilled', density=True,
                         bins=bins, ec=edgecolor, fc=facecolor, lw=2)

            if quants is not None:
                for q in np.percentile(vals, quants):
                    ax[ind].axvline(q, color=colors[ind], ls='--')
                # TODO annotate the quants on the top axis (c. mpl_ticker)
                # ax.set_xticks(np.r_[ax[ind].get_xticks()), q])

            ax[ind].set_xlabel(key if math_labels is None else math_labels[ind])

        return fig

    def plot_acceptance(self, fig=None, ax=None):

        if not self.has_stats:
            raise AttributeError("No statistics stored in file")
        else:
            stat_grp = self.file['statistics']

        fig, ax = self._setup_artist(fig, ax)

        acc = stat_grp['acceptance_rate'][self.iterations]

        if isinstance(self.walkers, str):
            reduc = self._REDUC_METHODS[self.walkers]
            acc = reduc(acc, axis=1)
        else:
            acc = acc[:, self.walkers]

        ax.plot(self._iteration_domain, acc)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Acceptance Rate')

        return fig

    def plot_probability(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        prob = self.file[self._gname]['log_prob'][self.iterations]

        if isinstance(self.walkers, str):
            reduc = self._REDUC_METHODS[self.walkers]
            prob = reduc(prob, axis=1)
        else:
            prob = prob[:, self.walkers]

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
        # Maybe set ylims ased on prior bounds? if they're not too large

        for i in range(chain.shape[-1]):

            if box:
                axes[i].boxplot(chain[..., i])

            if violin:
                axes[i].violinplot(chain[..., i])

            axes[i].set_xlabel(labels[i])

            axes[i].tick_params(axis='y', direction='in', right=True)
            # pad=-18, labelrotation=90??

    def print_summary(self, out=None, content='all'):
        '''write a summary of the run results, to a `out` file-like or stdout
        content : {'all', 'results', 'metadata'}
        '''
        # TODO add more 2nd level results, like comments on BH masses, etc

        if out is None:
            out = sys.stdout

        mssg = f'{self}'
        mssg += f'\n{"=" * len(mssg)}\n'

        # RESULTS

        # organize this more like it is in cum_mass plots
        if content == 'all' or content == 'results':

            # median and 16, 84 percentiles of all params
            labels, chain = self._get_chains()

            chain = chain.reshape((-1, chain.shape[-1]))

            p16, p50, p84 = np.percentile(chain, [16, 50, 84], axis=0)

            uncert_minus, uncert_plus = p50 - p16, p84 - p50

            for ind, param in enumerate(labels):

                if 'fixed' in param:
                    mssg += (f'{param[:-8]:>5} = {p50[ind]:.3f} '
                             f'({"fixed":^14})\n')
                else:
                    mssg += (f'{param:>5} = {p50[ind]:.3f} '
                             f'(+{uncert_plus[ind]:.3f}, '
                             f'-{uncert_minus[ind]:.3f})\n')

        if content == 'all' or content == 'metadata':

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

                # TODO add specified bounds/priors
                # mssg += 'Specified prior bounds'

        out.write(mssg)


class NestedVisualizer(_RunVisualizer):

    def __init__(self, file, observations, group='nested', name=None):

        # TODO this needs to be closed properly, probably
        if isinstance(file, h5py.File):
            self.file = file
        else:
            self.file = h5py.File(file, 'r')

        self._gname = group

        if name is not None:
            self.name = name

        # TODO could also try to get obs automatically from cluster name
        self.obs = observations

        self.has_meta = 'metadata' in self.file

    def _get_results(self, finite_only=False):
        '''return a dynesty-style `Results` class'''
        from dynesty.results import Results

        res = self.file[self._gname]

        if finite_only:
            inds = res['logl'][:] > -1e300
        else:
            inds = slice(None)

        r = {}

        for k, d in res.items():

            if k in ('current_batch', 'bound'):
                continue

            if d.shape and (d.shape[0] == res['logl'].shape[0]):
                d = np.array(d)[inds]
            else:
                d = np.array(d)

            r[k] = d

        if finite_only:
            # remove the amount of non-finite values we removed from niter
            r['niter'] -= (r['niter'] - r['logl'].size)

        r['bound'] = self._get_bounds()

        return Results(r)

    def _get_bounds(self):
        '''
        based on the bound info stored in file, get actual dynesty bound objects
        '''
        from dynesty import bounding

        res = self.file['nested']
        bnd_grp = res['bound']

        bnds = []
        for i in range(len(bnd_grp)):

            ds = bnd_grp[str(i)]
            btype = ds.attrs['type']

            if btype == 'UnitCube':
                bnds.append(bounding.UnitCube(ds.attrs['ndim']))

            elif btype == 'MultiEllipsoid':
                ctrs = ds['centres'][:]
                covs = ds['covariances'][:]
                bnds.append(bounding.MultiEllipsoid(ctrs=ctrs, covs=covs))

            elif btype == 'RadFriends':
                cov = ds['covariances'][:]
                ndim = ds.attrs['ndim']
                bnds.append(bounding.RadFriends(ndim=ndim, cov=cov))

            elif btype == 'SupFriends':
                cov = ds['covariances'][:]
                ndim = ds.attrs['ndim']
                bnds.append(bounding.SupFriends(ndim=ndim, cov=cov))

            else:
                raise RuntimeError('unrecognized type ', btype)

        return bnds

    # TODO how we handle current_batch stuff will probably need to be sorted out
    def _get_chains(self, current_batch=True):
        '''for nested sampling results (current Batch)'''

        if current_batch:
            chain = self.file[self._gname]['current_batch']['vstar'][:]

        else:
            chain = self.file[self._gname]['samples'][:]

        labels = list(self.obs.initials)

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
