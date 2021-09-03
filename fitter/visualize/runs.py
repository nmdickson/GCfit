from .models import CIModelVisualizer, ModelVisualizer
from ..probabilities import priors

import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr


__all__ = ['MCMCVisualizer', 'NestedVisualizer']


class _RunVisualizer:
    '''base class for all visualizers of all run types'''

    _cmap = plt.cm.get_cmap('viridis')

    def _setup_artist(self, fig, ax, *, use_name=True):
        '''setup a plot (figure and ax) with one single ax'''

        if ax is None:
            if fig is None:
                # no figure or ax provided, make one here
                fig, ax = plt.subplots()

            else:
                # Figure provided, no ax provided. Try to grab it from the fig
                # if that doens't work, create it
                cur_axes = fig.axes

                if len(cur_axes) > 1:
                    raise ValueError(f"figure {fig} already has too many axes")

                elif len(cur_axes) == 1:
                    ax = cur_axes[0]

                else:
                    ax = fig.add_subplot()

        else:
            if fig is None:
                # ax is provided, but no figure. Grab it's figure from it
                fig = ax.get_figure()

        if hasattr(self, 'name') and use_name:
            fig.suptitle(self.name)

        return fig, ax

    def _setup_multi_artist(self, fig, shape, *, allow_blank=True,
                            use_name=True, constrained_layout=True, **sub_kw):
        '''setup a subplot with multiple axes'''

        def create_axes(base, shape, **kw):
            '''create the axes of `shape` on this base (fig)'''

            # make sure shape is a tuple of atleast 1d, at most 2d

            if not isinstance(shape, tuple):
                # TODO doesnt work on an int
                shape = tuple(shape)

            if len(shape) == 1:
                shape = (shape, 1)

            elif len(shape) > 2:
                mssg = f"Invalid `shape` for subplots {shape}, must be 2D"
                raise ValueError(mssg)

            # split into dict of nrows, ncols

            shape = dict(zip(("nrows", "ncols"), shape))

            # if either of them is also a tuple, means we want columns or rows
            #   of varying sizes, switch to using subfigures

            # TODO what are the chances stuff like `sharex` works correctly?

            if isinstance(shape['nrows'], tuple):

                subfigs = base.subfigures(ncols=shape['ncols'], nrows=1,
                                          squeeze=False)

                for ind, sf in enumerate(subfigs.flatten()):

                    try:
                        nr = shape['nrows'][ind]
                    except IndexError:

                        if allow_blank:
                            continue

                        mssg = (f"Number of row entries {shape['nrows']} must "
                                f"match number of columns ({shape['ncols']})")
                        raise ValueError(mssg)

                    sf.subplots(ncols=1, nrows=nr, **kw)

            elif isinstance(shape['ncols'], tuple):

                subfigs = base.subfigures(nrows=shape['nrows'], ncols=1,
                                          squeeze=False)

                for ind, sf in enumerate(subfigs.flatten()):

                    try:
                        nc = shape['ncols'][ind]
                    except IndexError:

                        if allow_blank:
                            continue

                        mssg = (f"Number of col entries {shape['ncols']} must "
                                f"match number of rows ({shape['nrows']})")
                        raise ValueError(mssg)

                    sf.subplots(nrows=1, ncols=nc, **kw)

            # otherwise just make a simple subplots and return that
            else:
                base.subplots(**shape, **kw)

            return base, base.axes

        # ------------------------------------------------------------------
        # Create figure, if necessary
        # ------------------------------------------------------------------

        if fig is None:
            fig = plt.figure(constrained_layout=constrained_layout)

        # ------------------------------------------------------------------
        # If no shape is provided, just return the figure, probably empty
        # ------------------------------------------------------------------

        if shape is None:
            axarr = []

        # ------------------------------------------------------------------
        # Otherwise attempt to first grab this figures axes, or create them
        # ------------------------------------------------------------------

        else:

            # this fig has axes, check that they match shape
            if axarr := fig.axes:
                if axarr.shape != shape:
                    mssg = (f"figure {fig} already contains axes with "
                            f"mismatched shape ({axarr.shape} != {shape})")
                    raise ValueError(mssg)

            else:
                fig, axarr = create_axes(fig, shape, **sub_kw)

        # ------------------------------------------------------------------
        # If desired, default to titling the figure based on it's "name"
        # ------------------------------------------------------------------

        if hasattr(self, 'name') and use_name:
            fig.suptitle(self.name)

        # ------------------------------------------------------------------
        # Ensure the axes are always returned in an array
        # ------------------------------------------------------------------

        return fig, np.atleast_1d(axarr)


class MCMCVisualizer(_RunVisualizer):
    '''All the plots based on a model run, like the chains and likelihoods
    and marginals corner plots and etc

    based on an output file I guess?
    '''

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

        self.obs = observations

        self.has_indiv = 'blobs' in self.file[self._gname]
        self.has_stats = 'statistics' in self.file
        self.has_meta = 'metadata' in self.file

        # Ensure the dimensions are initialized correctly
        self.iterations = slice(None)
        self.walkers = slice(None)

    # ----------------------------------------------------------------------
    # Dimensions
    # ----------------------------------------------------------------------

    def _reduce(self, array, *, only_iterations=False):
        '''apply the necesary iterations and walkers slicing to given `array`
        '''

        # Apply iterations cut

        array = array[self.iterations]

        # Apply walkers cut

        if not only_iterations:

            if callable(self.walkers):

                # Call on array, and ensure the dimensions still work out

                dims = array.shape

                try:
                    array = self.walkers(array, axis=1)
                except TypeError:
                    array = self.walkers(array)

                newdim = array.shape

                if not (len(dims) == len(newdim) and dims[::2] == newdim[::2]):
                    mssg = ("Invalid `walkers`, callables must operate along "
                            "only the 1st axis, or accept an `axis` keyword")
                    raise ValueError(mssg)

            else:
                # assume walkers is a slice or 1-d array
                array = array[:, self.walkers, :]

        return array

    @property
    def walkers(self):
        '''Walkers must be a slice, or a reduction method name, like "median"'''
        return self._walkers

    @walkers.setter
    def walkers(self, value):
        '''walkers must be a slice, callable to be applied to walkers axes or
        1-D boolean mask array
        '''

        if value is None or value is Ellipsis:
            value = slice(None)

        self._walkers = value

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

    def _get_chains(self):
        '''get the chains, properly using the iterations and walkers set,
        and accounting for fixed params'''

        labels = list(self.obs.initials)

        chain = self._reduce(self.file[self._gname]['chain'])

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

    # TODO method which creates a mask array for walkers based on a condition
    #   i.e. "walkers where final delta > 0.35" or something

    # ----------------------------------------------------------------------
    # Model Visualizers
    # ----------------------------------------------------------------------

    def get_model(self, method='median'):
        # TODO there should be a method for comparing models w/ diff chain inds
        #   i.e. seeing how a model progresses over iterations

        labels, chain = self._get_chains()

        return ModelVisualizer.from_chain(chain, self.obs, method)

    def get_CImodel(self, N=100, Nprocesses=1):
        import multiprocessing

        labels, chain = self._get_chains()

        with multiprocessing.Pool(processes=Nprocesses) as pool:
            return CIModelVisualizer.from_chain(chain, self.obs, N, pool=pool)

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

        probs = self.file[self._gname]['blobs']

        fig, axes = self._setup_multi_artist(fig, (len(probs.dtype), ),
                                             sharex=True)

        for ind, ax in enumerate(axes.flatten()):

            label = probs.dtype.names[ind]

            indiv = self._reduce(probs[:][label])

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

        fig, ax = self._setup_artist(fig, ax)

        acc = self._reduce(self.file['statistics']['acceptance_rate'])

        ax.plot(self._iteration_domain, acc)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Acceptance Rate')

        return fig

    def plot_probability(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        prob = self._reduce(self.file[self._gname]['log_prob'])

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

    @property
    def weights(self):
        return np.exp(self.results.logwt - self.results.logz[-1])

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

        self.results = self._get_results()

        self.has_meta = 'metadata' in self.file

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

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

        r['bound'] = self._reconstruct_bounds()

        return Results(r)

    def _reconstruct_bounds(self):
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
    def _get_chains(self, current_batch=False, include_fixed=True):
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

            if include_fixed:
                for k, v, i in fixed:
                    labels[i] += ' (fixed)'
                    chain = np.insert(chain, i, v, axis=-1)
            else:
                for *_, i in reversed(fixed):
                    del labels[i]

        return labels, chain

    def _reconstruct_priors(self):
        '''based on the stored "specified_priors" get a PriorTransform object'''

        if not self.has_meta:
            raise AttributeError("No metadata stored in file")

        stored_priors = self.file['metadata']['specified_priors'].attrs
        fixed = self.file['metadata']['fixed_params'].attrs

        prior_params = {}

        for key in list(self.obs.initials):
            try:
                type_ = stored_priors[f'{key}_type'].decode('utf-8')
                args = stored_priors[f'{key}_args']

                if args.dtype.kind == 'S':
                    args = args.astype('U')

                prior_params[key] = (type_, *args)
            except KeyError:
                continue

        prior_kwargs = {'fixed_initials': fixed, 'err_on_fail': False}
        return priors.PriorTransforms(prior_params, **prior_kwargs)

    # ----------------------------------------------------------------------
    # Model Visualizers
    # ----------------------------------------------------------------------

    def get_model(self, method='median'):

        labels, chain = self._get_chains()

        return ModelVisualizer.from_chain(chain, self.obs, method)

    def get_CImodel(self, N=100, Nprocesses=1):
        import multiprocessing

        labels, chain = self._get_chains()

        with multiprocessing.Pool(processes=Nprocesses) as pool:
            return CIModelVisualizer.from_chain(chain, self.obs, N, pool=pool)

    # ----------------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------------

    def plot_marginals(self, fig=None, **corner_kw):
        import corner

        fig, ax = self._setup_multi_artist(fig, shape=None)

        labels, chain = self._get_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        # ugly
        ranges = [1. if 'fixed' not in lbl
                  else (chain[0, i] - 1, chain[0, i] + 1)
                  for i, lbl in enumerate(labels)]

        corner_kw.setdefault('plot_datapoints', False)

        return corner.corner(chain, labels=labels, fig=fig,
                             range=ranges, **corner_kw)

    def plot_bounds(self, iteration, fig=None, show_live=False, **kw):
        from dynesty import plotting as dyplot
        from matplotlib.patches import Patch

        # TODO id rather use contours or polygons showing the bounds,
        #   rather than how dyplot does it by sampling a bunch of random points

        # TODO this doesn't seem to work the same way corner did
        # fig = self._setup_multi_artist(fig, shape=(10,10))
        # TODO real strange bug with failing on 4th ind on second function call

        priors = self._reconstruct_priors()

        clr = kw.pop('color', None)

        labels, _ = self._get_chains(include_fixed=False)

        try:
            N = len(iteration)
        except TypeError:
            N = 1
            iteration = [iteration]

        legends = []

        for ind, it in enumerate(iteration):

            if N > 1:
                clr = self._cmap((ind + 1) / N)

            if show_live:
                kw.setdefault('live_color', clr)
                kw.setdefault('live_kwargs', {'marker': 'x'})

            fig = dyplot.cornerbound(self.results, it, fig=fig, labels=labels,
                                     prior_transform=priors, color=clr,
                                     show_live=show_live, **kw)

            legends.append(Patch(facecolor=clr, label=f'Iteration {it}'))

        fig[0].legend(handles=legends)

        return fig[0]
