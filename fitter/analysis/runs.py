from .. import Observations
from ..probabilities import priors
from .models import CIModelVisualizer, ModelVisualizer, ModelCollection

import sys
import pathlib
import logging
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr


__all__ = ['RunCollection', 'MCMCRun', 'NestedRun']


# --------------------------------------------------------------------------
# Individual Run Analysis
# --------------------------------------------------------------------------


# TODO a way to plot our priors, probably for both vizs
class _RunAnalysis:
    '''base class for all visualizers of all run types'''

    _cmap = plt.cm.get_cmap('viridis')

    def __str__(self):
        try:
            return f'{self.file.filename} - Run Results'
        except AttributeError:
            return "Run Results"

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
                            use_name=True, constrained_layout=True,
                            subfig_kw=None, **sub_kw):
        '''setup a subplot with multiple axes'''

        if subfig_kw is None:
            subfig_kw = {}

        def create_axes(base, shape):
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
                                          squeeze=False, **subfig_kw)

                for ind, sf in enumerate(subfigs.flatten()):

                    try:
                        nr = shape['nrows'][ind]
                    except IndexError:

                        if allow_blank:
                            continue

                        mssg = (f"Number of row entries {shape['nrows']} must "
                                f"match number of columns ({shape['ncols']})")
                        raise ValueError(mssg)

                    sf.subplots(ncols=1, nrows=nr, **sub_kw)

            elif isinstance(shape['ncols'], tuple):

                subfigs = base.subfigures(nrows=shape['nrows'], ncols=1,
                                          squeeze=False, **subfig_kw)

                for ind, sf in enumerate(subfigs.flatten()):

                    try:
                        nc = shape['ncols'][ind]
                    except IndexError:

                        if allow_blank:
                            continue

                        mssg = (f"Number of col entries {shape['ncols']} must "
                                f"match number of rows ({shape['nrows']})")
                        raise ValueError(mssg)

                    sf.subplots(nrows=1, ncols=nc, **sub_kw)

            # otherwise just make a simple subplots and return that
            else:
                base.subplots(**shape, **sub_kw)

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
                # TODO this won't actually work, cause fig.axes is just a list
                if axarr.shape != shape:
                    mssg = (f"figure {fig} already contains axes with "
                            f"mismatched shape ({axarr.shape} != {shape})")
                    raise ValueError(mssg)

            else:
                fig, axarr = create_axes(fig, shape)

        # ------------------------------------------------------------------
        # If desired, default to titling the figure based on it's "name"
        # ------------------------------------------------------------------

        if hasattr(self, 'name') and use_name:
            fig.suptitle(self.name)

        # ------------------------------------------------------------------
        # Ensure the axes are always returned in an array
        # ------------------------------------------------------------------

        return fig, np.atleast_1d(axarr)


class MCMCRun(_RunAnalysis):
    '''All the plots based on a model run, like the chains and likelihoods
    and marginals corner plots and etc

    based on an output file I guess?
    '''

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
        return priors.Priors(prior_params, **prior_kwargs)

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

        # TODO maybe make this match Nested's `plot_params` more

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

        # TODO make the names of plots match more between MCMC and nested

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


class NestedRun(_RunAnalysis):

    @property
    def weights(self):

        from dynesty.dynamicsampler import weight_function

        # If maxfrac is added as arg, make sure to add here as well
        if self.has_meta:
            stop_kw = {'pfrac': self.file['metadata'].attrs['pfrac']}
        else:
            stop_kw = {}

        return weight_function(self.results, stop_kw, return_weights=True)[1][2]

    @property
    def ESS(self):
        '''effective sample size'''
        from scipy.special import logsumexp
        logwts = self.results.logwt
        logneff = logsumexp(logwts) * 2 - logsumexp(logwts * 2)
        return np.exp(logneff)

    @property
    def _resampled_weights(self):
        from scipy.stats import gaussian_kde
        from dynesty.utils import resample_equal

        # "resample" logvols so they all have equal weights
        eq_logvol = resample_equal(-self.results.logvol, self.weights)

        # Compute the KDE of resampled logvols and evaluate on normal logvols
        return gaussian_kde(eq_logvol)(-self.results.logvol)

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
        self.has_stats = 'statistics' in self.file

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

            if k in ('current_batch', 'initial_batch', 'bound'):
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
                raise RuntimeError('unrecognized bound type ', btype)

        return bnds

    # TODO some ways of handling and plotting initial_batch only clusters
    def _get_chains(self, include_fixed=True):
        '''for nested sampling results (current Batch)'''

        try:
            chain = self.file[self._gname]['samples'][:]
        except KeyError as err:
            mssg = f'{err.args[0]}. This run may not yet have converged'
            raise KeyError(mssg)

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

    def _get_equal_weight_chains(self, include_fixed=True, add_errors=False):
        from dynesty.utils import resample_equal

        if add_errors is False:
            chain = self.file[self._gname]['samples'][:]
            eq_chain = resample_equal(chain, self.weights)

        else:
            from dynesty.dynamicsampler import weight_function
            sim_run = self._sim_errors(1)[0]
            sim_wt = weight_function(sim_run, {'pfrac': 1.}, True)[1][2]
            eq_chain = resample_equal(sim_run.samples, sim_wt)

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
                    eq_chain = np.insert(eq_chain, i, v, axis=-1)
            else:
                for *_, i in reversed(fixed):
                    del labels[i]

        return labels, eq_chain

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

    def get_model(self, method='mean', add_errors=False):

        if method == 'mean':
            theta = self.parameter_means()[0]
            return ModelVisualizer.from_theta(theta, self.obs)

        else:
            labels, chain = self._get_equal_weight_chains(add_errors=add_errors)
            return ModelVisualizer.from_chain(chain, self.obs, method)

    def get_CImodel(self, N=100, Nprocesses=1, add_errors=False, shuffle=True):
        import multiprocessing

        labels, chain = self._get_equal_weight_chains(add_errors=add_errors)

        if shuffle:
            np.random.default_rng().shuffle(chain, axis=0)

        with multiprocessing.Pool(processes=Nprocesses) as pool:
            return CIModelVisualizer.from_chain(chain, self.obs, N, pool=pool)

    # ----------------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------------

    def plot_marginals(self, fig=None, full_volume=False, **corner_kw):
        import corner
        # TODO the formatting of this is still ugly, check out dyplot's version

        fig, ax = self._setup_multi_artist(fig, shape=None)

        if full_volume:
            labels, chain = self._get_chains()
        else:
            labels, chain = self._get_equal_weight_chains()

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

        # TODO doesnt work for some bound types (like balls)

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

    def plot_weights(self, fig=None, ax=None, show_bounds=False,
                     resampled=False, filled=False, **kw):

        fig, ax = self._setup_artist(fig, ax)

        wts = self._resampled_weights if resampled else self.weights

        line, = ax.plot(-self.results.logvol, wts, **kw)

        if filled:
            color = mpl_clr.to_rgb(line.get_color())
            facecolor = color + (0.33, )

            ax.fill_between(-self.results.logvol, 0, wts,
                            color=color, fc=facecolor)

        if show_bounds:
            if self.has_meta:
                maxfrac = self.file['metadata'].attrs['maxfrac']

            else:
                maxfrac = 0.8

                mssg = "No metadata stored in file, `maxfrac` defaults to 80%"
                warnings.warn(mssg)

            ax.axhline(maxfrac * max(wts), c='g')

        ax.set_ylabel('weights')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_probability(self, fig=None, ax=None, **kw):

        fig, ax = self._setup_artist(fig, ax)

        finite = self.results.logl > -1e300

        ax.plot(-self.results.logvol[finite], self.results.logl[finite], **kw)

        ax.set_ylabel('Total Log Likelihood')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_evidence(self, fig=None, ax=None, error=False, **kw):

        fig, ax = self._setup_artist(fig, ax)

        finite = self.results.logz > -1e300

        logvol = self.results.logvol[finite]
        logz = self.results.logz[finite]

        line, = ax.plot(-logvol, logz, **kw)

        if error:
            err_up = logz + self.results.logzerr[finite]
            err_down = logz - self.results.logzerr[finite]

            ax.fill_between(-logvol, err_up, err_down,
                            color=line.get_color(), alpha=0.5)

        ax.set_ylabel(r'Estimated Evidence $\log(Z)$')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_H(self, fig=None, ax=None, **kw):

        fig, ax = self._setup_artist(fig, ax)

        finite = self.results.information > -1e250

        logvol = self.results.logvol[finite]

        ax.plot(-logvol, self.results.information[finite], **kw)

        ax.set_ylabel(r'Information $H \equiv \int_{\Omega_{\Theta}} '
                      r'P(\Theta)\ln\frac{P(\Theta)}{\pi(\Theta)} \,d\Theta$')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_HN(self, fig=None, ax=None, **kw):

        fig, ax = self._setup_artist(fig, ax)

        finite = self.results.information > -1e250

        HN = self.results.information * self.results.samples_n

        ax.plot(HN[finite], **kw)

        x = np.arange(0, HN[finite].size)
        ax.plot(x, c='k', alpha=0.15)

        ax.set_ylabel(r'HN')
        ax.set_xlabel('Iteration')

        return fig

    def plot_nlive(self, fig=None, ax=None, **kw):

        fig, ax = self._setup_artist(fig, ax)

        ax.plot(-self.results.logvol, self.results.samples_n, **kw)

        ax.set_ylabel(r'Number of live points')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_ncall(self, fig=None, ax=None, **kw):

        fig, ax = self._setup_artist(fig, ax)

        kw.setdefault('where', 'mid')

        ax.step(-self.results.logvol, self.results.ncall, **kw)

        ax.set_ylabel(r'Number of likelihood calls')
        ax.set_xlabel(r'$-\ln(X)$')

        return fig

    def plot_KL_divergence(self, fig=None, ax=None, Nruns=100,
                           kl_kwargs=None, **kw):
        from dynesty.utils import kld_error

        fig, ax = self._setup_artist(fig, ax)

        if kl_kwargs is None:
            kl_kwargs = {}

        kw.setdefault('color', 'b')
        kw.setdefault('alpha', 0.25)

        for _ in range(Nruns):

            KL = kld_error(self.results, **kl_kwargs)

            ax.plot(KL, **kw)

        ax.set_ylabel('KL Divergence')
        ax.set_xlabel('Iterations')

        return fig

    def plot_params(self, fig=None, params=None, *,
                    posterior_color='tab:blue', posterior_border=True,
                    show_weight=True, fill_type='weights', ylims=None,
                    truths=None, **kw):

        from scipy.stats import gaussian_kde
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # ------------------------------------------------------------------
        # Setup plotting kwarg defaults
        # ------------------------------------------------------------------

        color = mpl_clr.to_rgb(posterior_color)
        facecolor = color + (0.33, )

        kw.setdefault('marker', '.')

        # ------------------------------------------------------------------
        # Determine which property will define the color-scale of the samples
        # ------------------------------------------------------------------

        if fill_type in ('weights', 'weight', 'wts', 'wt', 'logwt'):
            c = self._resampled_weights

        elif fill_type in ('iterations', 'iters', 'samples_it'):
            c = self.results.samples_it

        elif fill_type in ('id', 'samples_id'):
            c = self.results.samples_id

        elif fill_type in ('batch', 'samples_batch'):
            # TODO when showing batches, make the initial sample distinguishable
            c = self.results.samples_batch

        elif fill_type in ('bound', 'samples_bound'):
            c = self.results.samples_bound

        else:
            mssg = ('Invalid fill type, must be one of '
                    '{weights, iters, id, batch, bound}')
            raise ValueError(mssg)

        # ------------------------------------------------------------------
        # Get the sample chains (weighted and unweighted), paring down the
        # chains to only the desired params, if provided
        # ------------------------------------------------------------------

        labels, chain = self._get_chains()
        eq_chain = self._get_equal_weight_chains()[1]

        # params is None or a list of string labels
        if params is not None:
            prm_inds = [labels.index(p) for p in params]

            labels = params
            chain, eq_chain = chain[..., prm_inds], eq_chain[..., prm_inds]

        # ------------------------------------------------------------------
        # Setup the truth values and confidence intervals
        # ------------------------------------------------------------------

        if truths is not None and truths.ndim == 2:
            # Assume confidence bounds rather than single truth value

            truth_ci = truths[:, 1:]
            truths = truths[:, 0]

        else:
            truth_ci = None

        # ------------------------------------------------------------------
        # Setup axes
        # ------------------------------------------------------------------

        if ylims is None:
            ylims = [(None, None)] * len(labels)

        elif len(ylims) != len(labels):
            mssg = "`ylims` must match number of params"
            raise ValueError(mssg)

        gs_kw = {}

        if (shape := len(labels) + show_weight) > 5 + show_weight:
            shape = (int(np.ceil(shape / 2)) + show_weight, 2)

            if show_weight:
                gs_kw = {"height_ratios": [0.5] + [1] * (shape[0] - 1)}

        fig, axes = self._setup_multi_artist(fig, shape, sharex=True,
                                             gridspec_kw=gs_kw)

        axes = axes.reshape(shape)

        for ax in axes[-1]:
            ax.set_xlabel(r'$-\ln(X)$')

        # ------------------------------------------------------------------
        # If showing weights explicitly, format the ax and use the
        # `plot_weights` method
        # ------------------------------------------------------------------

        if show_weight:
            for ax in axes[0]:
                # plot weights above scatter plots
                # TODO figure out what colors to use
                self.plot_weights(fig=fig, ax=ax, resampled=True, filled=True,
                                  color=self._cmap(np.inf))

                ax.set_xticklabels([])
                ax.set_xlabel(None)
                ax.set_yticklabels([])
                ax.set_ylabel(None)

                # Theres probably a cleaner way to do this
                divider = make_axes_locatable(ax)
                spacer = divider.append_axes('right', size="25%", pad=0)
                spacer.set_visible(False)

        # ------------------------------------------------------------------
        # Plot each parameter
        # ------------------------------------------------------------------

        for ind, ax in enumerate(axes[1:].flatten()):

            # --------------------------------------------------------------
            # Get the relevant samples.
            # If necessary, remove any unneeded axes
            # (should be handled by above todo)
            # --------------------------------------------------------------

            try:
                prm, eq_prm = chain[:, ind], eq_chain[:, ind]
                lbl = labels[ind]
            except IndexError:
                # If theres an odd number of (>5) params need to delete last one
                # TODO preferably this would also resize this column of plots
                ax.remove()
                continue

            # --------------------------------------------------------------
            # Divide the ax to accomodate the posterior plot on the right
            # --------------------------------------------------------------

            divider = make_axes_locatable(ax)
            post_ax = divider.append_axes('right', size="25%", pad=0, sharey=ax)

            post_ax.set_xticks([])

            # --------------------------------------------------------------
            # Plot the samples with respect to ln(X)
            # --------------------------------------------------------------

            # TODO the y tick values have disappeared should be on the last axis
            ax.scatter(-self.results.logvol, prm, c=c, cmap=self._cmap, **kw)

            ax.set_ylabel(lbl)
            ax.set_xlim(left=0)

            # --------------------------------------------------------------
            # Plot the posterior distribution (accounting for weights)
            # --------------------------------------------------------------

            kde = gaussian_kde(eq_prm)

            y = np.linspace(eq_prm.min(), eq_prm.max(), 500)

            post_ax.fill_betweenx(y, 0, kde(y), color=color, fc=facecolor)

            if truths is not None:
                post_ax.axhline(truths[ind], c='tab:red')

                if truth_ci is not None:
                    post_ax.axhspan(*truth_ci[ind], color='tab:red', alpha=0.33)

            if not posterior_border:
                post_ax.axis('off')

            # TODO maybe put ticks on right side as well?
            for tk in post_ax.get_yticklabels():
                tk.set_visible(False)

            post_ax.set_xlim(left=0)

            ax.set_ylim(ylims[ind])

        return fig

    # ----------------------------------------------------------------------
    # Parameter estimation
    # ----------------------------------------------------------------------

    def _sim_errors(self, Nruns=250):
        '''add the statistical and sampling errors not normally accounted for
        by using the built-in `resample_run` function

        returns list `Nruns` results
        '''
        from dynesty.utils import resample_run

        return [resample_run(self.results) for _ in range(Nruns)]

    def parameter_means(self, Nruns=250, sim_runs=None, return_samples=True):
        '''
        return the means of each parameter, and the corresponding error on that
        mean
        errors come from the two main sources of error present in nested
        sampling and are computed using the standard deviation of the mean
        from `Nruns` simulated (resampled and jittered) runs of this sampling
        run. See https://dynesty.readthedocs.io/en/latest/errors.html for more
        '''
        from dynesty.utils import mean_and_cov

        if sim_runs is None:
            sim_runs = self._sim_errors(Nruns)

        means = []
        for res in sim_runs:
            wt = np.exp(res.logwt - res.logz[-1])
            means.append(mean_and_cov(res.samples, wt)[0])

        mean = np.mean(means, axis=0)
        err = np.std(means, axis=0)

        if return_samples:
            return mean, err, np.array(means)
        else:
            return mean, err

    def parameter_vars(self, Nruns=250, sim_runs=None, return_samples=True):
        '''
        return the variance of each parameter, and the corresponding error on
        that variance
        See `parameter_means` for more
        '''
        from dynesty.utils import mean_and_cov

        if sim_runs is None:
            sim_runs = self._sim_errors(Nruns)

        vars_ = []
        for res in sim_runs:
            wt = np.exp(res.logwt - res.logz[-1])
            vars_.append(mean_and_cov(res.samples, wt)[1])

        mean = np.mean(vars_, axis=0)
        err = np.std(vars_, axis=0)

        if return_samples:
            return mean, err, np.array(vars_)
        else:
            return mean, err

    # ----------------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------------

    def print_summary(self, out=None, content='all', *, N_simruns=50):
        '''write a summary of the run results, to a `out` file-like or stdout
        content : {'all', 'results', 'metadata'}
        '''
        # TODO add more 2nd level results, like comments on BH masses, etc

        if out is None:
            out = sys.stdout

        mssg = f'{self}'
        mssg += f'\n{"=" * len(mssg)}\n'

        # RESULTS

        # organize this arg (content) more like it is in cum_mass plots
        if content == 'all' or content == 'results':

            sr = self._sim_errors(N_simruns)

            mns, σ_mns = self.parameter_means(sim_runs=sr, return_samples=False)
            vrs, σ_vrs = self.parameter_vars(sim_runs=sr, return_samples=False)
            std, σ_std = np.sqrt(np.diag(vrs)), np.sqrt(np.diag(σ_vrs))

            # median and 16, 84 percentiles of all params
            labels, _ = self._get_chains()

            mssg += f'{" " * 8}{"Mean":^14} | {"Std. Dev.":^14}\n'

            for ind, param in enumerate(labels):

                if 'fixed' in param:
                    mssg += (f'{param[:-8]:>5} = {mns[ind]:.3f} '
                             f'({"fixed":^14}) | ')
                    mssg += f'{"-" * 14}\n'
                else:
                    mssg += (f'{param:>5} = {mns[ind]:.3f} '
                             f'(±{σ_mns[ind]:.3f}) | ')
                    mssg += (f'{std[ind]:.3f} (±{σ_std[ind]:.3f})\n')

        if content == 'all' or content == 'metadata':

            # INFO OF RUN
            mssg += f'\nRun Metadata'
            mssg += f'\n{"=" * 12}\n'

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


# --------------------------------------------------------------------------
# Collections of Runs
# --------------------------------------------------------------------------


class _Annotator:
    '''Annotate points on click with run/cluster names
    picker=True must be set on the actual plot
    '''

    arrow = {'arrowstyle': "-", "connectionstyle": "arc3", "color": "black"}

    def __init__(self, fig, runs, xdata, ydata):
        self.fig = fig
        self.runs, self.xdata, self.ydata = runs, xdata, ydata

        # TODO this size needs to be somewhat automatic, maybe based on errs
        self.delx = ((np.nanmax(self.xdata) - np.nanmin(self.xdata)) / 10)
        self.dely = ((np.nanmax(self.ydata) - np.nanmin(self.ydata)) / 10)

        self.fig.canvas.mpl_connect('pick_event', self)

        self.cur_annot = None
        self.cur_ind = None

    def __call__(self, event):
        ind = event.ind[0]
        ax = event.artist.axes

        xy = (self.xdata[ind], self.ydata[ind])

        xtext = self.xdata[ind] + self.delx
        ytext = self.ydata[ind] + self.dely

        if ytext > np.nanmax(self.ydata):
            ytext = np.nanmax(self.ydata)

        # cluster = self.runs[ind]
        cluster = self.runs[ind].name

        # clear old annotation
        if self.cur_annot is not None:
            self.cur_annot.remove()

        if ind == self.cur_ind:
            self.cur_ind = None
            self.cur_annot = None

        else:
            self.cur_ind = ind
            self.cur_annot = ax.annotate(cluster, xy, (xtext, ytext),
                                         arrowprops=self.arrow)

        self.fig.canvas.draw()


class RunCollection(_RunAnalysis):
    '''For analyzing a collection of runs all at once
    '''

    def __str__(self):
        return "Collection of Runs"

    def __iter__(self):
        '''return an iterator over the individual Runs'''
        return iter(self.runs)

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------

    def __init__(self, runs):
        self.runs = runs

    @classmethod
    def from_dir(cls, directory, pattern='NGC*', strict=False):
        '''init by finding all run files in a directory'''

        directory = pathlib.Path(directory)

        runs = []

        for fn in directory.glob(pattern):

            cluster = fn.name

            file = pathlib.Path(fn.resolve(), f'{cluster}_sampler.hdf')

            obs = Observations(cluster)

            try:
                # TODO support both nested and mcmc, somehow organically
                run = NestedRun(file, obs, name=obs.cluster)

            except KeyError as err:

                mssg = f'Failed to create run for {obs.cluster}: {err}'

                if strict:
                    raise RuntimeError(mssg)
                else:
                    logging.debug(mssg)
                    continue

            runs.append(run)

        if not runs:
            mssg = f"No valid runs found in {directory}"
            raise RuntimeError(mssg)

        runs.sort(key=lambda run: run.name)

        return cls(runs)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _get_params(self, N_simruns=100, label_fixed=True):

        # TODO make these into a sort of property which checks if N_simruns is
        #   the same as last time it was computed and just returns that then

        params = []

        for run in self.runs:

            lbls, _ = run._get_chains()

            if label_fixed is False:
                lbls = [k[:-8] if k.endswith(' (fixed)') else k for k in lbls]

            sr = run._sim_errors(Nruns=N_simruns)

            mn = run.parameter_means(sim_runs=sr)[0]
            std = np.sqrt(np.diag(run.parameter_vars(sim_runs=sr)[0]))

            params.append({k: (mn[i], std[i]) for i, k in enumerate(lbls)})

        return params

    # ----------------------------------------------------------------------
    # Model Collection Visualizers
    # ----------------------------------------------------------------------

    def get_models(self, load=True):

        if load:
            filenames = [run.file.filename for run in self.runs]
            return ModelCollection.load(filenames, ci=False)

        else:
            chains = [run.parameter_means(1)[0] for run in self.runs]

            obs_list = [run.obs for run in self.runs]

            return ModelCollection.from_chains(chains, obs_list, ci=False)

    def get_CImodels(self, N=100, Nprocesses=1, add_errors=False, shuffle=True,
                     load=True):
        import multiprocessing

        if load:
            filenames = [run.file.filename for run in self.runs]
            return ModelCollection.load(filenames, ci=True)

        else:
            chains = []
            obs_list = []

            for run in self.runs:
                _, ch = run._get_equal_weight_chains(add_errors=add_errors)

                if shuffle:
                    np.random.default_rng().shuffle(ch, axis=0)

                chains.append(ch)
                obs_list.append(run.obs)

            with multiprocessing.Pool(processes=Nprocesses) as pool:

                return ModelCollection.from_chains(chains, obs_list, ci=True,
                                                   N=N, pool=pool)

    # ----------------------------------------------------------------------
    # Iterative plots
    # ----------------------------------------------------------------------

    def iter_plots(self, plot_func, yield_run=False, *args, **kwargs):
        '''calls each run's `plot_func`, yields a figure
        all args, kwargs passed to plot func
        '''
        for run in self.runs:
            fig = getattr(run, plot_func)(*args, **kwargs)

            yield (fig, run) if yield_run else fig

    def save_plots(self, plot_func, fn_pattern=None, save_kw=None,
                   *args, **kwargs):
        '''
        fn_pattern is format string which will be passed the kw "cluster" name
            (i.e. `fn_pattern.format(cluster=run.name)`)
            if None, will be ./{cluster}_{plot_func[5:]}
            (Include the desired dir here too)
        '''

        if fn_pattern is None:
            fn_pattern = f'./{{cluster}}_{plot_func[5:]}'

        if save_kw is None:
            save_kw = {}

        for fig, run in self.iter_plots(plot_func, True, *args, **kwargs):

            save_kw['fname'] = fn_pattern.format(cluster=run.name)

            fig.savefig(**save_kw)

    # ----------------------------------------------------------------------
    # Comparison plots
    # ----------------------------------------------------------------------

    def plot_a3_FeHe(self, fig=None, ax=None, show_kroupa=False,
                     *args, **kwargs):
        '''Some special cases of plot_relation can have their own named func'''

        fig = self.plot_relation('FeHe', 'a3', fig, ax, *args, **kwargs)

        if show_kroupa:

            ax = fig.gca()

            ax.axhline(y=2.3, color='r')

            ax2 = ax.secondary_yaxis('right')

            ax2.set_yticks([2.3], [r'Kroupa ($\alpha_3=2.3$)'], c='r')

        return fig

    def plot_relation(self, param1, param2, fig=None, ax=None, *,
                      errors='bars', N_simruns=100, annotate=False, **kwargs):
        '''plot correlation between two param means with all runs

        errorbars, or 2d-ellipses
        '''

        def _get_data(prms, mdata, key):
            '''helper func for loading data from params or from metadata'''
            try:
                return prms[key]
            except KeyError as err:
                try:
                    return mdata[key], 0.
                except KeyError:
                    raise KeyError(f'{key} is not a valid param') from err

        fig, ax = self._setup_artist(fig, ax)

        # TODO also support metadata
        params = self._get_params(N_simruns=N_simruns)

        clrs = self._cmap(np.linspace(0., 1., len(self.runs)))

        x, dx = np.array([_get_data(prms, run.obs.mdata, param1)
                          for run, prms in zip(self.runs, params)]).T
        y, dy = np.array([_get_data(prms, run.obs.mdata, param2)
                          for run, prms in zip(self.runs, params)]).T

        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', ecolor=clrs, **kwargs)
        ax.scatter(x, y, color=clrs, picker=True, **kwargs)  # TODO pickradius?

        # TODO mathy labels (just need a better labels method, sep from chains)
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)

        if annotate:
            _Annotator(fig, self.runs, x, y)

        else:
            # TODO not happy with any legend
            fig.legend(loc='upper center', ncol=10)

        return fig

    def plot_lit_comp(self, param, truths, e_truths=None, src_truths='',
                      fig=None, ax=None, *,
                      clr_param=None, residuals=False, inset=False,
                      N_simruns=100, annotate=False, diagonal=True, **kwargs):
        '''plot a x-y comparison against provided literature values'''
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        def _get_data(prms, mdata, key):
            '''helper func for loading data from params or from metadata'''
            try:
                return prms[key]
            except KeyError as err:
                try:
                    return mdata[key], 0.
                except KeyError:
                    raise KeyError(f'{key} is not a valid param') from err

        def add_residuals(ax, y1, y2, e1, e2, clrs=None, res_ax=None):
            # make a res ax
            if res_ax is None:
                divider = make_axes_locatable(ax)
                res_ax = divider.append_axes('bottom', size="15%", pad=0.1, sharex=ax)

            res_ax.grid()
            res_ax.set_xscale(ax.get_xscale())
            res_ax.set_ylabel(r"% difference")

            # plot residuals (in percent)
            res = 100 * (y2 - y1) / y1
            res_err = 100 * np.sqrt(e1**2 + e2**2) / y1
            res_ax.errorbar(y1, res, yerr=res_err, fmt='none', ecolor=clrs)
            res_ax.scatter(y1, res, color=clrs,)

            return res_ax

        fig, ax = self._setup_artist(fig, ax)

        # TODO also support metadata
        params = self._get_params(N_simruns=N_simruns)

        if clr_param is None:
            clrs = self._cmap(np.linspace(0., 1., len(self.runs)))
        else:
            cvals = np.array([_get_data(prms, run.obs.mdata, clr_param)
                              for run, prms in zip(self.runs, params)]).T[0]
            cnorm = mpl_clr.Normalize(cvals.min(), cvals.max())
            clrs = self._cmap(cnorm(cvals))

        x, dx = np.array([_get_data(prms, run.obs.mdata, param)
                          for run, prms in zip(self.runs, params)]).T

        y, dy = truths, e_truths

        ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', ecolor=clrs, **kwargs)
        pnts = ax.scatter(x, y, color=clrs, picker=True, **kwargs)
        # TODO pickradius?

        if diagonal:
            grid_kw = {
                'color': plt.rcParams.get('grid.color'),
                'linestyle': plt.rcParams.get('grid.linestyle'),
                'linewidth': plt.rcParams.get('grid.linewidth'),
                'alpha': plt.rcParams.get('grid.alpha'),
                'zorder': 0.5
            }
            ax.axline((0, 0), (1, 1), **grid_kw)

        # TODO mathy labels (just need a better labels method, sep from chains)
        ax.set_xlabel(param)
        ax.set_ylabel(src_truths)

        ax.set_xlim(0.)
        ax.set_ylim(0.)

        divider = make_axes_locatable(ax)

        if clr_param is not None:
            # TODO all the colours stuff is all confused
            #   highly doubt this is the best way to be marking them
            cticks = [0, .25, .5, .75, 1.]
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar = fig.colorbar(pnts, cax=cax, ticks=cticks)
            cbar.ax.set_ylabel(clr_param)
            cbar.ax.set_yticklabels([f'{t:.2f}' for t in cnorm.inverse(cticks)])

        if residuals:
            res_ax = divider.append_axes('bottom', size="15%", pad=0, sharex=ax)
            res_ax = add_residuals(ax, x, y, dx, dy, clrs, res_ax=res_ax)
            res_ax.set_xlabel(param)

        if annotate:
            _Annotator(fig, self.runs, x, y)
        else:
            # TODO not happy with any legend
            fig.legend(loc='upper center', ncol=10)

        # ALL THIS IS TEMPORARY BUT I ALSO KINDA LIKE IT CAUSE ω-cen IS BIG

        if inset:
            axins = ax.inset_axes([0.4, 0.03, 0.57, 0.57])
            axins.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', ecolor=clrs, **kwargs)
            axins.scatter(x, y, color=clrs, picker=True, **kwargs)
            x1, x2, y1, y2 = 0, 1, 0, 1
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels([])
            axins.set_yticklabels([])

            if diagonal:
                grid_kw = {
                    'color': plt.rcParams.get('grid.color'),
                    'linestyle': plt.rcParams.get('grid.linestyle'),
                    'linewidth': plt.rcParams.get('grid.linewidth'),
                    'alpha': plt.rcParams.get('grid.alpha'),
                    'zorder': 0.5
                }
                axins.axline((0, 0), (1, 1), **grid_kw)

            ax.indicate_inset_zoom(axins, edgecolor="black")

        return fig

    # ----------------------------------------------------------------------
    # Summary plots
    # ----------------------------------------------------------------------

    def summary(self, out=sys.stdout, *, N_simruns=100):
        '''output a table of all parameter means for each cluster'''
        # TODO also a latex table version (use pandas)

        labels = list(self.runs[0].obs.initials)

        hdr = "\t".join(f"{lbl}\tσ_{lbl}" for lbl in labels)

        out.write(f'Cluster\t{hdr}\n')

        params = self._get_params(N_simruns=N_simruns)

        for run, prms in zip(self.runs, params):

            # TODO be sure this is ordered correctly
            line = "\t".join(f"{p[0]:.4f}\t{p[1]:.4f}" for p in prms.values())

            out.write(f'{run.name}\t{line}\n')