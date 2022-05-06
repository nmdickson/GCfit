from .. import Observations
from ..probabilities import priors
from .models import CIModelVisualizer, ModelVisualizer, ModelCollection

import sys
import pathlib
import logging
import warnings
import itertools
import contextlib

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_clr
import matplotlib.offsetbox as mpl_obx

from mpl_toolkits.axes_grid1 import make_axes_locatable


__all__ = ['RunCollection', 'MCMCRun', 'NestedRun']


# --------------------------------------------------------------------------
# Individual Run Analysis
# --------------------------------------------------------------------------


# TODO a way to plot our priors, probably for both vizs
class _RunAnalysis:
    '''base class for all visualizers of all run types

    filename : path to run output file
    group : base group for sampler outputs, probably either 'nested' or 'mcmc'
    '''

    _cmap = plt.cm.get_cmap('viridis')

    def __str__(self):
        try:
            return f'{self._filename} - Run Results'
        except AttributeError:
            return "Run Results"

    def __init__(self, filename, observations, group, name=None, *,
                 strict=True):

        self._filename = filename
        self._gname = group

        # if strict, ensure all necessary groups exist in the given file
        if strict:
            with h5py.File(filename, 'r') as file:
                reqd_groups = {group, 'metadata'}

                if missing_groups := (reqd_groups - file.keys()):
                    mssg = (f"Output file {filename} is invalid: "
                            f"missing {missing_groups} groups. "
                            "Are you sure this was created by GCfit?")
                    raise RuntimeError(mssg)

        if name is not None:
            self.name = name

        if observations is not None:
            self.obs = observations

        else:
            try:
                with h5py.File(filename, 'r') as file:
                    cluster = file['metadata'].attrs['cluster']

                self.obs = Observations(cluster)

            except KeyError as err:
                mssg = "No cluster name in metadata, must supply observations"
                raise ValueError(mssg) from err

    @contextlib.contextmanager
    def _openfile(self, group=None, mode='r'):
        file = h5py.File(self._filename, mode)

        try:

            if group is not None:
                yield file[group]

            else:
                yield file

        finally:
            file.close()

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

    def add_residuals(self, ax, y1, y2, e1, e2, clrs=None,
                      res_ax=None, loc='bottom', size='15%', pad=0.1):
        '''add an axis showing the residuals of two quantities'''

        # make a newres ax if needed
        if res_ax is None:
            divider = make_axes_locatable(ax)
            res_ax = divider.append_axes(loc, size=size, pad=pad, sharex=ax)

        res_ax.grid()
        res_ax.set_xscale(ax.get_xscale())
        res_ax.set_ylabel(r"% difference")

        # plot residuals (in percent)
        res = 100 * (y2 - y1) / y1
        res_err = 100 * np.sqrt(e1**2 + e2**2) / y1
        res_ax.errorbar(y1, res, yerr=res_err, fmt='none', ecolor=clrs)
        res_ax.scatter(y1, res, color=clrs,)

        return res_ax


class MCMCRun(_RunAnalysis):
    '''All the plots based on a model run, like the chains and likelihoods
    and marginals corner plots and etc

    based on an output file I guess?
    '''

    def __init__(self, filename, observations=None, group='mcmc', name=None):

        super().__init__(filename, observations, group, name)

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
        if not isinstance(value, slice):
            mssg = f"`iteration` must be a slice, not {type(value)}"
            raise TypeError(mssg)

        if value.stop is None and self.cut_incomplete:

            with self._openfile(self._gname) as file:
                stop = file.attrs['iteration']

            value = slice(value.start, stop, value.step)

        self._iterations = value

    @property
    def _iteration_domain(self):

        if (start := self.iterations.start) is None:
            start = 0

        if (stop := self.iterations.stop) is None:

            with self._openfile(self._gname) as file:
                stop = file['chain'].shape[0]

        step = self.iterations.step

        return np.arange(start + 1, stop + 1, step)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _get_labels(self, label_fixed=True, math_labels=False):

        labels = list(self.obs.initials)

        if math_labels:

            math_mapping = {
                'W0': r'$\phi_0$',
                'M': r'$M$',
                'rh': r'$r_h$',
                'ra': r'$\log\left(r_a\right)$',
                'g': r'$g$',
                'delta': r'$\delta$',
                's2': r'$s^2$',
                'F': r'$F$',
                'a1': r'$\alpha_1$',
                'a2': r'$\alpha_2$',
                'a3': r'$\alpha_3$',
                'BHret': r'$\mathrm{BH}_{ret}$',
                'd': r'$d$',
            }

            labels = [math_mapping[lbl] for lbl in labels]

        if label_fixed:

            with self._openfile('metadata') as mdata:

                fixed = sorted(
                    ((k, labels.index(k)) for k in
                     mdata['fixed_params'].attrs),
                    key=lambda item: labels.index(item[0])
                )

            for k, i in fixed:
                labels[i] += ' (fixed)'

        return labels

    def _get_chains(self, flatten=False):
        '''get the chains, properly using the iterations and walkers set,
        and accounting for fixed params'''

        with self._openfile() as file:

            labels = list(self.obs.initials)

            chain = self._reduce(file[self._gname]['chain'])

            # Handle fixed parameters

            fixed = sorted(
                ((k, v, labels.index(k)) for k, v in
                 file['metadata']['fixed_params'].attrs.items()),
                key=lambda item: labels.index(item[0])
            )

            for k, v, i in fixed:
                labels[i] += ' (fixed)'
                chain = np.insert(chain, i, v, axis=-1)

        if flatten:
            chain = chain.reshape((-1, chain.shape[-1]))

        return labels, chain

    def _reconstruct_priors(self):
        '''based on the stored "specified_priors" get a PriorTransform object'''

        with self._openfile('metadata') as mdata:

            stored_priors = dict(mdata['specified_priors'].attrs)
            fixed = dict(mdata['fixed_params'].attrs)

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

        labels, chain = self._get_chains()

        return ModelVisualizer.from_chain(chain, self.obs, method)

    def get_CImodel(self, N=100, Nprocesses=1, load=False):
        import multiprocessing

        if load:
            return CIModelVisualizer.load(self._filename)

        else:

            labels, chain = self._get_chains()

            with multiprocessing.Pool(processes=Nprocesses) as pool:
                return CIModelVisualizer.from_chain(chain, self.obs,
                                                    N, pool=pool)

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

    def plot_params(self, fig=None, params=None, *,
                    posterior_color='tab:blue', posterior_border=True,
                    ylims=None, truths=None, **kw):

        # ------------------------------------------------------------------
        # Setup plotting kwarg defaults
        # ------------------------------------------------------------------

        color = mpl_clr.to_rgb(posterior_color)
        facecolor = color + (0.33, )

        kw.setdefault('marker', '.')

        # ------------------------------------------------------------------
        # Get the sample chains (weighted and unweighted), paring down the
        # chains to only the desired params, if provided
        # ------------------------------------------------------------------

        labels, chain = self._get_chains(flatten=True)

        # params is None or a list of string labels
        if params is not None:
            prm_inds = [labels.index(p) for p in params]

            labels = params
            chain = chain[..., prm_inds]

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

        if shape := len(labels) > 5:
            shape = int(np.ceil(shape / 2), 2)

        fig, axes = self._setup_multi_artist(fig, shape, sharex=True,
                                             gridspec_kw=gs_kw)

        axes = axes.reshape(shape)

        for ax in axes[-1]:
            ax.set_xlabel(r'$-\ln(X)$')

        # ------------------------------------------------------------------
        # Plot each parameter
        # ------------------------------------------------------------------

        for ind, ax in enumerate(axes[1:].flatten()):

            # --------------------------------------------------------------
            # Get the relevant samples.
            # If necessary, remove any unneeded axes
            # --------------------------------------------------------------

            try:
                lbl, prm = labels[ind], chain[:, ind]
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
            ax.scatter(self._iteration_domain, prm, cmap=self._cmap, **kw)

            ax.set_ylabel(lbl)
            ax.set_xlim(left=0)

            # --------------------------------------------------------------
            # Plot the posterior distribution (accounting for weights)
            # --------------------------------------------------------------

            post_kw = {
                'chain': prm,
                'flipped': True,
                'truth': truths if truths is None else truths[ind],
                'truth_ci': truth_ci if truth_ci is None else truth_ci[ind],
                'color': color,
                'fc': facecolor
            }

            self.plot_posterior(lbl, fig=fig, ax=post_ax, **post_kw)

            if not posterior_border:
                post_ax.axis('off')

            # TODO maybe put ticks on right side as well?
            for tk in post_ax.get_yticklabels():
                tk.set_visible(False)

            ax.set_ylim(ylims[ind])

        return fig

    def plot_indiv(self, fig=None):

        with self._openfile(self._gname) as file:
            try:
                probs = file['blobs']
            except KeyError as err:
                mssg = f"No individial likelihoods stored as blobs"
                raise KeyError(mssg) from err

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

        fig, ax = self._setup_multi_artist(fig, shape=None,
                                           constrained_layout=False)

        labels, chain = self._get_chains()

        chain = chain.reshape((-1, chain.shape[-1]))

        # ugly
        ranges = [1. if 'fixed' not in lbl
                  else (chain[0, i] - 1, chain[0, i] + 1)
                  for i, lbl in enumerate(labels)]

        corner_kw.setdefault('plot_datapoints', False)
        corner_kw.setdefault('labelpad', 0.25)

        fig = corner.corner(chain, labels=labels, fig=fig,
                            range=ranges, **corner_kw)

        fig.subplots_adjust(left=0.05, bottom=0.06)

        return fig

    def plot_posterior(self, param, fig=None, ax=None, chain=None,
                       flipped=True, truth=None, truth_ci=None,
                       *args, **kwargs):
        '''Plot the posterior distribution of a single parameter

        to be used on the (flipped) right side of the full param plot

        param : str name
        chain : chain to create the posterior from. by default will get the
                usual full chain
        flipped : bool, whether to plot horizontal (True, default) or vertical
        '''
        from scipy.stats import gaussian_kde

        fig, ax = self._setup_artist(fig, ax)

        labels = self._get_labels()

        if param not in labels:
            mssg = f'Invalid param "{param}". Must be one of {labels}'
            raise ValueError(mssg)

        if chain is None:

            prm_ind = labels.index(param)
            chain = self._get_chains(flatten=True)[1][..., prm_ind]

        kde = gaussian_kde(chain)

        domain = np.linspace(chain.min(), chain.max(), 500)

        if flipped:

            ax.fill_betweenx(domain, 0, kde(domain), *args, **kwargs)

            if truth is not None:
                ax.axhline(truth, c='tab:red')

                if truth_ci is not None:
                    ax.axhspan(*truth_ci, color='tab:red', alpha=0.33)

            ax.set_xlim(left=0)

        else:

            ax.fill_between(domain, 0, kde(domain), *args, **kwargs)

            if truth is not None:
                ax.axvline(truth, c='tab:red')

                if truth_ci is not None:
                    ax.axvspan(*truth_ci, color='tab:red', alpha=0.33)

            ax.set_ylim(bottom=0)

        return fig

    def plot_acceptance(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        with self._openfile() as file:
            try:
                acc = self._reduce(file['statistics']['acceptance_rate'])
            except KeyError as err:
                mssg = f"No acceptance rate stored"
                raise KeyError(mssg) from err

        ax.plot(self._iteration_domain, acc)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Acceptance Rate')

        return fig

    def plot_probability(self, fig=None, ax=None):

        fig, ax = self._setup_artist(fig, ax)

        with self._openfile(self._gname) as file:
            prob = self._reduce(file['log_prob'])

        ax.plot(self._iteration_domain, prob)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Total Log Posterior Probability')

        return fig

    # ----------------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------------

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

            with self._openfile() as file:

                # INFO OF RUN
                mssg += f'\nRun Metadata'
                mssg += f'\n{"=" * 12}\n'

                # number of iterations
                Niter = file[self._gname].attrs['iteration']
                mssg += f'Iterations = {Niter}\n'

                # dimensions ndim, nwalkers
                Ndim = file[self._gname].attrs['ndim']
                Nwalkers = file[self._gname].attrs['nwalkers']
                mssg += f'Dimensions = ({Nwalkers}, {Ndim})\n'

                mdata = file['metadata']

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

        # TODO If maxfrac is added as arg, make sure to add here as well

        with self._openfile('metadata') as mdata:
            try:
                stop_kw = {'pfrac': mdata.attrs['pfrac']}

            except KeyError:
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
    def AIC(self):

        with self._openfile() as file:

            exc = [L.decode() for L in
                   file['metadata/excluded_likelihoods'].attrs.values()]

            N = sum([self.obs[comp[0]].size for comp in
                     self.obs.filter_likelihoods(exc, True)])

            k = len(self._get_chains(include_fixed=False)[1])
            lnL0 = np.max(file[self._gname]['logl'][:])

        AIC = -2 * lnL0 + (2 * k) + ((2 * k * (k + 1)) / (N - k - 1))

        return AIC

    @property
    def BIC(self):

        with self._openfile() as file:

            exc = [L.decode() for L in
                   file['metadata/excluded_likelihoods'].attrs.values()]

            N = sum([self.obs[comp[0]].size for comp in
                     self.obs.filter_likelihoods(exc, True)])

            k = len(self._get_chains(include_fixed=False)[1])
            lnL0 = np.max(file[self._gname]['logl'][:])

        BIC = -2 * lnL0 + (k * np.log(N))

        return BIC

    @property
    def _resampled_weights(self):
        from scipy.stats import gaussian_kde
        from dynesty.utils import resample_equal

        # "resample" logvols so they all have equal weights
        eq_logvol = resample_equal(-self.results.logvol, self.weights)

        # Compute the KDE of resampled logvols and evaluate on normal logvols
        return gaussian_kde(eq_logvol)(-self.results.logvol)

    def __init__(self, filename, observations=None, group='nested', name=None):

        super().__init__(filename, observations, group, name)

        self.results = self._get_results()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _get_results(self, finite_only=False):
        '''return a dynesty-style `Results` class'''
        from dynesty.results import Results

        with self._openfile(self._gname) as file:

            if finite_only:
                inds = file['logl'][:] > -1e300
            else:
                inds = slice(None)

            r = {}

            for k, d in file.items():

                if k in ('current_batch', 'initial_batch', 'bound'):
                    continue

                if d.shape and (d.shape[0] == file['logl'].shape[0]):
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

        with self._openfile(self._gname) as file:

            bnd_grp = file['bound']

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

    def _get_labels(self, label_fixed=True, math_labels=False):

        labels = list(self.obs.initials)

        # TODO make sure all plots are showing math, units and stuff in labels
        if math_labels:

            math_mapping = {
                'W0': r'$\phi_0$',
                'M': r'$M$',
                'rh': r'$r_h$',
                'ra': r'$\log\left(r_a\right)$',
                'g': r'$g$',
                'delta': r'$\delta$',
                's2': r'$s^2$',
                'F': r'$F$',
                'a1': r'$\alpha_1$',
                'a2': r'$\alpha_2$',
                'a3': r'$\alpha_3$',
                'BHret': r'$\mathrm{BH}_{ret}$',
                'd': r'$d$',
            }

            labels = [math_mapping[lbl] for lbl in labels]

        if label_fixed:

            with self._openfile('metadata') as mdata:

                fixed = sorted(
                    ((k, labels.index(k)) for k in mdata['fixed_params'].attrs),
                    key=lambda item: labels.index(item[0])
                )

            for k, i in fixed:
                labels[i] += ' (fixed)'

        return labels

    # TODO some ways of handling and plotting initial_batch only clusters
    def _get_chains(self, include_fixed=True):
        '''for nested sampling results (current Batch)'''

        with self._openfile() as file:

            chain = file[self._gname]['samples'][:]

            labels = list(self.obs.initials)

            fixed = sorted(
                ((k, v, labels.index(k)) for k, v in
                 file['metadata']['fixed_params'].attrs.items()),
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

        with self._openfile() as file:

            if add_errors is False:
                chain = file[self._gname]['samples'][:]
                eq_chain = resample_equal(chain, self.weights)

            else:
                from dynesty.dynamicsampler import weight_function
                sim_run = self._sim_errors(1)[0]
                sim_wt = weight_function(sim_run, {'pfrac': 1.}, True)[1][2]
                eq_chain = resample_equal(sim_run.samples, sim_wt)

            labels = list(self.obs.initials)

            fixed = sorted(
                ((k, v, labels.index(k)) for k, v in
                 file['metadata']['fixed_params'].attrs.items()),
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

        with self._openfile('metadata') as mdata:

            stored_priors = dict(mdata['specified_priors'].attrs)
            fixed = dict(mdata['fixed_params'].attrs)

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

    def get_CImodel(self, N=100, Nprocesses=1, add_errors=False, shuffle=True,
                    load=False):
        import multiprocessing

        if load:
            return CIModelVisualizer.load(self._filename)

        else:
            labels, chain = self._get_equal_weight_chains(add_errors=add_errors)

            if shuffle:
                np.random.default_rng().shuffle(chain, axis=0)

            with multiprocessing.Pool(processes=Nprocesses) as pool:
                return CIModelVisualizer.from_chain(chain, self.obs,
                                                    N, pool=pool)

    # ----------------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------------

    def plot_marginals(self, fig=None, full_volume=False, **corner_kw):
        import corner

        fig, ax = self._setup_multi_artist(fig, shape=None,
                                           constrained_layout=False)

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
        corner_kw.setdefault('labelpad', 0.25)

        fig = corner.corner(chain, labels=labels, fig=fig,
                            range=ranges, **corner_kw)

        fig.subplots_adjust(left=0.05, bottom=0.06)

        return fig

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

            with self._openfile('metadata') as mdata:

                try:
                    maxfrac = mdata.attrs['maxfrac']

                except KeyError:

                    maxfrac = 0.8

                    mssg = "No maxfrac stored in metadata, defaulting to 80%"
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

    def plot_posterior(self, param, fig=None, ax=None, chain=None,
                       flipped=True, truth=None, truth_ci=None,
                       *args, **kwargs):
        '''Plot the posterior distribution of a single parameter

        to be used on the (flipped) right side of the full param plot

        param : str name
        chain : chain to create the posterior from. by default will compute the
                equal weighted chain
        flipped : bool, whether to plot horizontal (True, default) or vertical
        '''
        from scipy.stats import gaussian_kde

        fig, ax = self._setup_artist(fig, ax)

        labels = self._get_labels()

        if param not in labels:
            mssg = f'Invalid param "{param}". Must be one of {labels}'
            raise ValueError(mssg)

        if chain is None:

            prm_ind = labels.index(param)
            chain = self._get_equal_weight_chains()[1][..., prm_ind]

        kde = gaussian_kde(chain)

        domain = np.linspace(chain.min(), chain.max(), 500)

        if flipped:

            ax.fill_betweenx(domain, 0, kde(domain), *args, **kwargs)

            if truth is not None:
                ax.axhline(truth, c='tab:red')

                if truth_ci is not None:
                    ax.axhspan(*truth_ci, color='tab:red', alpha=0.33)

            ax.set_xlim(left=0)

        else:

            ax.fill_between(domain, 0, kde(domain), *args, **kwargs)

            if truth is not None:
                ax.axvline(truth, c='tab:red')

                if truth_ci is not None:
                    ax.axvspan(*truth_ci, color='tab:red', alpha=0.33)

            ax.set_ylim(bottom=0)

        return fig

    def plot_params(self, fig=None, params=None, *,
                    posterior_color='tab:blue', posterior_border=True,
                    show_weight=True, fill_type='weights', ylims=None,
                    truths=None, **kw):

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

            post_kw = {
                'chain': eq_prm,
                'flipped': True,
                'truth': truths if truths is None else truths[ind],
                'truth_ci': truth_ci if truth_ci is None else truth_ci[ind],
                'color': color,
                'fc': facecolor
            }

            self.plot_posterior(lbl, fig=fig, ax=post_ax, **post_kw)

            if not posterior_border:
                post_ax.axis('off')

            # TODO maybe put ticks on right side as well?
            for tk in post_ax.get_yticklabels():
                tk.set_visible(False)

            ax.set_ylim(ylims[ind])

        return fig

    def plot_IMF(self, fig=None, ax=None, show_canonical='all', ci=True):
        '''Plot the IMF, based on the alpha exponents'''
        def salpeter(m):
            return m**-2.35

        def chabrier(m):
            k = 0.158 * np.exp(-(-np.log10(0.08))**2 / (2 * 0.69**2))
            imf = k * m**-2.3
            imf[m <= 1] = (0.158 * (1. / m[m <= 1])
                           * np.exp(-(np.log10(m[m <= 1]) - np.log10(0.08))**2
                                    / (2 * 0.69**2)))
            return imf

        def kroupa(m):
            imf = 0.08**-0.3 * (0.5 / 0.08)**-1.3 * (m / 0.5)**-2.3
            imf[m < 0.5] = 0.08**-0.3 * (m[m < 0.5] / 0.08)**-1.3
            imf[m < 0.08] = m[m < 0.08]**-0.3
            return imf

        def this_imf(m, perc=50):
            '''perc is percentile of alpha chain to use'''

            ch = self._get_equal_weight_chains()[1]
            a1, a2, a3 = np.percentile(ch[:, 8:11], perc, axis=0)

            imf = 0.5**-a1 * (1 / 0.5)**-a2 * (m / 1)**-a3
            imf[m < 1] = 0.5**-a1 * (m[m < 1] / 0.5)**-a2
            imf[m < 0.5] = m[m < 0.5]**-a1
            return imf

        fig, ax = self._setup_artist(fig, ax)

        m0 = np.array([1])
        m_domain = np.logspace(-2, 2, 400)

        if show_canonical is True or show_canonical == 'all':
            show_canonical = {'salpeter', 'chabrier', 'kroupa'}

        if 'salpeter' in show_canonical:
            norm = salpeter(m0)
            ax.loglog(m_domain, salpeter(m_domain) / norm, label='Salpeter')

        if 'chabrier' in show_canonical:
            norm = chabrier(m0)
            ax.loglog(m_domain, chabrier(m_domain) / norm, label='Chabrier')

        if 'kroupa' in show_canonical:
            norm = kroupa(m0)
            ax.loglog(m_domain, kroupa(m_domain) / norm, label='Kroupa')

        # plot median
        med_plot, = ax.loglog(m_domain, this_imf(m_domain) / this_imf(m0))

        # if ci, plot confidence interval
        if ci:
            lower = this_imf(m_domain, perc=15.87) / this_imf(m0, perc=15.87)
            upper = this_imf(m_domain, perc=84.13) / this_imf(m0, perc=84.13)

            # TODO better label?
            ax.fill_between(m_domain, upper, lower,
                            alpha=0.3, color=med_plot.get_color(),
                            label=getattr(self, 'name', None))

        ax.set_xlabel(r'Mass $[M_{\odot}]$')
        ax.set_ylabel(r'Mass Function $\xi(m)\Delta m$')

        ax.legend()

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

    def parameter_summary(self, *, N_simruns=100, label_fixed=False):
        '''return a dictionary with mean & std for each parameter'''

        labels = self._get_labels(label_fixed=label_fixed)

        sr = self._sim_errors(N_simruns)
        mns, _ = self.parameter_means(sim_runs=sr, return_samples=False)
        vrs, _ = self.parameter_vars(sim_runs=sr, return_samples=False)
        std = np.sqrt(np.diag(vrs))

        return {lbl: (mns[ind], std[ind]) for ind, lbl in enumerate(labels)}

    def print_summary(self, out=None, content='all', *, N_simruns=100):
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
            labels = self._get_labels()

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

            with self._openfile('metadata') as mdata:

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

    highlight_style = {'marker': 'D', 'linestyle': 'None',
                       'mfc': 'none', 'mec': 'red', 'mew': 2.0}

    def set_text(self, text):
        ''''get the corresponding `TextArea` and set its text'''
        return self.annotation.get_child().set_text(text)

    def set_highlight(self, x, y):
        self.highlight, = self.ax.plot(x, y, **self.highlight_style)

    def remove_highlight(self):
        if self.highlight:
            self.highlight.remove()
            self.highlight = None

    def __init__(self, fig, ax, runs, xdata, ydata,
                 loc='upper right', **annot_kw):
        self.fig, self.ax = fig, ax
        self.runs, self.xdata, self.ydata = runs, xdata, ydata

        self.fig.canvas.mpl_connect('pick_event', self)

        self.cur_ind = None

        # initialize annotation box
        self.annotation = mpl_obx.AnchoredText(None, loc=loc, **annot_kw)
        self.ax.add_artist(self.annotation)
        self.annotation.set_visible(False)

        self.highlight = None

    def __call__(self, event):
        ind = event.ind[0]

        cluster = self.runs[ind].name

        # get rid of the current highlight point
        self.remove_highlight()

        # rehitting the same one, hide the annotation and highlight
        if ind == self.cur_ind:
            self.cur_ind = None

            self.annotation.set_visible(False)

            self.set_text(None)

        # hitting new one, reset the text, ensure its visible and add highlight
        else:
            self.cur_ind = ind

            self.annotation.set_visible(True)

            self.set_text(cluster)

            self.set_highlight(self.xdata[ind], self.ydata[ind])

        self.fig.canvas.draw()


class RunCollection(_RunAnalysis):
    '''For analyzing a collection of runs all at once
    '''

    _src = None
    models = None

    def __str__(self):
        mssg = f"Collection of Runs"

        if self._src:
            mssg += f" from {self._src}"

        return mssg

    def __len__(self):
        return self.runs.__len__()

    # ----------------------------------------------------------------------
    # Interacting with Runs
    # ----------------------------------------------------------------------

    @property
    def names(self):
        return [r.name for r in self.runs]

    def __iter__(self):
        '''return an iterator over the individual Runs'''
        # Important that the order of self.runs (and thus this iter) is constant
        return iter(self.runs)

    def get_run(self, name):
        '''Return the run with a name `name`'''
        for run in self.runs:
            if run.name == name:
                return run

        else:
            mssg = f"No Run found with name {name}"
            raise ValueError(mssg)

    def filter_runs(self, pattern):
        '''filter runs based on name and return a new runcollection with them'''
        import fnmatch

        try:
            runs = [self.get_run(n)
                    for n in fnmatch.filter(self.names, pattern)]

        except TypeError:

            try:
                runs = [self.get_run(n)
                        for n in (set(self.names) & set(pattern))]

            except TypeError:

                mssg = (f'expected str pattern or list of names, '
                        f'not {type(pattern)}')

                raise TypeError(mssg)

        rc = RunCollection(runs, N_simruns=self._N_simruns)
        rc._cmap = self._cmap

        return rc

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------

    def __init__(self, runs, *, N_simruns=100, sort=True):

        if sort:
            runs.sort(key=lambda run: run.name)

        self.runs = runs
        self._N_simruns = N_simruns

        self._params = [r.parameter_summary(N_simruns=N_simruns) for r in runs]
        self._mdata = [{k: (v, 0) for k, v in r.obs.mdata.items()}
                       for r in runs]

    @classmethod
    def from_dir(cls, directory, pattern='**/*hdf', strict=False,
                 *args, sampler='nested', **kwargs):
        '''init by finding all run files in a directory'''

        cls._src = f'{directory}/{pattern}'

        directory = pathlib.Path(directory)

        if sampler == 'nested':
            run_cls = NestedRun
        elif sampler == 'mcmc':
            run_cls = MCMCRun
        else:
            mssg = "Invalid sampler. Must be one of {'nested', 'mcmc'}"
            raise ValueError(mssg)

        runs = []

        for fn in directory.glob(pattern):

            try:
                run = run_cls(fn)
                run.name = run.obs.cluster

            except KeyError as err:

                mssg = f'Failed to create run for {fn}: {err}'

                if strict:
                    raise RuntimeError(mssg)
                else:
                    logging.debug(mssg)
                    continue

            runs.append(run)

        if not runs:
            mssg = f"No valid runs found in {directory}"
            raise RuntimeError(mssg)

        return cls(runs, *args, **kwargs)

    @classmethod
    def from_files(cls, file_list, strict=False,
                   *args, sampler='nested', **kwargs):
        '''init by finding all run files in a directory'''

        if not file_list:
            mssg = f"`file_list` must not be empty"
            raise ValueError(mssg)

        if sampler == 'nested':
            run_cls = NestedRun
        elif sampler == 'mcmc':
            run_cls = MCMCRun
        else:
            mssg = "Invalid sampler. Must be one of {'nested', 'mcmc'}"
            raise ValueError(mssg)

        runs = []

        for file in file_list:

            file = pathlib.Path(file).resolve()

            if not file.exists():
                mssg = f"No such file: '{file}'"
                raise FileNotFoundError(mssg)

            try:
                run = run_cls(file)
                run.name = run.obs.cluster

            except KeyError as err:

                mssg = f'Failed to create run for {file}: {err}'

                if strict:
                    raise RuntimeError(mssg)
                else:
                    logging.debug(mssg)
                    continue

            runs.append(run)

        return cls(runs, *args, **kwargs)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _get_from_run(self, param):
        '''get a property from each run (BIC, AIC, ESS, etc)'''
        return [(getattr(run, param), 0) for run in self.runs]

    def _get_from_model(self, param, *, statistic=False, with_units=True,
                        **kwargs):
        '''get one of the "integrated" attributes from models (like BH mass)
        if statistic, return only the mean/std for each (used in params),
        otherwise return full array for each modelviz (used explicitly)

        if havent generated models already (using the get_*models function),
        then they will be computed here, with all **kwargs pass to it.
        if N is passed, will gen CI models, otherwise normal mean models
        '''

        # Compute models now
        if self.models is None:
            if 'N' in kwargs or 'load' in kwargs:
                self.get_CImodels(**kwargs)
            else:
                # try to load a CI first, then revert to a single model
                try:
                    self.get_CImodels(load=True, **kwargs)

                except RuntimeError:
                    self.get_models(**kwargs)

        data = getattr(self.models, param)

        if not with_units:
            try:
                data = [ds.value for ds in data]
            except AttributeError:
                pass

        if statistic:
            # Compute average and std for each run
            return [(np.median(ds), np.std(ds)) for ds in data]

        else:
            # return the full dataset for each run
            return data

    def _get_param(self, param, *, from_model=True, **kwargs):
        '''return the mean & std for a θ, metadata or model quntity "param"
        for all runs

        from_model=False if you want to really avoid model params (i.e. dont
        want to compute the models) all kwargs are passed to get_model otherwise
        '''

        err_mssg = f'No such parameter "{param}" was found'

        # try to get it from the best-fit params or metadata
        try:

            # join the param and metadata dicts (union in 3.9)
            out = [
                {**self._params[ind], **self._mdata[ind]}[param]
                for ind, run in enumerate(self.runs)
            ]

        # otherwise try to get from model properties
        # this is only worst case because may take a long time to gen models
        except KeyError as err:

            try:
                out = self._get_from_run(param)
            except AttributeError:

                if from_model:
                    try:
                        out = self._get_from_model(param, statistic=True,
                                                   with_units=False, **kwargs)
                    except AttributeError:
                        raise ValueError(err_mssg)
                else:
                    raise ValueError(err_mssg) from err

        return out

    def _get_param_chains(self, param, *, from_model=True, **kwargs):
        '''return the full chain for a θ, metadata or model quntity "param"
        for all runs

        from_model=False if you want to really avoid model params (i.e. dont
        want to compute the models) all kwargs are passed to get_model otherwise
        '''

        err_mssg = f'No such parameter "{param}" was found'

        # try to get it from the best-fit params or metadata
        try:

            labels = self.runs[0]._get_labels(label_fixed=False)

            theta = [dict(zip(labels, r._get_equal_weight_chains()[1].T))
                     for r in self.runs]

            mdata = [{k: [v, ] for k, v in r.obs.mdata.items()}
                     for r in self.runs]

            # join the param and metadata dicts (union in 3.9)
            out = [
                {**theta[ind], **mdata[ind]}[param]
                for ind, run in enumerate(self.runs)
            ]

        # otherwise try to get from model properties
        # this is only worst case because may take a long time to gen models
        except KeyError as err:

            try:
                out = [[v, ] for v, e in self._get_from_run(param)]
            except AttributeError:

                if from_model:
                    try:
                        out = self._get_from_model(param, statistic=False,
                                                   with_units=False, **kwargs)

                    except AttributeError:
                        raise ValueError(err_mssg)
                else:
                    raise ValueError(err_mssg) from err

        return out

    def _get_latex_labels(self, param):
        '''return the param names in math mode, for plotting'''

        # TODO FEHE should always be FeH, change everything (25)
        math_mapping = {
            'W0': r'$\phi_0$',
            'M': r'$M\ [10^6 M_\odot]$',
            'rh': r'$r_h\ [\mathrm{pc}]$',
            'ra': r'$\log\left(r_a/\mathrm{pc}\right)$',
            'g': r'$g$',
            'delta': r'$\delta$',
            's2': r'$s^2$',
            'F': r'$F$',
            'a1': r'$\alpha_1$',
            'a2': r'$\alpha_2$',
            'a3': r'$\alpha_3$',
            'BHret': r'$\mathrm{BH}_{ret}$',
            'd': r'$d$',
            'FeH': r'$[\mathrm{Fe}/\mathrm{H}]$',
            'Ndot': r'$\dot{N}$',
            'RA': r'RA $[\deg]$',
            'DEC': r'DEC $[\deg]$',
            'chi2': r'$\chi^2$',
            'BH_mass': r'$\mathrm{M}_{BH}\ [M_\odot]$',
            'BH_num': r'$\mathrm{N}_{BH}$',
            'f_rem': r'$f_{\mathrm{remn}}$',
        }

        return math_mapping.get(param, param)

    def _add_colours(self, ax, mappable, cparam, clabel=None, *, alpha=1.,
                     add_colorbar=True, extra_artists=None, math_label=True,
                     fix_cbar_ticks=True):
        '''add colours to all artists and add the relevant colorbar to ax'''
        import matplotlib.colorbar as mpl_cbar

        # Get colour values
        try:
            cvalues = np.array(self._get_param(cparam))[:, 0]
            clabel = cparam if clabel is None else clabel

        except TypeError:
            cvalues = cparam

        cnorm = mpl_clr.Normalize(cvalues.min(), cvalues.max())
        colors = self._cmap(cnorm(cvalues))

        colors[:, -1] = alpha

        # apply colour to all artists
        if mappable is not None:
            mappable.set_color(colors)
            mappable.cmap = self._cmap

        if extra_artists is not None:
            for artist in extra_artists:

                # Set colors normally
                try:
                    artist.set_color(colors)

                # If fails, attempt to set one colour at a time
                except (ValueError, AttributeError) as err:

                    try:
                        for i, subart in enumerate(artist):
                            subart.set_color(colors[i])

                    except (ValueError, TypeError):
                        mssg = f'Cannot `set_color` of extra artist "{artist}"'
                        raise TypeError(mssg) from err

        if add_colorbar:
            # make ax for colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)

            # make colorbar
            cbar = mpl_cbar.Colorbar(cax, mappable, cmap=self._cmap)

            clabel = self._get_latex_labels(clabel) if math_label else clabel
            cbar.set_label(clabel)

            # if desired, explicitly set ticks at 25% intervals of bar
            if fix_cbar_ticks:
                cticks = [0, .25, .5, .75, 1.]
                ctick_labels = [f'{t:.2f}' for t in cnorm.inverse(cticks)]
                cbar.set_ticks(cticks, labels=ctick_labels)

            return cbar

        else:
            return None

    # ----------------------------------------------------------------------
    # Model Collection Visualizers
    # ----------------------------------------------------------------------

    def get_models(self):

        chains = [run.parameter_means(1)[0] for run in self.runs]

        obs_list = [run.obs for run in self.runs]

        mc = ModelCollection.from_chains(chains, obs_list, ci=False)

        # save a copy of models here
        self.models = mc

        return mc

    def get_CImodels(self, N=100, Nprocesses=1, add_errors=False, shuffle=True,
                     load=True):
        import multiprocessing

        if load:
            filenames = [run._filename for run in self.runs]
            mc = ModelCollection.load(filenames)

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

                mc = ModelCollection.from_chains(chains, obs_list, ci=True,
                                                 N=N, pool=pool)

        # save a copy of models here
        self.models = mc

        return mc

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

    def save_plots(self, plot_func, fn_pattern=None, save_kw=None, size=None,
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

            if size is not None:
                fig.set_size_inches(size)

            save_kw['fname'] = fn_pattern.format(cluster=run.name)

            fig.savefig(**save_kw)

            plt.close(fig)

    # ----------------------------------------------------------------------
    # Comparison plots
    # ----------------------------------------------------------------------

    def plot_a3_FeH(self, fig=None, ax=None, show_kroupa=False,
                     *args, **kwargs):
        '''Some special cases of plot_relation can have their own named func'''

        fig = self.plot_relation('FeH', 'a3', fig, ax, *args, **kwargs)

        if show_kroupa:

            ax = fig.gca()

            ax.axhline(y=2.3, color='r')

            ax2 = ax.secondary_yaxis('right')

            ax2.set_yticks([2.3], [r'Kroupa ($\alpha_3=2.3$)'], c='r')

        return fig

    def plot_relation(self, param1, param2, fig=None, ax=None, *,
                      errors='bars', annotate=False, annotate_kwargs=None,
                      clr_param=None, clr_kwargs=None, **kwargs):
        '''plot correlation between two param means with all runs

        errorbars, or 2d-ellipses
        '''

        fig, ax = self._setup_artist(fig, ax)

        x, dx = zip(*self._get_param(param1))
        y, dy = zip(*self._get_param(param2))

        errbar = ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', **kwargs)
        points = ax.scatter(x, y, picker=True, **kwargs)

        ax.set_xlabel(self._get_latex_labels(param1))
        ax.set_ylabel(self._get_latex_labels(param2))

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            err_artists = itertools.chain.from_iterable(errbar[1:])

            self._add_colours(ax, points, clr_param,
                              extra_artists=err_artists, **clr_kwargs)

        if annotate:

            if annotate_kwargs is None:
                annotate_kwargs = {}

            _Annotator(fig, ax, self.runs, x, y, **annotate_kwargs)

        else:
            # TODO not happy with any legend
            fig.legend(loc='upper center', ncol=10)

        return fig

    def plot_lit_comp(self, param, truths, e_truths=None, src_truths='',
                      fig=None, ax=None, *,
                      clr_param=None, clr_kwargs=None,
                      annotate=False, annotate_kwargs=None,
                      residuals=False, inset=False, diagonal=True,
                      **kwargs):
        '''plot a x-y comparison against provided literature values

        Meant to compare 1-1 the same parameter (i.e. mass vs mass, etc)
        '''

        fig, ax = self._setup_artist(fig, ax)

        x, dx = zip(*self._get_param(param))
        y, dy = truths, e_truths

        errbar = ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', **kwargs)
        points = ax.scatter(x, y, picker=True, **kwargs)

        if diagonal:
            grid_kw = {
                'color': plt.rcParams.get('grid.color'),
                'linestyle': plt.rcParams.get('grid.linestyle'),
                'linewidth': plt.rcParams.get('grid.linewidth'),
                'alpha': plt.rcParams.get('grid.alpha'),
                'zorder': 0.5
            }
            ax.axline((0, 0), (1, 1), **grid_kw)

        prm_lbl = self._get_latex_labels(param)

        ax.set_xlabel(prm_lbl)
        ax.set_ylabel(prm_lbl + (f' ({src_truths})' if src_truths else ''))

        ax.set_xlim(0.)
        ax.set_ylim(0.)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            err_artists = itertools.chain.from_iterable(errbar[1:])

            self._add_colours(ax, points, clr_param,
                              extra_artists=err_artists, **clr_kwargs)

        if residuals:
            clrs = points.get_facecolors()
            res_ax = self.add_residuals(ax, x, y, dx, dy, clrs, pad=0)
            res_ax.set_xlabel(param)

        if annotate:

            if annotate_kwargs is None:
                annotate_kwargs = {}

            _Annotator(fig, ax, self.runs, x, y, **annotate_kwargs)

        else:
            # TODO not happy with any legend
            fig.legend(loc='upper center', ncol=10)

        return fig

    def plot_lit_relation(self, param,
                          lit, e_lit=None, param_lit='', src_lit='',
                          fig=None, ax=None, *, lit_on_x=False,
                          clr_param=None, clr_kwargs=None, residuals=False,
                          annotate=False, annotate_kwargs=None, **kwargs):
        '''plot a relation plot against provided literature values

        Meant to compare two different parameters, with one from outside source
        '''

        fig, ax = self._setup_artist(fig, ax)

        x, dx = zip(*self._get_param(param))
        y, dy = lit, e_lit

        xlabel = self._get_latex_labels(param)
        ylabel = (self._get_latex_labels(param_lit)
                  + (f' ({src_lit})' if src_lit else ''))

        # optionally flip the x and y
        if lit_on_x:
            x, y = y, x
            dx, dy = dy, dx
            xlabel, ylabel = ylabel, xlabel

        errbar = ax.errorbar(x, y, xerr=dx, yerr=dy, fmt='none', **kwargs)
        points = ax.scatter(x, y, picker=True, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            err_artists = itertools.chain.from_iterable(errbar[1:])

            self._add_colours(ax, points, clr_param,
                              extra_artists=err_artists, **clr_kwargs)

        if residuals:
            clrs = points.get_facecolors()
            res_ax = self.add_residuals(ax, x, y, dx, dy, clrs, pad=0)
            res_ax.set_xlabel(param)

        if annotate:

            if annotate_kwargs is None:
                annotate_kwargs = {}

            _Annotator(fig, ax, self.runs, x, y, **annotate_kwargs)

        else:
            # TODO not happy with any legend
            fig.legend(loc='upper center', ncol=10)

        return fig

    # ----------------------------------------------------------------------
    # Summary plots
    # ----------------------------------------------------------------------

    def plot_param_means(self, param, fig=None, ax=None,
                         clr_param=None, clr_kwargs=None, **kwargs):
        '''plot mean and std errorbars for each run of the given param'''
        fig, ax = self._setup_artist(fig, ax)

        mean, std = np.array(self._get_param(param)).T

        xticks = np.arange(len(self.runs))

        labels = self.names

        errbar = ax.errorbar(x=xticks, y=mean, yerr=std, fmt='none', **kwargs)
        points = ax.scatter(x=xticks, y=mean, picker=True, **kwargs)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            err_artists = itertools.chain.from_iterable(errbar[1:])

            self._add_colours(ax, points, clr_param,
                              extra_artists=err_artists, **clr_kwargs)

        ax.set_xticks(xticks, labels=labels, rotation=45)

        ax.grid(axis='x')

        ax.set_ylabel(self._get_latex_labels(param))

        return fig

    def plot_param_bar(self, param, fig=None, ax=None,
                       clr_param=None, clr_kwargs=None, *args, **kwargs):
        '''plot mean and std bar chart for each run of the given param'''
        fig, ax = self._setup_artist(fig, ax)

        mean, std = np.array(self._get_param(param)).T

        xticks = np.arange(len(self.runs))

        labels = self.names

        bars = ax.bar(x=xticks, height=mean, yerr=std, *args, **kwargs)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            clr_kwargs.setdefault('alpha', 0.3)

            self._add_colours(ax, None, clr_param,
                              extra_artists=(bars,), **clr_kwargs)

        ax.set_xticks(xticks, labels=labels, rotation=45)

        ax.set_ylabel(self._get_latex_labels(param))

        return fig

    def plot_param_violins(self, param, fig=None, ax=None,
                           clr_param=None, clr_kwargs=None, alpha=0.3,
                           *args, **kwargs):
        '''plot violins for each run of the given param'''
        fig, ax = self._setup_artist(fig, ax)

        chains = self._get_param_chains(param)

        xticks = np.arange(len(self.runs))

        labels = self.names

        parts = ax.violinplot(chains, positions=xticks, *args, **kwargs)

        if clr_param is not None:

            if clr_kwargs is None:
                clr_kwargs = {}

            clr_kwargs.setdefault('alpha', alpha)

            self._add_colours(ax, None, clr_param,
                              extra_artists=parts.values(), **clr_kwargs)

        for part in parts['bodies']:
            part.set_alpha(alpha)

        ax.set_xticks(xticks, labels=labels, rotation=45)

        ax.grid(axis='x')

        ax.set_ylabel(self._get_latex_labels(param))

        return fig

    def summary_dataframe(self, *, include_FeH=True, include_BH=False):
        import pandas as pd
        # TODO pandas isn't in the setup requirements

        # Get name of all desired parameters

        labels = self.runs[0]._get_labels(label_fixed=False)

        if include_FeH:
            labels = ['FeH'] + labels

        if include_BH:
            labels += ['BH_mass', 'BH_num', 'f_rem']

        # Fill in a dictionary of column data

        data = {}

        data['Cluster'] = [run.name for run in self.runs]

        for param in labels:
            data[param], data[f'σ_{param}'] = zip(*self._get_param(param))

        # Create dataframe

        return pd.DataFrame.from_dict(data)

    def output_summary(self, outfile=sys.stdout, style='latex', *,
                       include_FeH=True, include_BH=False, **kwargs):
        '''output a table of all parameter means for each cluster'''

        # get dataframe

        df = self.summary_dataframe(include_FeH=include_FeH,
                                    include_BH=include_BH)

        # output in desired format

        kwargs.setdefault('index', False)

        if style in ('table' or 'dat'):
            df.to_string(buf=outfile, **kwargs)

        elif style == 'latex':
            df.to_latex(buf=outfile, **kwargs)

        elif style == 'hdf':
            df.to_hdf(outfile, **kwargs)

        elif style == 'csv':
            df.to_csv(outfile, **kwargs)

        elif style == 'html':
            df.to_html(buf=outfile, **kwargs)

        else:
            mssg = ("Invalid style. Must be one of "
                    "{'table', 'latex', 'hdf', 'csv', 'html'}")
            raise ValueError(mssg)

        return df
