from ..core.data import DEFAULT_THETA

import numpy as np
from scipy import stats

import logging
import operator


__all__ = ["DEFAULT_PRIORS", "DEFAULT_EV_PRIORS",
           "Priors", "UniformPrior", "GaussianPrior",
           "BoundedGaussianPrior", "CromwellUniformPrior", "ArbitraryPrior"]


_OPER_MAP = {
    '<': operator.lt, 'lt': operator.lt,
    '<=': operator.le, '=<': operator.le, 'le': operator.le,
    '>=': operator.ge, '=>': operator.ge, 'ge': operator.ge,
    '>': operator.gt, 'gt': operator.gt,
    '=': operator.eq, '==': operator.eq, 'eq': operator.eq,
    '!=': operator.ne, 'ne': operator.ne,
}


class Priors:
    '''Representation of the prior likelihood functions on various parameters.

    Contains the individual prior likelihood functions for each parameter,
    as a corresponding `*Prior` object, and handles all interation with them.

    Calls to the prior-likelihood should be made to this class, with a given
    `theta`, which will be distributed to the relevant parameter-specific
    functions.

    Parameters
    ----------
    priors : dict
        A dictionary describing the prior function for each parameter. Each
        value should be either one of the `_PriorBase` subclasses, or a list
        of ["prior name", *function args]. Any missing parameters will be filled
        in with `DEFAULT_PRIORS`.

    fixed_initials : dict, optional
        A dictionary of any parameters which have fixed values, used in cases of
        dependant priors with these values.

    logged : bool, optional
        Whether to log the returned likelihoods. Defaults to True.

    err_on_fail : bool, optional
        If any called `theta` parameters return "invalid" priors (i.e. L <= 0),
        will raise an exception. If False (default), will simply log the
        failure and continue.
    '''

    def __call__(self, theta, *, return_indiv=False):
        '''Return the total prior likelihood for a given theta.

        Distribute `theta` to each parameter's prior function, alongside
        relevant arguments and dependant parameters, and return the prior
        likelihood value.

        If any prior values are "invalid", that is, are less than or equal to
        a probability of 0, or are `nan`, then the reasons for this occurrence
        will be compiled and logged (or raised, if `err_on_fail` was `True`).

        Parameters
        ----------
        theta : numpy.ndarray or dict
            The parameter values to pass to the prior likelihood functions.

        return_indiv : bool, optional
            Whether to return the individual likelihood values of each
            parameter, or to sum and return a single overall probability.

        Returns
        -------
        numpy.ndarray or float
            The computed prior likelihood values, either individually or (if
            `return_indiv` is `True`) summed together.

        Raises
        ------
        ValueError
            If any priors returned "invalid" likelihoods, and the `err_on_fail`
            flag was set on initialization.
        '''

        if not isinstance(theta, dict):
            theta = dict(zip(self.var_params, theta), **self.fixed_initials)

        L = {p: 0. for p in theta}
        inv = []

        for param, prior in self.priors.items():

            if prior.dependants:
                deps = {d: theta[d] for d in prior.dependants}
                P = prior(theta[param], **deps)

            else:
                P = prior(theta[param])

            # if invalid (ie 0, outside of bounds / real bad) record it's reason
            if P <= 0. or np.isnan(P):
                inv.append(prior.inv_mssg)

            L[param] += P

        # If any priors were invalid, combine the mssgs and output that
        if inv:
            # TODO needs improvement, currently does't mention why each failed
            mssg = f"L(Θ) failed on priors: {'; '.join(inv)}"
            if self._strict:
                raise ValueError(mssg)
            else:
                logging.debug(mssg)

        # Convert to an array
        L = np.fromiter(L.values(), dtype=np.float64)

        if not return_indiv:
            L = np.prod(L)

        if self._log:
            L = np.log(L)

        return L

    def __init__(self, priors, fixed_initials=None, *,
                 logged=True, err_on_fail=False, evolved=False):

        self._log = logged
        self._strict = err_on_fail

        defaults = DEFAULT_PRIORS if not evolved else DEFAULT_EV_PRIORS

        if extraneous_params := (priors.keys() - defaults.keys()):
            raise ValueError(f"Invalid parameters: {extraneous_params}")

        # ------------------------------------------------------------------
        # Prep for any fixed parameters
        # ------------------------------------------------------------------

        # In normal prior likelihoods, the fixed values will also be evaled

        if fixed_initials is None:
            fixed_initials = {}

        self.fixed_initials = fixed_initials

        # get list of variable params, sorted for the later unpacking of theta
        var_params = defaults.keys() - fixed_initials.keys()
        self.var_params = sorted(var_params, key=list(defaults).index)

        # ------------------------------------------------------------------
        # Initialize all Prior objects
        # ------------------------------------------------------------------

        # Fill in unspecified parameters with default priors
        self.priors = {**defaults, **priors}

        # Fill the dict with actual priors objects
        for param in self.priors:

            if isinstance(self.priors[param], _PriorBase):

                if self.priors[param]._transform is True:
                    mssg = (f"Prior {self.priors[param]} was already "
                            f"initialized with transform=True")
                    raise RuntimeError(mssg)

            else:
                # TODO Not happy with how nested the uniform args have to be
                prior_key, *args = self.priors[param]

                prior_key = prior_key.lower().replace('prior', '').strip()

                self.priors[param] = _PRIORS_MAP[prior_key](param, *args)


class PriorTransforms(Priors):
    '''Representation of the prior transform functions on various parameters.

    Contains the individual prior transformation functions for each parameter,
    as a corresponding `*Prior` object, and handles all interation with them.

    Calls to the prior-transforms should be made to this class, with a given
    Unif~[0,1] value for each parameter, which will be distributed to their
    relevant parameter-specific functions.

    Based on the `Priors` object, this class differs by ensuring calls to
    each prior distribution compute the "ppf", rather than the "pdf"
    This class is not meant to return prior-likelihood values for a given
    `theta`, but to transform a uniform parameter-space sample to the
    corresponding sample of `theta`, based on the prior distribution.

    This class is to be used instead of `Priors` in all nested sampling code.

    Parameters
    ----------
    priors : dict
        A dictionary describing the prior function for each parameter. Each
        value should be either one of the `_PriorBase` subclasses, or a list
        of ["prior name", *function args]. Any missing parameters will be filled
        in with `DEFAULT_PRIORS`.

    fixed_initials : dict, optional
        A dictionary of any parameters which have fixed values, used in cases of
        dependant priors with these values.

    logged : bool, optional
        Whether to log the returned likelihoods. Defaults to True.

    err_on_fail : bool, optional
        If any called `theta` parameters return "invalid" priors (i.e. L <= 0),
        will raise an exception. If False (default), will simply log the
        failure and continue.
    '''

    def _compile_dependants(self, prior, U, theta=None):
        '''Transform a passed U to theta recursively, so can use dependants'''

        # TODO potentially repeating a lot of prior calls by not saving to theta

        if not prior.dependants:
            return {}

        deps = {}

        for dep_param in prior.dependants:

            # already computed, re-use the theta value
            if dep_param in theta:
                deps[dep_param] = theta[dep_param]

            # not a fixed param, not computed yet, compute and use it's theta
            elif dep_param in U:
                dep_prior = self.priors[dep_param]
                dep_deps = self._compile_dependants(dep_prior, U, theta)
                deps[dep_param] = dep_prior(U[dep_param], **dep_deps)

            # assume it's fixed, use its fixed value
            else:
                # TODO might get hard to understand keyerror here if not fixed?
                deps[dep_param] = self.fixed_initials[dep_param]

        return deps

    def __call__(self, U):
        '''Return the corresponding theta parameters for a given uniform sample.

        Distribute `U` to each parameter's prior transform function, alongside
        relevant arguments and dependant parameters, and return the relevant
        theta values.

        If any prior values are "invalid", that is, return `nan` due to the
        input `U` being outside the [0,1] unit cube, then the reasons for
        this occurrence will be compiled and logged (or raised, if `err_on_fail`
        was `True`). This should not happen in normal circumstances, as long as
        `U` is correctly defined.

        Parameters
        ----------
        U : numpy.ndarray or dict
            The parameter values to pass to the prior transform functions. Must
            be i.i.d. within the N-dimensional unit cube (i.e. uniformly
            distributed from 0 to 1).

        Returns
        -------
        numpy.ndarray
            The computed theta parameter values.

        Raises
        ------
        ValueError
            If any priors returned "invalid" likelihoods, and the `err_on_fail`
            flag was set on initialization.
        '''

        if len(U) != len(self.var_params):
            mssg = (f"Incorrect number of parameters passed: "
                    f"expected {len(self.var_params)}, got {len(U)}")

            raise ValueError(mssg)

        if not isinstance(U, dict):
            U = dict(zip(self.var_params, U))

        theta = {}
        inv = []

        for param, prior in self.priors.items():

            deps = self._compile_dependants(prior, U, theta)

            theta[param] = prior(U[param], **deps)

            # if invalid, record it's reason
            if np.isnan(theta[param]):
                inv.append(prior.inv_mssg)

        # If any priors were invalid, combine the mssgs and output that
        if inv:
            # TODO needs improvement, currently does't mention why each failed
            mssg = f"L(Θ) failed on priors: {'; '.join(inv)}"
            if self._strict:
                raise ValueError(mssg)
            else:
                logging.debug(mssg)

        # Convert to an array
        theta = np.fromiter(theta.values(), dtype=np.float64)

        return theta

    def __init__(self, priors, fixed_initials=None, *, err_on_fail=False,
                 evolved=False):

        self._strict = err_on_fail

        defaults = DEFAULT_PRIORS if not evolved else DEFAULT_EV_PRIORS

        if extraneous_params := (priors.keys() - defaults.keys()):
            raise ValueError(f"Invalid parameters: {extraneous_params}")

        # ------------------------------------------------------------------
        # Prep for any fixed parameters
        # ------------------------------------------------------------------

        # In prior transforms, fixed values will be basically ignored

        if fixed_initials is None:
            fixed_initials = {}

        self.fixed_initials = fixed_initials

        # get list of variable params, sorted for the later unpacking of U
        var_params = defaults.keys() - fixed_initials.keys()
        self.var_params = sorted(var_params, key=list(defaults).index)

        # ------------------------------------------------------------------
        # Initialize all Prior objects
        # ------------------------------------------------------------------

        # Fill in unspecified parameters with default priors
        self.priors = {**defaults, **priors}

        for key in self.fixed_initials:
            del self.priors[key]

        # Fill the dict with actual priors objects
        for param in self.priors:

            if isinstance(self.priors[param], _PriorBase):

                if self.priors[param]._transform is False:
                    mssg = (f"Prior {self.priors[param]} was already "
                            f"initialized with transform=False")
                    raise RuntimeError(mssg)

            else:
                prior_key, *args = self.priors[param]

                prior_key = prior_key.lower().replace('prior', '').strip()

                kw = {'transform': True}

                self.priors[param] = _PRIORS_MAP[prior_key](param, *args, **kw)


class _PriorBase:
    '''Base class for all prior functions, handling some defaults.

    This class should not be instantiated directly, only used as a base for
    specific functional priors.

    Attributes
    ----------
    inv_mssg : str
        The message explaining why this prior may have returned an invalid value

    inv_value : float
        The value to return should this prior be invalid.

    dependants : list of str
        A list of parameters that this prior depends on (i.e. a parameter whose
        value is needed to set the bounds of the prior).

    param : str
        The name of the parameter this prior corresponds to
    '''

    dependants = None
    param = None

    def __repr__(self):
        return f'repr {self.__class__.__name__} prior'

    def __str__(self):
        return f'str {self.__class__.__name__} prior'

    @property
    def inv_mssg(self):
        try:
            return self._inv_mssg
        except AttributeError:
            return f"Invalid {self.__class__.__name__} on {self.param}"

    @property
    def inv_value(self):
        return np.nan if self._transform else 0.

    def _init_val(self, val):
        '''try to convert val to a float,
        else assume its a param and add to dependants'''
        try:
            # val is a specific number
            val = float(val)

        except ValueError:
            # val is a name of a param

            if val not in DEFAULT_THETA:
                raise ValueError(f'Invalid dependant parameter {val}')

            self.dependants.append(val)

        return val


class UniformPrior(_PriorBase):
    '''Flat uniform prior function.

    Represents a normalized uniform prior likelihood distribution, defined by
    N bounding pairs which must all overlap smoothly. The returned likelihood
    value for any point between the minimum and maximum bounds is defined by
    the size of the distribution, and normalized to 1.

    Bounds are evaluated on each call, in order to support dependant parameters,
    and the `inv_value` will be returned in the case of bound-pairs which do not
    overlap.

    Parameters
    ----------
    param : str
        Name of the corresponding parameter.

    edges : list of 2-tuple
        List of all bound pairs (lower, upper). Bounds can be either a float,
        for a fixed bound, or a string name of another parameter, for a
        dependant bound.

    transform : bool, optional
        Whether this is a prior likelihood or a prior transform function.
        Changes whether the distribution PDF or PPF is evaluated upon calling.

    Attributes
    ----------
    bounds : list of 2-tuple
        The bounds, as defined by the `edges` parameter.

    See Also
    --------
    scipy.stats.uniform : Distribution class used for pdf/ppf evaluation.
    '''

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'("{self.param}", {self.bounds}, transform={self._transform})')

    def __call__(self, param_val, *args, **kwargs):
        '''Evaluate this prior function at the value `param_val`.'''

        # check that all dependants were supplied
        if (missing_deps := set(self.dependants) - kwargs.keys()):
            mssg = f"Missing required dependant params: {missing_deps}"
            raise TypeError(mssg)

        # get values for any dependant params
        lowers, uppers = zip(*self.bounds)

        lowers = np.array([kwargs.get(p, p) for p in lowers])
        uppers = np.array([kwargs.get(p, p) for p in uppers])

        # Check if the bounds themselves are valid (all overlapping correctly)
        if not (valid := np.less_equal.outer(lowers, uppers)).all():

            inv_lowers = lowers[np.where(~valid)[0]]
            inv_uppers = uppers[np.where(~valid)[1]]

            inv_pairs = np.c_[inv_lowers, inv_uppers]

            nlt = u"\u226E"

            self._inv_mssg = (
                f"Invalid UniformPrior on {self.param}, improper bounds: "
                + ",".join(f"{li} {nlt} {ui}" for li, ui in inv_pairs)
            )

            return self.inv_value

        # compute overall bounds, and loc/scale
        l_bnd, r_bnd = lowers.max(), uppers.min()
        loc, scale = l_bnd, r_bnd - l_bnd

        # evaluate the dist
        return self._caller(param_val, loc=loc, scale=scale)

    def __init__(self, param, edges, *, transform=False):

        self._transform = transform

        self._caller = stats.uniform.pdf if not transform else stats.uniform.ppf

        self.param = param

        self.bounds = []
        self.dependants = []

        for bounds in edges:

            if len(bounds) != 2:
                raise ValueError(f"Invalid bounds {bounds}, must be [low, up]")

            self.bounds.append(tuple(self._init_val(bnd) for bnd in bounds))


# TODO needs a much better name (and implementation tbh)
class ArbitraryPrior(_PriorBase):
    '''Prior function defined by arbitrary operations. NotImplemented.'''

    # Consists of a number of operation: value pairs which are evaluated.
    #   operation can be anything.

    def __repr__(self):
        return (f'{self.__class__.__name__}("{self.param}", {self._eval})')

    def __call__(self, param_val, *args, **kwargs):

        for oper, bnd in self._eval:

            try:
                check = oper(param_val, bnd)
            except TypeError:
                check = oper(param_val, kwargs[bnd])

            if not check:
                self._inv_mssg = (f'Invalid ArbitraryPrior on {self.param}: '
                                  f'{param_val} not {oper.__name__} {bnd}')
                return 0.

        # TODO what value to return on success?
        return 1.

    def __init__(self, param, edges, *, transform=False):
        '''
        param is the name, just for mssgs
        edges is a list of bounds
            or (operation, param name)
            or (operation, bound)
        '''

        self._transform = transform

        if transform:
            mssg = "ArbitraryPrior does not support `transform` or ppfs"
            raise NotImplementedError(mssg)

        self.param = param

        self._eval = []
        self.dependants = []

        for bounds in edges:

            if len(bounds) != 2:
                mssg = f"Invalid bounds {bounds}, must be [oper, val]"
                raise ValueError(mssg)

            oper_str, bnd = bounds

            bnd = self._init_val(bnd)

            self._eval.append((_OPER_MAP[oper_str], bnd))


class GaussianPrior(_PriorBase):
    '''1D Gaussian normal prior function.

    Represents a Gaussian prior likelihood distribution, defined by a mean μ
    and width σ.

    Parameters
    ----------
    param : str
        Name of the corresponding parameter.

    mu : float
        Distribution mean.

    sigma : float
        Distribution width/standard deviation.

    transform : bool, optional
        Whether this is a prior likelihood or a prior transform function.
        Changes whether the distribution PDF or PPF is evaluated upon calling.

    See Also
    --------
    scipy.stats.norm : Distribution class used for pdf/ppf evaluation.
    '''

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'("{self.param}", {self.mu}, {self.sigma}, '
                f'transform={self._transform})')

    def __call__(self, param_val, *args, **kw):
        '''Evaluate this prior function at the value `param_val`.'''
        return self._caller(param_val)

    def __init__(self, param, mu, sigma, *, transform=False):

        self._transform = transform
        self.param = param

        self.mu, self.sigma = mu, sigma

        self.dist = stats.norm(loc=self.mu, scale=self.sigma)

        self._caller = self.dist.pdf if not transform else self.dist.ppf


class BoundedGaussianPrior(_PriorBase):
    '''Gaussian normal prior function with truncated bounds.

    Could maybe use truncnorm?

    NotImplemented
    '''
    pass


class CromwellUniformPrior(_PriorBase):
    '''Flat uniform prior function with slightly non-zero out-of-bounds values.

    Cromwell's Rule states that prior probabilities with absolute 1 or 0 values
    should be avoided, to allow an infinitesimal chance for the most unlikely
    of occurrences. This function behaves exactly like `UniformPrior` with the
    notable exception of an extremely small non-zero value returned outside of
    the bounds.

    NotImplemented
    '''
    pass


DEFAULT_PRIORS = {
    'W0': ('uniform', [(3, 20)]),
    'M': ('uniform', [(0.01, 5)]),
    'rh': ('uniform', [(0.5, 15)]),
    'ra': ('uniform', [(0, 5)]),
    'g': ('uniform', [(0, 3.5)]),
    'delta': ('uniform', [(0.3, 0.5)]),
    's2': ('uniform', [(0, 15)]),
    'F': ('uniform', [(1, 3)]),
    'a1': ('uniform', [(-1, 2.35)]),
    'a2': ('uniform', [(-1, 2.35), ('a1', np.inf)]),
    'a3': ('uniform', [(1.6, 4), ('a2', np.inf)]),
    'BHret': ('uniform', [(0, 100)]),
    'd': ('uniform', [(2, 18)])
}

DEFAULT_EV_PRIORS = {
    'W0': ('uniform', [(3, 20)]),
    'M0': ('uniform', [(0.001, 10)]),
    'rh0': ('uniform', [(0.1, 50)]),
    'ra': ('uniform', [(0, 5)]),
    'g': ('uniform', [(0, 3.5)]),
    'delta': ('uniform', [(0.3, 0.5)]),
    's2': ('uniform', [(0, 15)]),
    'F': ('uniform', [(1, 3)]),
    'a1': ('uniform', [(-1, 2.35)]),
    'a2': ('uniform', [(-1, 2.35), ('a1', np.inf)]),
    'a3': ('uniform', [(1.6, 4), ('a2', np.inf)]),
    'd': ('uniform', [(2, 18)])
}

_PRIORS_MAP = {
    "uniform": UniformPrior,
    "gaussian": GaussianPrior,
    "boundedgaussian": BoundedGaussianPrior,
    "cromwelluniform": CromwellUniformPrior,
}
