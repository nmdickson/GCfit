from ..core.data import DEFAULT_INITIALS

import numpy as np
from scipy import stats

import logging
import operator


__all__ = ["Priors", "DEFAULT_PRIORS", "UniformPrior", "GaussianPrior",
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
    """Container class representing the prior *logged* likelihoods,
    to be called on θ and *added* to the log likelihood"""

    def __call__(self, theta):
        '''return the total prior likelihood given by theta'''
        if not isinstance(theta, dict):
            theta = dict(zip(DEFAULT_INITIALS, theta))

        L = 0.
        inv = []

        for param, prior in self.priors.items():

            if prior.dependants:
                deps = {d: theta[d] for d in prior.dependants}
                P = prior(theta[param], **deps)

            else:
                P = prior(theta[param])

            # if invalid (i.e. -inf, outside of bounds) record it's reason
            if not np.isfinite(P):
                inv.append(prior.inv_mssg)

            L += P

        # If any priors were invalid, combine the mssgs and output that
        if inv:
            mssg = f"L(Θ) failed on priors: {'; '.join(inv)}"
            if self._strict:
                raise ValueError(mssg)
            else:
                logging.debug(mssg)

        return L

    def __init__(self, priors, transform, *, err_on_fail=False):
        '''
        priors: dict where key is a parameter, and eavh value is either a
        `*Prior` object, or ["name of class", *args for that class]

        transform : bool, use if doing nested sampling, switches to ppf
        err_on_fail : bool, if the likelihood is <= 0 will raise an error
        '''
        # TODO may be the spot to set up an initial check rather than a call

        self._strict = err_on_fail

        if extraneous_params := (priors.keys() - DEFAULT_PRIORS.keys()):
            raise ValueError(f"Invalid parameters: {extraneous_params}")

        # Fill in unspecified parameters with default priors bounds
        self.priors = {**DEFAULT_PRIORS, **priors}

        kw = {'transform': transform}

        # Fill the dict with actual priors objects
        for param in self.priors:

            if isinstance(self.priors[param], _PriorBase):
                # TODO need to add the correct transform to these
                continue

            else:
                # TODO Not happy with how nested the uniform args have to be
                prior_key, *args = self.priors[param]

                prior_key = prior_key.lower().replace('prior', '').strip()

                self.priors[param] = _PRIORS_MAP[prior_key](param, *args, **kw)


class _PriorBase:
    '''remember *logged*
    '''

    dependants = None

    def __repr__(self):
        return 'repr prior'

    def __str__(self):
        return 'str prior'

    @property
    def inv_mssg(self):
        try:
            return self._inv_mssg
        except AttributeError:
            return f"Invalid {self.__class__.__name__}"

    def _init_val(self, val):
        '''try to convert val to a float,
        else assume its a param and add to dependants'''
        try:
            # val is a specific number
            val = float(val)

        except ValueError:
            # val is a name of a param

            if val not in DEFAULT_INITIALS:
                raise ValueError(f'Invalid dependant parameter {val}')

            self.dependants.append(val)

        return val


class UniformPrior(_PriorBase):

    def __call__(self, param_val, *args, **kwargs):

        # get values for any dependant params
        lowers, uppers = zip(*self.bounds)

        lowers = map(lambda p: kwargs.get(p, p), lowers)
        uppers = map(lambda p: kwargs.get(p, p), uppers)

        # Check bounds are valid
        if not (valid := np.less_equal.outer(lowers, uppers)).all():
            # TODO get a good error message here
            self._inv_mssg = f'these bounds arent valid: {valid}'
            # self._inv_mssg = (f'{self.param}={param_val}, '
            #                   f'not {oper.__name__} {bnd}')
            return -np.inf

        # compute overall bounds, and loc/scale
        l_bnd, r_bnd = lowers.min(), uppers.max()
        loc, scale = l_bnd, r_bnd - l_bnd

        # evaluate the dist
        return np.log(self._caller(param_val, loc=loc, scale=scale))

    def __init__(self, param, edges, *, transform=False):
        '''
        param is the name, just for mssgs
        edges is a list of bounds
            eahc is either (lower bound, upper bound)
            bounds can be either a number or a param
        '''

        self._caller = stats.uniform.pdf if not transform else stats.uniform.ppf

        self.param = param

        self.bounds = []
        self.dependants = []

        for bounds in edges:

            if len(bounds) != 2:
                raise ValueError(f"Invalid edge: {bounds}")

            self.bounds.append((self._init_val(bnd) for bnd in bounds))


# TODO needs a much better name
class ArbitraryPrior(_PriorBase):
    '''Consists of a number of operation: value pairs which are evalutated.
    operation can be anything. '''

    def __call__(self, param_val, *args, **kwargs):

        L = 0.

        for oper, bnd in self._eval:

            try:
                check = oper(param_val, bnd)
            except TypeError:
                check = oper(param_val, kwargs[bnd])

            if not check:
                self._inv_mssg = (f'{self.param}={param_val}, '
                                  f'not {oper.__name__} {bnd}')
                return -np.inf

        return L

    def __init__(self, param, edges, *, transform=False):
        '''
        param is the name, just for mssgs
        edges is a list of bounds
            or (operation, param name)
            or (operation, bound)
        '''

        if transform:
            mssg = "ArbitraryPrior does not support `transform` or ppfs"
            raise NotImplementedError(mssg)

        self.param = param

        self._eval = []
        self.dependants = []

        for bounds in edges:

            if len(bounds) != 2:
                raise ValueError(f"Invalid edge: {bounds}")

            oper_str, bnd = bounds

            bnd = self._init_val(bnd)

            self._eval.append((_OPER_MAP[oper_str], bnd))

class GaussianPrior(_PriorBase):

    def __call__(self, param_val, *args, **kw):
        # TODO ah, what about this log, that probably shouldnt be there?
        return np.log(self._caller(param_val))

    def __init__(self, param, mu, sigma, *, transform=False):
        '''
        μ is a number
        σ is a number
        '''
        self.param = param

        self.mu, self.sigma = mu, sigma

        self.dist = stats.norm(loc=self.mu, scale=self.sigma)

        self._caller = self.dist.pdf if not transform else self.dist.ppf


class BoundedGaussianPrior(_PriorBase):
    # Could maybe use truncnorm?
    pass


class CromwellUniformPrior(_PriorBase):
    pass


DEFAULT_PRIORS = {
    'W0': UniformPrior('W0', [(3, 20)]),
    'M': UniformPrior('M', [(0.01, 10)]),
    'rh': UniformPrior('rh', [(0.5, 15)]),
    'ra': UniformPrior('ra', [(0, 5)]),
    'g': UniformPrior('g', [(0, 2.3)]),
    'delta': UniformPrior('delta', [(0.3, 0.5)]),
    's2': UniformPrior('s2', [(0, 15)]),
    'F': UniformPrior('F', [(1, 3)]),
    'a1': UniformPrior('a1', [(0, 6)]),
    'a2': UniformPrior('a2', [(0, 6), ('>=', 'a1')]),
    'a3': UniformPrior('a3', [(1.6, 6), ('>=', 'a2')]),
    'BHret': UniformPrior('BHret', [(0, 100)]),
    'd': UniformPrior('d', [(2, 8)])
}

_PRIORS_MAP = {
    "uniform": UniformPrior,
    "gaussian": GaussianPrior,
    "boundedgaussian": BoundedGaussianPrior,
    "cromwelluniform": CromwellUniformPrior,
}
