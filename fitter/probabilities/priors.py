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
    """Container class representing the prior (logged) likelihoods,
    to be called on θ and (added) to the log likelihood"""

    def __call__(self, theta, *, return_indiv=False):
        '''return the total prior likelihood given by theta'''
        if not isinstance(theta, dict):
            theta = dict(zip(DEFAULT_INITIALS, theta))

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

    def __init__(self, priors, transform=False, *,
                 logged=True, err_on_fail=False):
        '''
        priors: dict where key is a parameter, and eavh value is either a
        `*Prior` object, or ["name of class", *args for that class]

        transform : bool, use if doing nested sampling, switches to ppf
        err_on_fail : bool, if the likelihood is <= 0 will raise an error
        '''

        self._log = logged
        self._strict = err_on_fail

        if extraneous_params := (priors.keys() - DEFAULT_PRIORS.keys()):
            raise ValueError(f"Invalid parameters: {extraneous_params}")

        # Fill in unspecified parameters with default priors bounds
        self.priors = {**DEFAULT_PRIORS, **priors}

        # TODO might want to change how we define a "transform" priors
        #   cause it's not a prior anymore, its a prior_transform
        kw = {'transform': transform}
        if transform:
            self._log = False

        # Fill the dict with actual priors objects
        for param in self.priors:

            if isinstance(self.priors[param], _PriorBase):

                if self.priors[param]._transform is not transform:
                    mssg = (f"Prior {self.priors[param]} already "
                            f"initialized without {transform=}")
                    raise RuntimeError(mssg)

            else:
                # TODO Not happy with how nested the uniform args have to be
                prior_key, *args = self.priors[param]

                prior_key = prior_key.lower().replace('prior', '').strip()

                self.priors[param] = _PRIORS_MAP[prior_key](param, *args, **kw)


class _PriorBase:

    dependants = None
    params = None

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

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'("{self.param}", {self.bounds}, transform={self._transform})')

    def __call__(self, param_val, *args, **kwargs):

        # get values for any dependant params
        lowers, uppers = zip(*self.bounds)

        # TODO the error if the needed param is not in kwargs is *not* nice
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

            return 0.

        # compute overall bounds, and loc/scale
        l_bnd, r_bnd = lowers.min(), uppers.max()
        loc, scale = l_bnd, r_bnd - l_bnd

        # evaluate the dist
        return self._caller(param_val, loc=loc, scale=scale)

    def __init__(self, param, edges, *, transform=False):
        '''
        param is the name, just for mssgs
        edges is a list of bounds
            eahc is either (lower bound, upper bound)
            bounds can be either a number or a param
        '''

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
    '''Consists of a number of operation: value pairs which are evalutated.
    operation can be anything. '''

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

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'("{self.param}", {self.mu}, {self.sigma}, '
                f'transform={self._transform})')

    def __call__(self, param_val, *args, **kw):
        return self._caller(param_val)

    def __init__(self, param, mu, sigma, *, transform=False):
        '''
        μ is a number
        σ is a number
        '''

        self._transform = transform
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
    'W0': ('uniform', [(3, 20)]),
    'M': ('uniform', [(0.01, 10)]),
    'rh': ('uniform', [(0.5, 15)]),
    'ra': ('uniform', [(0, 5)]),
    'g': ('uniform', [(0, 2.3)]),
    'delta': ('uniform', [(0.3, 0.5)]),
    's2': ('uniform', [(0, 15)]),
    'F': ('uniform', [(1, 3)]),
    'a1': ('uniform', [(0, 6)]),
    # TODO be careful to make sure these kinds of priors work for the ppfs:
    #   need tp check against the transformed "a1" not the [0,1] kind
    'a2': ('uniform', [(0, 6), (0, 'a1')]),
    # 'a3': ('uniform', [(1.6, 6), (0, 'a2')]),
    'a3': ('uniform', [(0, 6), (0, 'a2')]),
    # TODO might want to drastically decrease this upper bound for nest-samp.
    'BHret': ('uniform', [(0, 100)]),
    'd': ('uniform', [(2, 8)])
}

_PRIORS_MAP = {
    "uniform": UniformPrior,
    "gaussian": GaussianPrior,
    "boundedgaussian": BoundedGaussianPrior,
    "cromwelluniform": CromwellUniformPrior,
}
