from ..core.data import DEFAULT_INITIALS

import numpy as np

import logging
import operator


__all__ = ["Priors", "DEFAULT_PRIORS", "UniformPrior", "GaussianPrior",
           "BoundedGaussianPrior", "CromwellUniformPrior"]


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

            # evaluate prior
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

    def __init__(self, priors, *, err_on_fail=False):
        '''
        priors: dict where key is a parameter, and eavh value is either a
        `*Prior` object, or ["name of class", *args for that class]

        err_on_fail : bool, if the likelihood is <= 0 will raise an error
        '''
        # TODO may be the spot to set up an initial check rather than a call

        self._strict = err_on_fail

        if extraneous_params := (DEFAULT_PRIORS.keys() - priors.keys()):
            raise ValueError(f"Invalid parameters: {extraneous_params}")

        # Fill in unspecified parameters with default priors bounds
        self.priors = {**DEFAULT_PRIORS, **priors}

        # Fill the dict with actual priors objects
        for param in self.priors:

            if isinstance(self.priors[param], _PriorBase):
                continue

            else:
                prior_key, *args = self.priors[param]

                prior_key = prior_key.lower().replace('prior', '').strip()

                self.priors[param] = _PRIORS_MAP[prior_key](*args)


class _PriorBase:
    '''remember *logged*
    '''

    def __repr__(self):
        return 'repr prior'

    def __str__(self):
        return 'str prior'

    @property
    def inv_mssg(self):
        return self._inv_mssg


class UniformPrior(_PriorBase):
    '''
    -inf if <= 0, else 0
    '''

    def __call__(self, param_val):

        L = 0.

        for oper, bnd in self._eval:

            try:
                check = oper(param_val, bnd)
            except TypeError:
                # TODO the parameter dependant one, need to get the bnd from θ
                #   But don't really love the idea of always passing theta to
                #   everything
                #   Could instead have some sort of flag that tells Priors to
                #   pass certain values in?
                check = oper(param_val, theta[bnd])

            if not check:
                L = -np.inf

        return L

    def __init__(self, edges):
        '''
        edges is a list of bounds
            eahc is either (lower bound, upper bound)
            or (operation, param name)
            or (operation, bound)
        '''

        self._eval = []

        for bounds in edges:

            if len(bounds) != 2:
                raise ValueError(f"Invalid edge: {bounds}")

            # first element is operation
            if isinstance(bounds[0], str):
                oper_str, bnd = bounds

                try:
                    # bound is a specific element
                    bnd = float(bnd)

                except ValueError:
                    # bound is a name of a param

                    if bnd not in DEFAULT_PRIORS:
                        raise ValueError(f'Invalid dependant parameter {bnd}')

                self._eval.append((_OPER_MAP[oper_str], bnd))

            # (lower bound, upper bound)
            else:
                lower_bnd, upper_bnd = bounds

                if lower_bnd is not None:
                    self._eval.append((_OPER_MAP['>='], lower_bnd))

                if upper_bnd is not None:
                    self._eval.append((_OPER_MAP['<='], upper_bnd))


class GaussianPrior(_PriorBase):
    pass


class BoundedGaussianPrior(_PriorBase):
    pass


class CromwellUniformPrior(_PriorBase):
    pass


DEFAULT_PRIORS = {
    'W0': UniformPrior([(3, 20)]),
    'M': UniformPrior([(0.01, 10)]),
    'rh': UniformPrior([(0.5, 15)]),
    'ra': UniformPrior([(0, 5)]),
    'g': UniformPrior([(0, 2.3)]),
    'delta': UniformPrior([(0.3, 0.5)]),
    's2': UniformPrior([(0, 15)]),
    'F': UniformPrior([(1, 3)]),
    'a1': UniformPrior([(0, 6)]),
    'a2': UniformPrior([(0, 6), ('>=', 'a1')]),
    'a3': UniformPrior([(1.6, 6), ('>=', 'a2')]),
    'BHret': UniformPrior([(0, 100)]),
    'd': GaussianPrior(mu=4, sigma=1),
}

_PRIORS_MAP = {
    "uniform": UniformPrior,
    "gaussian": GaussianPrior,
    "boundedgaussian": BoundedGaussianPrior,
    "cromwelluniform": CromwellUniformPrior,
}
