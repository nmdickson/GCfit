from ..core.data import DEFAULT_INITIALS

import logging


DEFAULT_PRIORS = {
    'W0': [(3, 20)],
    'M': [(0.01, 10)],
    'rh': [(0.5, 15)],
    'ra': [(0, 5)],
    'g': [(0, 2.3)],
    'delta': [(0.3, 0.5)],
    's2': [(0, 15)],
    'F': [(1, 3)],
    'a1': [(0, 6)],
    'a2': [(0, 6), ('>=', 'a1')],
    'a3': [(1.6, 6), ('>=', 'a2')],
    'BHret': [(0, 100)],
    'd': [(2, 8)],
}


class Priors:
    """Container class representing the prior likelihoods, to be called on θ"""

    def __call__(self, theta):
        '''return the total prior likelihood given by theta'''
        if not isinstance(theta, dict):
            theta = dict(zip(DEFAULT_INITIALS, theta))

        inv = []
        res = 1

        for key, priors in self._eval.items():
            for oper, val in priors:

                try:
                    check = oper(theta[key], val)
                except TypeError:
                    check = oper(theta[key], theta[val])

                if not check:
                    inv.append(f'{key}={theta[key]}, not {oper.__name__} {val}')
                    res *= 0

        if inv:
            mssg = f"Theta failed priors checks: {'; '.join(inv)}"
            if self._strict:
                raise ValueError(mssg)
            else:
                logging.debug(mssg)

        return res

    def __init__(self, parameters, kind='uniform', *, err_on_fail=False):
        '''parameters: the parameters necessary for a given type of prior
                        should be a dict of keys from theta
        '''
        # TODO may be the spot to set up an initial check rather than a call

        import operator
        oper_map = {
            '<': operator.lt, 'lt': operator.lt,
            '<=': operator.le, 'le': operator.le,
            '>=': operator.ge, 'ge': operator.ge,
            '>': operator.gt, 'gt': operator.gt,
            '=': operator.eq, '==': operator.eq, 'eq': operator.eq,
            '!=': operator.ne, 'ne': operator.ne,
        }

        self.kind = kind

        self._strict = err_on_fail

        # Fill in unspecified parameters with default priors bounds
        parameters = {**DEFAULT_PRIORS, **parameters}

        if kind == 'uniform':
            # parameters is a dict of [bounds, dep_bounds]

            self._eval = {}

            for key, val in parameters.items():

                if key not in DEFAULT_PRIORS:
                    mssg = f'Invalid parameter: {key}'
                    raise ValueError(mssg)

                self._eval[key] = []

                for bounds in val:

                    # dependant parameter bounds
                    if isinstance(bounds[0], str):
                        oper_str, dep_key = bounds

                        if dep_key not in DEFAULT_PRIORS:
                            mssg = (f'Invalid dependant parameter for {key}:'
                                    f'{oper_str} {dep_key}')
                            raise ValueError(mssg)

                        self._eval[key].append((oper_map[oper_str], dep_key))

                    # normal bounds
                    else:
                        lower_bnd, upper_bnd = bounds

                        if lower_bnd is not None:
                            self._eval[key].append((oper_map['>='], lower_bnd))

                        if upper_bnd is not None:
                            self._eval[key].append((oper_map['<='], upper_bnd))

        else:
            raise NotImplementedError
