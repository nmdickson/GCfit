import numpy as np
from ..core.data import DEFAULT_INITIALS


__all__ = ['gaussian', 'RV_transform', 'gaussian_likelihood',
           'hyperparam_likelihood', 'hyperparam_effective',
           'compile_theta']


def compile_theta(func, fixed_initials=None):
    '''decorator for handling fixed parameters.
    Given theta, an array or dict, adds the fixed values and 
    '''
    import functools
    import inspect

    sig = inspect.signature(func)

    if fixed_initials is None:
        fixed_initials = {}

    @functools.wraps(func)
    def fixed_theta_decorator(*args, **kwargs):
        # TODO everything here with args/kwargs is iffy, might not always work

        # Combine the args and kwargs together, to ensure we can grab theta
        all_args = dict(zip(sig.parameters, args), **kwargs)

        try:

            theta = all_args['theta']

            # Unpack all theta values into dict, accounting for missing (fixed) keys
            if isinstance(theta, dict):
                theta = {**theta, **fixed_initials}

            else:
                param_sorter = list(DEFAULT_INITIALS).index

                variable_params = DEFAULT_INITIALS.keys() - fixed_initials.keys()
                params = sorted(variable_params, key=param_sorter)

                variable_theta = dict(zip(params, theta))

                theta = {**variable_theta, **fixed_initials}

                # Sort and finish with theta as an array
                theta = [theta[k] for k in sorted(theta, key=param_sorter)]

            # change the args theta to this theta (must be a better way...)
            if 'theta' in kwargs:
                kwargs['theta'] = theta
            else:
                args[list(sig.parameters).index('theta')] = theta

        except KeyError:
            pass

        return func(*args, **kwargs)

    return fixed_theta_decorator


# --------------------------------------------------------------------------
# Generic Distribution Helpers
# --------------------------------------------------------------------------


def gaussian(x, sigma, mu):
    '''Gaussian PDF, evaluated over `x` with mean `mu` and width `sigma`'''
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * (((x - mu) / sigma) ** 2))
    return norm * exponent


def RV_transform(domain, f_X, h, h_prime):
    '''Transformation of a random variable over a function :math:`g=h^{-1}`'''
    f_Y = f_X(h(domain)) * np.abs(h_prime(domain))
    return np.nan_to_num(f_Y)


# --------------------------------------------------------------------------
# Gaussian Likelihood
# --------------------------------------------------------------------------


def gaussian_likelihood(X_data, X_model, err):

    chi2 = (X_data - X_model)**2 / err**2

    err = (err / err.unit) if hasattr(err, 'unit') else err

    return -0.5 * np.sum(chi2 + np.log(err**2))


# --------------------------------------------------------------------------
# Gaussian Likelihood with Bayesian Hyperparameters
# --------------------------------------------------------------------------


def hyperparam_likelihood(X_data, X_model, err):
    '''compute the log likelihood of a Gaussian process with marginalized
    scaling hyperparameters (see Hobson et al., 2002)'''
    from scipy.special import gammaln

    n = (X_data.size / 2.) + 1
    chi2 = (X_data - X_model)**2 / err**2

    err = (err / err.unit) if hasattr(err, 'unit') else err

    return np.sum(
        np.log(2. / np.pi**(n - 1))
        + gammaln(n)
        - (n * (np.log(chi2 + 2)))
        - (0.5 * np.log(err**2))
    )


def hyperparam_effective(X_data, X_model, err):
    '''Compute the "effective" α_k scaling value for a given X_model
    (see Hobson et al., 2002), eq.44'''

    n_k = X_data.size
    chi2 = np.sum((X_data - X_model)**2 / err**2)

    return n_k / chi2
