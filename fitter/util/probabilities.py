import numpy as np


__all__ = ['gaussian', 'RV_transform',
           'hyperparam_likelihood', 'hyperparam_effective']


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
    f_Y = f_X(h(domain)) * h_prime(domain)
    return np.nan_to_num(f_Y)


# --------------------------------------------------------------------------
# Bayesian Hyperparameters
# --------------------------------------------------------------------------


def hyperparam_likelihood(x_data, x_model, err):
    '''compute the log likelihood of a Gaussian process with marginalized
    scaling hyperparameters (see Hobson et al., 2002)'''
    from scipy.special import gammaln

    n = (x_data.size / 2.) + 1
    chi2 = (x_data - x_model)**2 / err**2

    err = (err / err.unit) if hasattr(err, 'unit') else err

    return np.sum(
        np.log(2. / np.pi**(n - 1))
        + gammaln(n)
        - (n * (np.log(chi2 + 2)))
        - (0.5 * np.log(err**2))
    )


def hyperparam_effective(x_data, x_model, err):
    '''Compute the "effective" α_k scaling value for a given x_model
    (see Hobson et al., 2002), eq.44'''

    n_k = x_data.size
    chi2 = np.sum((x_data - x_model)**2 / err**2)

    return n_k / chi2
