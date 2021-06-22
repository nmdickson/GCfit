import numpy as np


__all__ = ['gaussian', 'RV_transform', 'gaussian_likelihood',
           'hyperparam_likelihood', 'hyperparam_effective', 'div_error']


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


# --------------------------------------------------------------------------
# Gaussian error propagation.
# --------------------------------------------------------------------------
def div_error(a, a_err, b, b_err):
    """
    Compute Gaussian error propagation for a÷b.
    """
    f = a / b
    return abs(f) * np.sqrt((a_err / a) ** 2 + (b_err / b) ** 2)
