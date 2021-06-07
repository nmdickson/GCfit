import numpy as np


__all__ = ['gaussian', 'RV_transform',
           'hyperparam_likelihood', 'hyperparam_effective', "norm_sample"]


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


# TODO: I think this only needed for the DM based calculation which is exclusive
# to 47 Tuc where we have the internal gas models (maybe Terzan 5 in future)

# TODO: this is broken in the same way the single-side trapezoid is, it sometimes
# overflows the bounds of the arrays. seems to only happen under emcee so we should
# try testing with just one thread and see

def norm_sample(x_array, y_array, target=1.0):
    """
    Lets us cut the domain of the Paz dists (or any other dist) before the numerical
    instability.

        Parameters:
            x_array (array): The domain of the probability distribution we want to
            trim.

            y_array (array): The probability distribution we want to trim.

        Returns:
            y_normed (array): The trimmed probability distribution with correct 
            normalization, values outside the proper normalization are set to zero.
    """
    # don't mess with original
    y_array = np.array(y_array)
    x_array = np.array(x_array)

    # tolerance
    eps = 1e-8

    # mid point of array
    mid = len(y_array) // 2

    #     print(f"len: {len(a_space)}, mid: {mid}")

    norm = 0.0
    offset = 1
    # TODO offset sometimes overflows the size of the array?
    # unclear why this is still overflowing
    while offset < (mid - 1):
        # try to catch the overflow?
        # TODO this really shouldn't be needed (fix this)
        if offset >= (mid - 1):
            offset = mid - 2
            break

        # get the spacing (getting each time should let us handle log spaced domains)
        delta = np.abs(x_array[mid + offset] - x_array[mid + offset - 1])
        #     print(delta)

        # get the slices - positive side
        P_a = y_array[mid + offset]
        P_b = y_array[mid + offset - 1]

        # negative side
        P_c = y_array[mid - offset]
        P_d = y_array[mid - offset + 1]

        # Integrate using trapezoid rule cumulatively
        norm += 0.5 * delta * (P_a + P_b)
        norm += 0.5 * delta * (P_c + P_d)

        #         print(f"offset: {offset}, norm: {norm}")

        # If converges, cut domain at this index
        if abs(target - norm) <= eps:
            break

        # If passes normalization, backup a step to cut domain close as possible
        elif norm >= target:
            offset -= 1
            break
        else:
            offset += 1
    #     print(offset)

    # set the rest to zero
    y_array[0: mid - offset] = 0
    y_array[mid + offset:] = 0

    return y_array


# --------------------------------------------------------------------------
# Bayesian Hyperparameters
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
