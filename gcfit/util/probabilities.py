import numpy as np
from .units import QuantitySpline

__all__ = ['gaussian', 'RV_transform', 'gaussian_likelihood',
           'hyperparam_likelihood', 'hyperparam_effective', 'div_error',
           'ModelParameters', 'trim_peaks', 'find_intersections']

# --------------------------------------------------------------------------
# Generic Distribution Helpers
# --------------------------------------------------------------------------


def gaussian(x, sigma, mu):
    '''Gaussian PDF, evaluated over `x` with mean `mu` and width `sigma`.'''
    norm = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-0.5 * (((x - mu) / sigma) ** 2))
    return norm * exponent


def RV_transform(domain, f_X, h, h_prime):
    '''Transformation of a random variable over a function :math:`g=h^{-1}`.'''
    f_Y = f_X(h(domain)) * np.abs(h_prime(domain))
    return np.nan_to_num(f_Y)


def find_intersections(y, x, value):
    '''Find the intersections of a function with a given value.'''
    spl = QuantitySpline(x, y - value, k=3, ext=2)
    return spl.roots()


# --------------------------------------------------------------------------
# Gaussian Likelihood
# --------------------------------------------------------------------------


def gaussian_likelihood(X_data, X_model, err):
    '''Gaussian log-likelihood function.'''

    chi2 = (X_data - X_model)**2 / err**2

    err = (err / err.unit) if hasattr(err, 'unit') else err

    return -0.5 * np.sum(chi2 + np.log(err**2))


# --------------------------------------------------------------------------
# Gaussian Likelihood with Bayesian Hyperparameters
# --------------------------------------------------------------------------


def hyperparam_likelihood(X_data, X_model, err):
    '''Gaussian log-likelihood function with marginalized hyperparameters.

    (see Hobson et al., 2002).
    '''

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
    '''Compute the "effective" α_k scaling value for a given X_model.

    (see Hobson et al., 2002; eq.44).
    '''

    n_k = X_data.size
    chi2 = np.sum((X_data - X_model)**2 / err**2)

    return n_k / chi2


# --------------------------------------------------------------------------
# Gaussian error propagation
# --------------------------------------------------------------------------


def div_error(a, a_err, b, b_err):
    '''Gaussian error propagation for division of two quantities with errors.'''
    return abs(a / b) * np.sqrt((a_err / a) ** 2 + (b_err / b) ** 2)


# --------------------------------------------------------------------------
# Theta handling helpers
# --------------------------------------------------------------------------


class ModelParameters:
    '''Helper class for parsing Model parameters for fitting

    In order to make the handling of free parameters during model fitting more
    flexible, extensible and secure, this class handles all arguments passed
    to the initialization of the desired model class. Based on the function
    signature of `__init__`, it determines which parameters are valid, and
    provides methods for easily transforming an array of free parameter
    values to a useable function signature.

    This class eliminates the need for having fixed sets of possible free
    model parameters, allowing each different fitting run to be its own.

    Models can be initialized by calling:
    `Model(*modelparams.build_args(θ).args, **modelparams.build_args(θ).kwargs)`

    Parameters
    ----------
    free_params : tuple of str
        The names of which parameters will be freely varied. The order of this
        must be matched by whatever data is passed to the constructor methods.
        The parameters must be valid (i.e. appear directly in the signature)
        for the relevant Model class, and must be parameters which accept a
        single scalar value.

    model_kwargs : dict, optional
        Values of any model parameters which will not be freely varying. Any
        parameters given will override the defaults of the Model class. These
        parameters are not restricted to scalar arguments. Any model parameters
        which do not have any defaults must be provided here, if they are
        not given in `free_params`.

    observations : gcfit.Observations, optional
        The `Observations` instance to be provided to the Model class. Appears
        here directly as it is handled specially. Should not be given in
        `model_kwargs`.

    evolved : bool, optional
        Whether to use the `Model` or `EvolvedModel` class. Important for
        determining the correct valid parameters. Defaults to `Model` class.

    transforms : dict, optional
        Optional dictionaries of scalar->scalar functions which will be applied
        to the relevant free parameters when given to the one of the argument
        building methods. Necessary if, for example, you wish to vary the log
        of a parameter, rather than the parameter itself.

    sympy_transforms : bool, optional
        Flag allowing the passed transforms to be strings parseable by sympy,
        rather than callable functions.

    compatibility_transforms : bool, optional
        By default, use some transforms that have always been used in the past,
        mostly for backwards compatibility. Namely, `M * 1e6` and `10**ra`.
    '''

    def __init__(self, free_params: tuple[str, ...],
                 model_kwargs: dict | None = None, observations=None,
                 evolved: bool = False,
                 transforms: dict | None = None, sympy_transforms: bool = False,
                 compatibility_transforms: bool = True):
        import inspect
        from ..core.data import Model, EvolvedModel

        self._signature = inspect.signature(EvolvedModel if evolved else Model)

        self._all_params = list(self._signature.parameters.values())

        self._reqd_params = [prm for prm in self._all_params
                             if (prm.default is prm.empty)
                             and (prm.kind is not prm.VAR_KEYWORD)]

        if model_kwargs is None:
            model_kwargs = dict()

        if extra := (set(free_params) - {prm.name for prm in self._all_params}):
            raise ValueError(f"Invalid parameters: {extra}")

        # Some params don't have defaults. It's not that they must be free, but
        # should be free or specifically given (and thus fixed) (M, W0, rh)
        given_params = set(free_params) | model_kwargs.keys()
        if miss := ({prm.name for prm in self._reqd_params} - given_params):
            raise ValueError(f"Missing required parameters: {miss}")

        if (ndim := len(free_params)) < 1:
            raise ValueError("Must have at least one free parameter.")

        # Check for sympy transforms (for fun)
        if transforms is not None and sympy_transforms:
            # I don't love the idea of making sympy the default, but they're
            # also the only kind of transforms I could easily store and recreate
            import sympy
            for prm, func in transforms.items():
                if not callable(func):
                    transforms[prm] = sympy.lambdify(prm, func)

        # For backwards compatibility, add in the historically used transforms
        if transforms is None and compatibility_transforms:
            transforms = {'M': lambda M: 1e6 * M,
                          'M0': lambda M: 1e6 * M,
                          'ra': lambda ra: 10**ra}

        # TODO check that params are not in both free and fixed, and model_kw

        self.free_params = free_params
        self.model_kwargs = model_kwargs
        self.ndim = ndim
        self._obs = observations  # explicit and special
        self.transforms = transforms

        fixedargs = self._signature.bind_partial(observations=self._obs,
                                                 **self.model_kwargs)
        fixedargs.apply_defaults()
        self.fixed_params = {k: v for k, v in fixedargs.arguments.items()
                             if k not in self.free_params}

    def label_theta(self, theta):
        '''Turn theta into a dict with correct parameter names'''
        try:
            return dict(zip(self.free_params, theta, strict=True))
        except ValueError as err:
            mssg = (f'Given theta (size {len(theta)}) does not match number '
                    f'of free parameters (size {self.ndim})')
            raise ValueError(mssg) from err

    def build_args(self, theta, *, return_dict=False, apply_transforms=True):
        '''Build the Model arguments required to init a model with theta.'''

        if isinstance(theta, dict):

            if theta.keys() != set(self.free_params):
                mssg = 'Given `theta` dict does not match the free parameters'
                raise ValueError(mssg)

            free_theta = theta

        else:

            free_theta = self.label_theta(theta=theta)

        if apply_transforms and (self.transforms is not None):
            free_theta = {prm: self.transforms.get(prm, lambda v: v)(val)
                          for prm, val in free_theta.items()}

        kwargs = free_theta | self.model_kwargs | {'observations': self._obs}

        boundargs = self._signature.bind(**kwargs)

        # Fill in given params with Model defaults
        boundargs.apply_defaults()

        if return_dict:
            return boundargs.arguments
        else:
            return boundargs


# --------------------------------------------------------------------------
# Distribution helpers
# --------------------------------------------------------------------------


def trim_peaks(az_domain, Paz):
    '''Remove all "peaks" from distribution while maintaining normalization.'''

    from scipy.signal import find_peaks
    from scipy.integrate import trapezoid

    Paz = Paz.copy()

    while (area := trapezoid(x=az_domain.value, y=Paz.value)) >= 0.98:

        peaks, _ = find_peaks(Paz, height=0, threshold=1e5, width=1)

        # break if all peaks have been eliminated
        if peaks.size == 0:
            break

        # set the peaks to 0 probability
        Paz[peaks] = 0

    # re-normalize:
    Paz /= area

    # return trimmed Paz
    return Paz

# --------------------------------------------------------------------------
# Nested Sampling helpers
# --------------------------------------------------------------------------


def plateau_weight_function(results, args=None, return_weights=False):
    '''Weight and logl-bound function which adjusts to handle jumping weights

    A new version of `dynesty.dynamicsampler.weight_function` which hopefully
    handles more difficult-to-converge posteriors better.

    In certain cases posterior distributions may possess small "plateaus" in
    their likelihoods near the peak of their distributions. When this occurs
    the initial sampling runs are not able to reach very low `dlogz` values
    before reaching and saturating the live points on this plateau, and
    entering a constant impossible search for high likerlihoods.

    When this happens, although the initial sampling batch can be stopped at
    that stage with little issue, the importance weights curve can become
    difficult, as the final added live points cover a large volume, and cause
    the weights to "jump" upwards, sometimes even dwarfing the typical peak
    of the curve. The default `weight_function` cannot easily handle these
    cases, and will cause the first dynamic batch to get stuck.

    This happens due to two factors. First, the usual choice of bounds as
    80% of the peak may, in the case of large jumps, lead to obscenely small
    bounds at the highest likelihood. Second, even if using a small fraction,
    the rightmost bound may initially be set to +inf. This means the stopping
    condition used by the first dynamic batch is only the `dlogz<0.0001`
    condition, which as mentioned before, is exceedingly difficult to reach
    given this posterior.

    To handle these problems, this new function first checks if the bounds
    found using the given `maxfrac` are overflowing on the right, and if
    so explicitly reduces the `maxfrac` to a quarter of it's original value and
    tries again. Secondly, when overflowing on the right, the maximum bound
    is not set as +inf, but rather the highest logl value. This will still
    require the sampler to probe the space around this "plateau", while still
    allowing for it to end before reaching extremely low `dlogz`
    (this may not be valid if the plateau is a perfectly flat plateau, however).

    Note
    ----
    This should not be required for `dynesty >= 2.1.0`, which has a more
    explicit (and probably correct) handling of plateaus within the likelihood
    sampling itself

    '''
    # TODO this^ explanation should be in some docs, not the docstring

    def compute_bounds(bound_frac):
        from dynesty.dynamicsampler import compute_weights

        zweight, pweight = compute_weights(results)

        # Compute combined weights.
        weight = (1. - pfrac) * zweight + pfrac * pweight

        # Compute logl bounds
        bounds = np.nonzero(weight > bound_frac * np.max(weight))[0]

        # we pad by lpad on each side (2*lpad total)
        # if this brings us outside the range on on side, I add it on another
        bounds = (bounds[0] - lpad, bounds[-1] + lpad)
        nsamps = weight.size

        # A temporary test to handle weight jumps due to plateaus
        # TODO emphasis on the *temporary* here, this is ugly (10)
        max_bound = (nsamps - 1) - results.samples_n[0]

        # overflow on the RHS, so we move the left side
        if bounds[1] >= max_bound:
            bounds = [bounds[0] - (bounds[1] - max_bound), max_bound]

        return bounds, weight

    # Initialize hyperparameters.
    if args is None:
        args = dict()

    pfrac = args.get('pfrac', 0.8)
    if not 0. <= pfrac <= 1.:
        mssg = f"The provided `pfrac` {pfrac} is not between 0. and 1."
        raise ValueError(mssg)

    maxfrac = args.get('maxfrac', 0.8)
    if not 0. < maxfrac <= 1.:
        mssg = f"The provided `maxfrac` {maxfrac} is not between 0. and 1."
        raise ValueError(mssg)

    lpad = args.get('pad', 1)
    if lpad < 0:
        raise ValueError(f"`lpad` {lpad} is less than zero.")

    # compute the bounding indices based on the weights
    bounds, weight = compute_bounds(bound_frac=maxfrac)

    # compute the logl values of said bounds
    logl = results.logl

    # If this is overflowing on the right, recompute with a much lower fraction
    # Won't change the bound, but is a hacky way to hopefully get a better LHS
    # TODO must be better way to determine if we've gone left of wt peak or not
    if bounds[1] == (logl.size - 1) - results.samples_n[0]:
        bounds, weight = compute_bounds(bound_frac=0.25 * maxfrac)

    # if we overflow on the leftside we set the edge to -inf and expand the RHS
    if bounds[0] < 0:
        logl_min = -np.inf
        logl_max = logl[min(bounds[1] - bounds[0], logl.size - 1)]

    else:
        logl_min, logl_max = logl[bounds[0]], logl[bounds[1]]

    # return bounds and weights
    if return_weights:
        return (logl_min, logl_max), weight

    else:
        return (logl_min, logl_max)
