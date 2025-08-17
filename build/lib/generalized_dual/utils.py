from .core import GeneralizedDual
import numpy as np
import mpmath
from functools import wraps


def to_mpf_or_mpc(x):
    """
    Converts numeric types in np.ndarrays into mpmath numbers.

    Parameters
    ----------
    X : np.ndarray or numeric
        An object which to enrich with mpmath percision.

    Returns
    -------
    np.ndarray or numeric
        Enriched object with support for mpmath arbitrary percision.

    Examples
    --------
    >>> X = np.array([[1, 3 + 2j], [4, 2]])
    >>> to_mpf_or_mpc(X)
    """
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, (mpmath.mpf, mpmath.mpc)):
        return x
    elif isinstance(x, complex):
        return mpmath.mpc(x.real, x.imag)
    else:
        return mpmath.mpf(x)
    
    
def initialize(*vars, m):
    """
    Initialize GeneralizedDual variables ready for diferentiation.
    It assigns dimensions (number of variables) and maximum order of Taylor terms (`m`).
    Numbers in variables will be automatically enriched with mpmath arbitrary percision.

    Parameters
    ----------
    vars : list of same dimensions np.ndarrays or numbers.
    
    m  : maximum order of Taylor terms we are interested in.

    Returns
    -------
    A tuple of GeneralizedDual variables
        Result are GeneralDual variables ready for mixed partial derivatives up to order `m`.

    Examples
    --------
    >>> x, y, z, w = initialize(3, 2, 1, 3, m=9)
    >>> q, w = initialize(np.ndarray([1, 3, 4]), np.array([2, 7, 4]), m=5)
    """
    n = len(vars)
    one = vars[0] * 0 + 1  # Creates ones like var, keeping types and structure
    key0 = (0,) * n
    variables = []
    mpf_vec = np.vectorize(to_mpf_or_mpc)
    for i in range(n):
        var = mpf_vec(vars[i])
        terms = {key0: var}
        key = list(key0)
        key[i] = 1
        terms[tuple(key)] = one
        variables.append(GeneralizedDual(terms, n, m))
    if n == 1:
        return variables[0]
    return tuple(variables)


def disp(npndarray):
    """Displays results from diff, gradient, hessian, derivatives_along, etc., 
    with correct formatting for real and complex numbers."""
    def format_num(x):
        if isinstance(x, complex) or (hasattr(x, 'imag') and x.imag != 0):
            real_str = mpmath.mp.nstr(x.real, mpmath.mp.dps)
            imag_str = mpmath.mp.nstr(x.imag, mpmath.mp.dps)
            sign = '+' if x.imag >= 0 else '-'
            return f"{real_str} {sign} {imag_str}j"
        else:
            return mpmath.mp.nstr(x, mpmath.mp.dps)
    disp_vec = np.vectorize(format_num)
    print(disp_vec(npndarray))
    
    
   
def vectorize_func(f):
    """ Helper function for build_taylor """
    @wraps(f)
    def wrapper(vars):
        # Find broadcasting shape
        shapes = [np.shape(v) for v in vars if isinstance(v, np.ndarray)]
        if not shapes:
            return f(vars)

        # Determine broadcasted shape
        try:
            out_shape = np.broadcast_shapes(*shapes)
        except ValueError:
            raise ValueError("Input arrays have incompatible shapes for broadcasting.")

        # Create output array
        result = np.empty(out_shape, dtype=np.result_type(*[np.array(v).dtype for v in vars]))

        # Iterate over all broadcasted indices
        for idx in np.ndindex(out_shape):
            inputs = []
            for v in vars:
                if isinstance(v, np.ndarray):
                    # Use broadcasting rules
                    inputs.append(v[idx] if v.shape == out_shape else np.broadcast_to(v, out_shape)[idx])
                else:
                    inputs.append(v)
            result[idx] = f(inputs)
        return result
    return wrapper

def build_taylor(F, *centers, to_float=False):
    """
    Returns vectorized taylor functions for points around (X, Y, ...), where F is a GeneralizedDual
    obtained by evaluating f(X, Y, ...).

    Parameters
    ----------
    X : GeneralizedDual
        Obtained by evaluating dual_f(*centers)
        
    centers : list or array
        List of variables at which f was evaluated
        
    to_float : boolean, optional
        Converts mpmath objects in GeneralizedDual to floats. Good practice before plotting.

    Returns
    -------
    np.ndarray of functions or function
        Result has same dimensions as each center from centers.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> x = initialize(x, m=3)
    >>> F = sin(x)
    >>> taylor = build_taylor(F, X)
    >>> taylor[0, 1]([np.linspace(0, 1, 1000)]) # evaluates taylor with center in x=X[0, 1]=2 in many points at once
    
    >>> X, Y = 1, 2.3
    >>> x, y = initialize(x, y, m=5)
    >>> F = sin(x + y)*fresnelc(y)
    >>> taylor = build_taylor(F, X, Y, to_float=True)
    >>> taylor([np.linspace(2, 3, 10), None]) # Fix y variable into y=Y=2.3
    """
    if to_float:
        F = F.to_float()

    zero = F._default_zero()
    n_vars = F.n
    zero_flat = zero.flatten()
    shape = zero.shape

    centers_flat = [np.array(c).flatten() for c in centers]
    N = centers_flat[0].size

    funcs = []

    for p in range(N):
        center_p = [c[p] for c in centers_flat]

        @vectorize_func
        def func(vars, center_p=center_p, p=p):
            result = zero_flat[p] * 0  # zero of correct type

            for key, coeff in F.terms.items():
                term = 1
                for i in range(n_vars):
                    if vars[i] is None and key[i] != 0:
                        term = 0
                        break
                    elif vars[i] is not None:
                        term *= (vars[i] - center_p[i]) ** key[i]
                if term != 0:
                    result += coeff.flatten()[p] * term
            return result

        funcs.append(func)
    if shape == (): return (np.array(funcs).reshape(shape)).item()
    return np.array(funcs).reshape(shape)