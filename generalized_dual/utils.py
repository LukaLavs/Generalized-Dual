from .core import GeneralizedDual
import numpy as np
import mpmath

def to_mpf_or_mpc(x):
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, (mpmath.mpf, mpmath.mpc)):
        return x
    elif isinstance(x, complex):
        return mpmath.mpc(x.real, x.imag)
    else:
        return mpmath.mpf(x)
    
def initialize(*vars, m):
    """ initializes variables 
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

