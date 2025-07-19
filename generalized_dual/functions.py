from .core import GeneralizedDual
from .utils import initialize
import numpy as np
import mpmath

# == FUNCTIONS ==

# --- Useful Functions ---

def dual_sum(X):
    """
    Compute the Sum function for a list of GeneralizedDual inputs.

    Parameters
    ----------
    X : list of GeneralizedDual or numeric objects
        List of objects to sum. First object must be GeneralizedDual.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> L = list(dual_pow(x, i) for i in range(1, 5))
    >>> dual_sum(L)  # where x is a GeneralizedDual instance
    """
    p = X
    for _ in range(1, X.m):
        p += X
    return p

def prod(X):
    """
    Compute the Exponential function for a list of GeneralizedDual inputs.

    Parameters
    ----------
    X : list of GeneralizedDual or numeric objects
        List of objects to multiply. First object must be GeneralizedDual.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> L = list(nroot(x, i) for i in range(1, 10))
    >>> prod(L)  # where x is a GeneralizedDual instance
    """
    p = X
    for _ in range(1, X.m):
        p *= X
    return p

def dual_abs(X):
    """
    Compute the Absolute Value or Complex Norm of a GeneralizedDual object.

    For complex-valued inputs, returns the norm √(Re(X)² + Im(X)²).
    For real inputs, returns the absolute value.

    Parameters
    ----------
    X : GeneralizedDual
        Input object, possibly complex-valued.

    Returns
    -------
    GeneralizedDual
        Absolute value or norm of `X`.

    Examples
    --------
    >>> abs_val = dual_abs(x)  # where x is GeneralizedDual (real or complex)
    >>> abs_val.diff((1,))
    """
    def has_complex_numbers(X):
        for arr in X.terms.values():
            has_complex = np.vectorize(lambda x: False if x.imag == 0 else True)(arr)
            if np.any(has_complex):
                return True
        return False
    if has_complex_numbers(X):
        X_real_terms = {}
        X_imag_terms = {}
        for key, val in X.terms.items():
            X_real_terms[key] = np.vectorize(lambda x: x.real)(val)
            X_imag_terms[key] = np.vectorize(lambda x: x.imag)(val)
        X_real = GeneralizedDual(X_real_terms, X.n, X.m)
        X_imag = GeneralizedDual(X_imag_terms, X.n, X.m)
        return sqrt(X_imag**2 + X_real**2)
    return X if X >= 0 else - X

def sign(X):
    """
    Compute the Sign function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual, not complex
        Point at which to evaluate the Sign function.

    Returns
    -------
    int
        1 if sign is positive, -1 if negative and 0 otherwise.

    Examples
    --------
    >>> sign(x)  # where x is a GeneralizedDual instance
    """
    return 1 if X > 0 else (-1 if X < 0 else 0)

# --- Log functions ---

def exp(X):
    """
    Compute the Exponential function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Exponential function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> exp(x)  # where x is a GeneralizedDual instance
    """
    f0, f_hat = X._decompose()
    result = GeneralizedDual._constant(1, size_like=X)
    base = GeneralizedDual._constant(1, size_like=X)
    for k in range(1, X.m + 1):
        base = base * f_hat
        result += base / mpmath.factorial(k)
    return result * np.vectorize(mpmath.exp)(f0)

def log(X, B=None):
    """
    Compute the Base 10 Logarithmic function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Logarithmic function.
        
    B:  GeneralizedDual or numeric
        Point at which to evaluate the Logarithmic function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.
        
    Notes
    -----
    If X is numeric, it is easy to convert it to GeneralizedDual class with:
    >>> X_new = GeneralizedDual._constant(X, size_like=T) # where T is same size as dual we need.
        # For percision purpuse, it is recomended that X is an array of mpmath numbers.

    Examples
    --------
    >>> log(x, b)  # where x and b are GeneralizedDual instances
    """
    if isinstance(B, int):
        B = GeneralizedDual._constant(B, size_like=X)
    if B:
        if B > 0 and B != 1:
            return log(X) / log(B)
    f0, f_hat = X._decompose()
    f_inv = GeneralizedDual._safe_inversion(f0)
    base = f_hat * f_inv
    result = GeneralizedDual._constant(0, size_like=X)
    term = base
    for k in range(1, X.m + 1):
        result += ((-1) ** k) * term / mpmath.mp.mpf(k)
        term = term * base
    return - result + np.vectorize(mpmath.log)(f0)

def log2(X):
    """
    Compute the Base 2 Logarithmic function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Base 2 Logarithmic function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> log2(x)  # where x is a GeneralizedDual instance
    """
    return log(X, 2)

def log10(X):
    """
    Compute the Base 10 Logarithmic function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Base 10 Logarithmic function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> log10(x)  # where x is a GeneralizedDual instance
    """
    return log(X, 10)

# --- X^Y functions ---

def dual_pow(X, A):
    """
    Compute the Power function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual or numeric
        Point at which to evaluate the Power function.
        
    A : GeneralizedDual or numeric
        Point at which to evaluate the Power function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

       Notes
    -----
    If X and A are numeric, it is easy to convert them to GeneralizedDual class with:
    >>> X_new = GeneralizedDual._constant(X, size_like=T) # where T is same size as dual we need.
        # For percision purpuse, it is recomended that X is an array of mpmath numbers.


    Examples
    --------
    >>> dual_pow(x, 2)      # x is GeneralizedDual, scalar exponent
    >>> dual_pow(3, a)      # scalar base, a is GeneralizedDual exponent
    >>> dual_pow(x, a)      # both GeneralizedDual
    """
    if isinstance(A, int):
        result = GeneralizedDual._constant(1, size_like=X)
        for _ in range(A):
            result = result * X
        return result
    if isinstance(A, GeneralizedDual):
        if not isinstance(X, GeneralizedDual):
            return exp(A * np.vectorize(mpmath.log)(X))
        return exp(A * log(X))
    A = np.vectorize(lambda x: mpmath.mpf(x))(A)
    def binomial(A, k):
        result = 1
        for i in range(k):
            result *= (A - i)
        return result / mpmath.factorial(k)
    f0, f_hat = X._decompose()
    f_inv = GeneralizedDual._safe_inversion(f0)
    base = f_hat * f_inv
    result = GeneralizedDual._constant(0, size_like=X)
    term = GeneralizedDual._constant(1, size_like=X)
    for k in range(X.m + 1):
        if k > 0:
            term = term * base
        result += binomial(A, k) * term
    return result * (f0 ** A) 

def sqrt(X):
    """
    Compute the Square Root function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Square Root function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> sqrt(x)  # where x is a GeneralizedDual instance
    """
    return dual_pow(X, mpmath.mpf("1/2"))

def cbrt(X):
    """
    Compute the Cube Root function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Cube Root function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> cbrt(x)  # where x is a GeneralizedDual instance
    """
    return sign(X) * dual_pow(dual_abs(X), mpmath.mpf("1/3"))

def nroot(X, n):
    """
    Compute the Nth Root function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Nth Root function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> nroot(x, 3)  # cube root of GeneralizedDual x
    >>> nroot(8, 3)  # scalar input returns scalar cube root
    """
    power = f"1/{n}"
    if n % 2 == 0:
        return dual_pow(X, mpmath.mpf(power))
    else: 
        return sign(X) * dual_pow(X, mpmath.mpf(power))
    
# --- Trig ---

def sin(X):
    """
    Compute the Sine function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Sine function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> sin(x)  # where x is a GeneralizedDual instance
    """
    f0, f_hat = X._decompose()
    sinf0 = np.vectorize(mpmath.sin)(f0)
    cosf0 = np.vectorize(mpmath.cos)(f0)
    result = GeneralizedDual._constant(0, size_like=X)
    base_power = GeneralizedDual._constant(1, size_like=X)
    for k in range(X.m + 1):
        deg = 2 * k
        if deg <= X.m:
            term = (base_power) / mpmath.factorial(deg)
            result += ((-1) ** k) * term * sinf0
        deg = 2 * k + 1
        base_power *= f_hat
        if deg <= X.m:
            term = (base_power) / mpmath.factorial(deg)
            result += ((-1) ** k) * term * cosf0
        base_power *= f_hat
    return result

def cos(X):
    """
    Compute the Cosine function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Cosine function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> cos(x)  # where x is a GeneralizedDual instance
    """
    return sin(X + mpmath.pi/2)

def tan(X):
    """
    Compute the Tangent function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Tangent function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> tan(x)  # where x is a GeneralizedDual instance
    """
    return sin(X) / cos(X)

def cot(X):
    """
    Compute the Cotangent function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Cotangent function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> cot(x)  # where x is a GeneralizedDual instance
    """
    return cos(X) / sin(X)

def atan(X):
    """
    Compute the Inverse Tangent function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Tangent function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> atan(x)  # where x is a GeneralizedDual instance
    """
    f0, f_hat = X._decompose()
    result = GeneralizedDual._constant(0, size_like=X)
    atan_f0 = np.vectorize(mpmath.atan)(f0)
    one_plus_f0_sq = 1 + f0 ** 2
    max_order = X.m
    f_hat_pow = GeneralizedDual._constant(1, size_like=X)
    one_plus_f0_sq_pow = 1
    # Precompute f0 powers for inner sum
    f0_powers = {0: 1}
    for n in range(1, max_order + 1):
        f_hat_pow = f_hat_pow * f_hat  # incrementally compute f_hat^n
        one_plus_f0_sq_pow *= one_plus_f0_sq
        # Build inner sum depending on n's parity
        inner_sum = 0
        if n % 2 == 1:
            for j in range(1, n // 2 + 1):
                k = 2 * j
                if k not in f0_powers:
                    f0_powers[k] = f0 ** k
                binom = GeneralizedDual._mp_comb(n, k)
                inner_sum += binom * f0_powers[k] * ((-1) ** j)
            coeff = (1 + inner_sum) / one_plus_f0_sq_pow
            result += ((-1) ** ((n + 1) // 2 + 1)) * (f_hat_pow / mpmath.mpf(n)) * coeff
        else:
            for j in range(1, n // 2 + 1):
                k = 2 * j - 1
                if k not in f0_powers:
                    f0_powers[k] = f0 ** k
                binom = GeneralizedDual._mp_comb(n, k)
                inner_sum += binom * f0_powers[k] * ((-1) ** (j + 1))
            coeff = inner_sum / one_plus_f0_sq_pow
            result += ((-1) ** (n // 2)) * (f_hat_pow / mpmath.mpf(n)) * coeff
    return result + atan_f0

def asin(X):
    """
    Compute the Inverse Sine function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Sine function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> asin(x)  # where x is a GeneralizedDual instance
    """
    return atan(X / sqrt(1 - X**2))

def acos(X):
    """
    Compute the Inverse Cosine function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Cosine function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> acos(x)  # where x is a GeneralizedDual instance
    """
    return mpmath.pi/2 - asin(X)

def acot(X):
    """
    Compute the Inverse Cotangent function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Cotangent function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> acot(x)  # where x is a GeneralizedDual instance
    """
    return atan(1 / X)

# --- Hyperbolic ---

def sinh(X):
    """
    Compute the Hyperbolic Sine function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Hyperbolic Sine function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> sinh(x)  # where x is a GeneralizedDual instance
    """
    return (exp(X) - exp(-X)) / 2

def cosh(X):
    """
    Compute the Hyperbolic Cosine function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Hyperbolic Cosine function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> cosh(x)  # where x is a GeneralizedDual instance
    """
    return (exp(X) + exp(-X)) / 2

def tanh(X):
    """
    Compute the Hyperbolic Tangent function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Hyperbolic Tangent function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> tanh(x)  # where x is a GeneralizedDual instance
    """
    return sinh(X) / cosh(X)

def coth(X): 
    """
    Compute the Hyperbolic Cotangent function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Hyperbolic Cotangent function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> coth(x)  # where x is a GeneralizedDual instance
    """
    return cosh(X) / sinh(X)

def sech(X):
    """
    Compute the Hyperbolic Secant function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Hyperbolic Secant function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> sech(x)  # where x is a GeneralizedDual instance
    """
    return 1 / cosh(X)

def csch(X):
    """
    Compute the Hyperbolic Cosecant function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Hyperbolic Cosecant function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> csch(x)  # where x is a GeneralizedDual instance
    """
    return 1 / sinh(X)

def asinh(X):
    """
    Compute the Inverse Hyperbolic Sine function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Hyperbolic Sine function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> asinh(x)  # where x is a GeneralizedDual instance
    """
    return log(X + sqrt(X**2 + 1))

def acosh(X):
    """
    Compute the Inverse Hyperbolic Cosine function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Hyperbolic Cosine function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> acosh(x)  # where x is a GeneralizedDual instance
    """
    return log(X + sqrt(X**2 - 1))

def atanh(X):
    """
    Compute the Inverse Hyperbolic Tangent function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Hyperbolic Tangent function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> atanh(x)  # where x is a GeneralizedDual instance
    """
    return log((1 + X) / (1 - X)) / 2

def acoth(X):
    """
    Compute the Inverse Hyperbolic Cotangent function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Hyperbolic Cotangent function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> acoth(x)  # where x is a GeneralizedDual instance
    """
    return log((X + 1) / (X - 1)) / 2


def asech(X):
    """
    Compute the Inverse Hyperbolic Secant function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Hyperbolic Secant function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> asech(x)  # where x is a GeneralizedDual instance
    """
    return log((1 + sqrt(1 - X**2)) / X)


def acsch(X):
    """
    Compute the Inverse Hyperbolic Cosecant function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Inverse Hyperbolic Cosecant function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> acsch(x)  # where x is a GeneralizedDual instance
    """
    return log(1/X + sqrt(1/X**2 + 1))

# --- Other ---

def erf(X):
    """
    Compute the Error function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Error function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> erf(x)  # where x is a GeneralizedDual instance
    """
    f0, f_hat = X._decompose()
    def hermite_poly(n, x):
        if n == 0:
            return 1
        elif n == 1:
            return 2 * x
        else:
            Hnm2 = 1
            Hnm1 = 2 * x
            for i in range(2, n + 1):
                Hn = 2 * x * Hnm1 - 2 * (i - 1) * Hnm2
                Hnm2, Hnm1 = Hnm1, Hn
            return Hn
    result = GeneralizedDual._constant(0, size_like=X)
    base = GeneralizedDual._constant(1, size_like=X)
    for k in range(1, X.m + 1):
        df = (- 1)**(k - 1) * hermite_poly(k - 1, f0) * np.vectorize(mpmath.exp)(- f0**2)
        base = base * f_hat
        result += base * df / mpmath.factorial(k)
    return 2 * result / np.vectorize(mpmath.sqrt)(mpmath.pi) + np.vectorize(mpmath.erf)(f0) 

# == Advanced Functions ==

# --- Gamma ---

def gamma(X):
    """
    Compute the Gamma function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Gamma function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> gamma(x)  # where x is a GeneralizedDual instance
    """
    f0, f_hat = X._decompose()
    f0 = np.vectorize(lambda x: mpmath.nan if (x % 1 == 0 and x <= 0) else x)(f0)
    def bell_poly(n, x):
        # x: seznam float, dolžine vsaj n
        B = [0] * (n + 1)
        B[0] = 1
        for k in range(1, n + 1):
            s = 0
            for j in range(1, k + 1):
                c = GeneralizedDual._mp_comb(k - 1, j - 1)
                s += x[j - 1] * B[k - j] * c
            B[k] = s
        return B[n]
    result = GeneralizedDual._constant(0, size_like=X)
    base = GeneralizedDual._constant(1, size_like=X)
    for k in range(1, X.m + 1):
        df = bell_poly(k, [np.vectorize(mpmath.psi)(0, f0)] + [np.vectorize(mpmath.psi)(i, f0) for i in range(1, k)])
        base = base * f_hat
        result += base * df / mpmath.factorial(k)
    return (1 + result) * np.vectorize(mpmath.gamma)(f0)

def loggamma(X):
    """
    Evaluates the Logarithmic Gamma function for a GeneralizedDual input.
    
    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Logarithmic Gamma function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> mpmath.mp.dps = 100
    >>> x = initialize(np.linspace(0, 10, 1000), m=7)
    >>> disp(loggamma(x).derivatives_along(0)) # display derivatives along zeroth variable (x)
    """
    f0, f_hat = X._decompose()
    f0 = np.vectorize(lambda x: mpmath.nan if (x % 1 == 0 and x <= 0) else x)(f0)
    result = GeneralizedDual._constant(0, size_like=X)
    base = 1
    for k in range(1, X.m + 1):
        df = np.vectorize(mpmath.psi)(k-1, f0)
        base = base * f_hat
        result += base * df / mpmath.factorial(k)
    return result + np.vectorize(mpmath.loggamma)(f0)

def factorial(X):
    """
    Compute the Factorial function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Factorial function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> factorial(x)  # where x is a GeneralizedDual instance
    """
    return gamma(X + 1)

def falling_factorial(X, N):
    """
    Evaluate the Falling Factorial function f(X, Y) := X * (X - 1) * ... * (X - N + 1).
    
    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Falling Factorial function.

    N : GeneralizedDual or numeric
        Point at which to evaluate the Falling Factorial function.
        
    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.
      
    Notes
    -----  
    If X is numeric, it is easy to convert them to GeneralizedDual class with:
    >>> X_new = GeneralizedDual._constant(X, size_like=T) # where T is same size as dual we need.
        # For percision purpuse, it is recomended that X is an array of mpmath numbers.


    Examples
    --------
    >>> falling_factorial(x, 3)  # for dual variable x
    >>> falling_factorial(5, n)  # for dual variable n
    """
    return gamma(X + 1) / gamma(X - N + 1)

def rising_factorial(X, N):
    """
    Evaluate the Rising Factorial function f(X, N) := X * (X + 1) * ... * (X + N - 1).

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Rising Factorial function.

    N : GeneralizedDual or numeric
        Point at which to evaluate the Rising Factorial function.
        
    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.
        
    Notes
    -----
    If X is numeric, it is easy to convert them to GeneralizedDual class with:
    >>> X_new = GeneralizedDual._constant(X, size_like=T) # where T is same size as dual we need.
        # For percision purpuse, it is recomended that X is an array of mpmath numbers.


    Examples
    --------
    >>> rising_factorial(x, 3)  # for dual variable x
    >>> rising_factorial(5, n)  # for dual variable n
    """
    return gamma(X + N) / gamma(X)

def comb(N, K):
    """
    Evaluate the Binomial Coefficient (N choose K) for a GeneralizedDual input. 

    Parameters
    ----------
    N : GeneralizedDual
        Point at which to evaluate the Binomial Coefficient function.
        
    K : GeneralizedDual
        Point at which to evaluate the Binomial Coefficient function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.
        
    Notes
    -----
    If K or N are numeric, it is easy to convert them to GeneralizedDual class with:
    >>> K_new = GeneralizedDual._constant(K, size_like=T) # where T is same size as dual we need.
        # For percision purpuse, it is recomended that X is an array of mpmath numbers.


    Examples
    --------
    >>> n, k = initialize(78, 21, m=3)
    >>> comb(n, k).gradient()
    """
    if N < K:
        if isinstance(K, GeneralizedDual):
            return GeneralizedDual._constant(0, size_like=K)
        elif isinstance(N, GeneralizedDual):
            return GeneralizedDual._constant(0, size_like=N)
    if not isinstance(K, GeneralizedDual) and not isinstance(N, GeneralizedDual):
        raise ValueError("In comb both values are not GeneralizedDual, cannot interpret size.")
    return gamma(N + 1) / (gamma(K + 1) * gamma(N - K + 1))

def beta(X, Y):
    """
    Evaluate Beta function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Beta function.
        
    Y: GeneralizedDual
        Point at which to evaluate the Beta function.    

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.
        
    Notes
    -----
    If X or Y are numeric, it is easy to convert them to GeneralizedDual class with:
    >>> X_new = GeneralizedDual._constant(X, size_like=T) # where T is same size as dual we need.
        # For percision purpuse, it is recomended that X is an array of mpmath numbers.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> Y = np.sin(X)
    >>> x, y = initialize(X, Y, m=3)
    >>> beta(x, y).hessian()
    """
    return gamma(X) * gamma(X) / gamma(X + Y)

# --- integrals of upper bound ---
 
def integral_upper(f, integrand, X):
    """
    Evaluates an `integral` of integrand with X as upper bound using 
    known primitive `f`.

    Parameters
    ----------
    f : callable
        Primitive function of `integrand`, applied pointwise.

    integrand : callable
        Function accepting GeneralizedDual and returning its integrand.

    X : GeneralizedDual
        Upper integration limit.

    Returns
    -------
    GeneralizedDual
        Result of integration with derivatives up to order `m`.

    Examples
    --------
    >>> f = lambda t: mpmath.ei(t)
    >>> integrand = lambda T: exp(T) / T
    >>> x = initialize(2.0, m=3)
    >>> integral_upper(f, integrand, x).derivatives_along(0)
    """
    f0, f_hat = X._decompose()
    result = GeneralizedDual._constant(np.vectorize(f)(f0), size_like=X)
    x = initialize(f0, m=X.m)
    derivs = integrand(x).derivatives_along(0)
    base = f_hat
    for k in range(1, X.m + 1):
        result += base * derivs[k - 1] / mpmath.factorial(k)
        base *= f_hat
    return result

def li(X):
    """
    Evaluate Logarithmic Integral for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Logarithmic Integral function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> x = initialize(3.0, m=3)
    >>> li(x).hessian()
    """
    return integral_upper(f=lambda t: mpmath.ei(mpmath.log(t)), 
                          integrand=lambda T: (1 / log(T)), 
                          X=X)
    
def ei(X):
    """
    Evaluate Exponential Integral for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Exponential Integral function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> x = initialize(1.0, m=4)
    >>> ei(x).derivatives_along(0)
    """
    return integral_upper(f=lambda t: mpmath.ei(t), 
                          integrand=lambda T: (exp(T) / T), 
                          X=X)
    
        
def si(X):
    """
    Evaluate Sine Integral for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Sine Integral function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> x = initialize(np.array([0.5, 1.5]), m=3)
    >>> si(x).diff((2,))
    """
    return integral_upper(f=lambda t: mpmath.si(t), 
                          integrand=lambda T: (sin(T) / T), 
                          X=X)
    
def ci(X):
    """
    Evaluate Cosine Integral for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Cosine Integral function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> x = initialize(np.array([0.1, 1.0, 2.0]), m=2)
    >>> ci(x).gradient()
    """
    return integral_upper(f=lambda t: mpmath.ci(t), 
                          integrand=lambda T: (cos(T) / T), 
                          X=X)
    
def fresnels(X):
    """
    Evaluate Fresnel Sine Integral for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Fresnel Sine Integral function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> x = initialize(np.array([0.2, 0.4, 0.6]), m=3)
    >>> fresnels(x).diff((1,))
    """
    return integral_upper(f=lambda t: mpmath.fresnels(t), 
                          integrand=lambda T: (sin(mpmath.pi*T**2/mpmath.mpf(2))), 
                          X=X)
    
def fresnelc(X):
    """
    Evaluate Fresnel Cosine Integral for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Fresnel Cosine Integral function.

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> x = initialize(np.array([0.1, 0.3, 0.5]), m=2)
    >>> fresnelc(x).derivatives_along(0)
    """
    return integral_upper(f=lambda t: mpmath.fresnelc(t), 
                          integrand=lambda T: (cos(mpmath.pi*T**2/mpmath.mpf(2))), 
                          X=X)
    
###################### INVERSE FUNCTIONS ###################################

def inverse(X, df):
    """
    Helps to evaluate an inverse function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point, with non-dual part already evaluated on inverse function, at which to evaluate the inverse function.

    df : list
        List containing deriviates of a function applied on non-dual part of starting non-dual part of X.

    Returns
    -------
    GeneralizedDual
        Result including derivatives of an inverse function up to order `m`.

    Examples
    --------
    >>> def erfinv(X): # Define a custum inverse function like this
            f0, f_hat = X._decompose() # Separate non-dual and dual part
            # apply inverse f function on non-dual part:
            f0 = np.vectorize(lambda x: mpmath.erfinv(x) if (-1 <= x <= 1) else mpmath.nan)(f0)
            X = GeneralizedDual._compose(f0, f_hat) # reconstruct a variable
            x0 = initialize(f0, m=X.m) # initialize an helper variable
            F = erf(x0) # define a helper dual function
            df = F.derivatives_along(0) # obtain deriviates of f
            return inverse(X, df) # You have succesfully defined erf^-1(X)
    """
    f0, f_hat = X._decompose()
    N = X.m
    def P(N, df): 
        # where P[(j,k)] = P(j,k) from the paper
        P = {}
        for j in range(1, N+1): 
            P[(j, j)] = df[1]**mpmath.mpf(j)
            for k in range(j+1, N+1):
                P[(j, k)] = 0
                for l in reversed(range(1, k - j + 1)):
                    P[(j, k)] += (l * j - k + j + l) * df[l + 1] / mpmath.factorial(l + 1) * P[(j, k - l)]
                P[(j, k)] = P[(j, k)] * 1 / (k - j) * 1 / df[1]
        return P
    P = P(N, df)
    b_n = {} # Vector of pre-computed dummy variable values
    b_n[1] = 1/df[1]
    c_n = {} # vector of Taylor series coefficients
    c_n[1] = b_n[1] / mpmath.factorial(1)
    for n in range(2,N+1):
        b_n[n] = 0
        for j in range(1,n): 
            b_n[n] = b_n[n] + b_n[j]/mpmath.factorial(j) * P[(j,n)]
        b_n[n] = b_n[n] * mpmath.factorial(n) * -1*b_n[1]**mpmath.mpf(n) 
        c_n[n] = b_n[n] / mpmath.factorial(n)
    c_n[0] = f0 ## f^-1(x0)
    result = GeneralizedDual._constant(c_n[0], size_like=X)
    base = GeneralizedDual._constant(1, size_like=X)
    for k in range(1, N + 1):
        base = base * f_hat
        result += base * c_n[k]
    return result

def lambertw(X, branch=0):
    """
    Evaluate the Lambert W function on a given branch for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Lambert W function.

    branch : int, optional
        Branch index (default is 0).

    Returns
    -------
    GeneralizedDual
        Result including derivatives up to order `m`.

    Examples
    --------
    >>> x = initialize(0.4, m=3)
    >>> lambertw(x, branch=0).derivatives_along(0)
    """
    f0, f_hat = X._decompose()
    f0 = np.vectorize(mpmath.lambertw)(f0, k=branch)
    X = GeneralizedDual._compose(f0, f_hat)
    x0 = initialize(f0, m=X.m)
    F = x0 * exp(x0)
    df = F.derivatives_along(0)
    return inverse(X, df)

def erfinv(X):
    """
    Inverse error function for a GeneralizedDual input.

    Parameters
    ----------
    X : GeneralizedDual
        Point at which to evaluate the Lambert W function.
        The domain of inverse error function is limited on -1 <= X <= 1.

    Returns
    -------
    GeneralizedDual
        Result with derivatives up to order `m`.

    Examples
    --------
    >>> x = initialize(0.5, m=3)
    >>> erfinv(x).diff((1,))
    """
    f0, f_hat = X._decompose()
    f0 = np.vectorize(lambda x: mpmath.erfinv(x) if (-1 <= x <= 1) else mpmath.nan)(f0)
    X = GeneralizedDual._compose(f0, f_hat)
    x0 = initialize(f0, m=X.m)
    F = erf(x0)
    df = F.derivatives_along(0)
    return inverse(X, df)

