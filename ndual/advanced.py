import math
from scipy.special import gamma as Sgamma
from scipy.special import digamma as Sdigamma
from scipy.special import polygamma as Spolygamma
from mpmath import hyp3f2
from .dualnumber import *

def evaluate_f(derivs, X):
    """Evaluate f(X), where X is an NDualNumber, using Taylor expansion."""
    n = X.n
    a = X.coeffs[0]
    B = NDualNumber([0.0] + X.coeffs[1:])
    result = NDualNumber([0.0] * (n + 1))
    B_power = NDualNumber([1.0] + [0.0] * n)  # B^0 = 1
    for k in range(n + 1):
        term_coeff = derivs[k](a) / math.factorial(k)
        term = B_power * term_coeff
        result = result + term
        B_power = B_power * B  # B^k → B^{k+1}
    return result

###################### ADVANCED FUNCTIONS ######################

###################### GAMMA ###################################

def bell_poly(n, x):
    """ Ni podprt za ndual. Eksponentni Bellov polinom B_n(x1, ..., xn), x je seznam dolžine n"""
    B = [0] * (n + 1)
    B[0] = 1  # B_0 = 1
    for k in range(1, n + 1):
        B[k] = 0
        for j in range(1, k + 1):
            B[k] += x[j - 1] * B[k - j] * math.comb(k - 1, j - 1)
    return B[n]

def gamma_derivative_funcs(n):
    funcs = []
    for j in range(n + 1):
        if j == 0:
            funcs.append(lambda x: float(Sgamma(x)))
        else:
            funcs.append(
                lambda x, j=j: float(Sgamma(x)) * float(
                    bell_poly(j, [float(Sdigamma(x))] + [float(Spolygamma(k, x)) for k in range(1, j)])
                )
            )
    return funcs

def Sin_gamma_domain(X):
    if isinstance(X, NDualNumber):
        a = X.coeffs[0]
    else:
        a = X
    if math.isinf(a) or a in [0, -1, -2, ...]:
        return False    
    return True

def gamma(X):
    if isinstance(X, (int, float)):
        return float(Sgamma(X))
    if not Sin_gamma_domain(X):
        return float("inf")
    n = X.n
    derivs = gamma_derivative_funcs(n)
    return evaluate_f(derivs, X)

def factorial(X):
    return gamma(X + 1)

def falling_factorial(X, N):
    """ returns X * ... * (X - N + 1)"""
    return gamma(X + 1) / gamma(X - N + 1)

def rising_factorial(X, N):
    """ returns X * ... * (X + N - 1)"""
    return gamma(X + N) / gamma(X)

def comb(N, K):
    """ returns (n k)"""
    if N < K:
        n = K.n if isinstance(K, NDualNumber) else (N.n if isinstance(N, NDualNumber) else False)
        if n == False: 
            return 0
        return to_dual(0, n=n)
    return gamma(N + 1) / (gamma(K + 1) * gamma(N - K + 1))

###################### Arcsin ###################################
#2^(-1 + n) Sqrt[\[Pi]] x^(1 - n)
#  HypergeometricPFQRegularized[{1/2, 1/2, 1}, {1 - n/2, (3 - n)/2}, 
#  x^2]




###################### Error function ###################################

#2^n x^(1 - n)
#  HypergeometricPFQRegularized[{1/2, 1}, {1 - n/2, (3 - n)/2}, -x^2]
