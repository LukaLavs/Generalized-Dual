from .gdual import *

import math
from scipy.special import gamma as Sgamma
from scipy.special import digamma as Sdigamma
from scipy.special import polygamma as Spolygamma

from mpmath import mp
mp.dps = 100
def gamma2(X):
    ## Da povsem točno
    f0, f_hat = X.decompose()
    f0 = mp.mpf(f0)
    if isinstance(f0, int) and f0 <= 0:
        return mp.mpf('inf')

    def bell_poly(n, x):
        B = [mp.mpf(0) for _ in range(n + 1)]
        B[0] = mp.mpf(1)
        for k in range(1, n + 1):
            s = mp.mpf(0)
            for j in range(1, k + 1):
                c = mp.binomial(k - 1, j - 1)
                s += x[j - 1] * B[k - j] * c
            B[k] = s
        return B[n]

    result = GDual.constant(mp.mpf(0), X.vars, X.order)
    base = mp.mpf(1)
    for k in range(1, X.order + 1):
        dig = mp.psi(0, f0)  # 0-ti odvod, torej psi funkcija (digamma)
        polys = [mp.psi(i, f0) for i in range(1, k)]  # i-ti odvod psi (poligama)
        df = bell_poly(k, [dig] + polys)
        base *= f_hat
        term = GDual(X.vars, X.order, {key: v / mp.fac(k) for key, v in (df*base).terms.items()})
        result += term #df * base / float(mp.fac(k))
    return (1 + GDual(X.vars, X.order, {k: float(mp.gamma(f0)*v) for k, v in result.terms.items()}))
    return float(mp.gamma(f0)) * (1 + GDual(X.vars, X.order, {k: float(v) for k, v in result.terms.items()}))



###################### ADVANCED FUNCTIONS / LESS STABLE ######################

# --- Gamma ---

def gamma(X):
    f0, f_hat = X.decompose()
    if isinstance(f0, int) and f0 <= 0: # then at the pole
        return float("inf") # In hopes to get soemetime later 1/inf -> 0, othervise main f not well difined
    def bell_poly(n, x):
        # x: seznam float, dolžine vsaj n
        B = [0.0] * (n + 1)
        B[0] = 1.0
        for k in range(1, n + 1):
            s = 0.0
            for j in range(1, k + 1):
                c = math.comb(k - 1, j - 1)
                s += x[j - 1] * B[k - j] * c
            B[k] = s
        return B[n]
    result = GDual.constant(0.0, X.vars, X.order)
    base = 1
    for k in range(1, X.order + 1):
        df = bell_poly(k, [Sdigamma(f0)] + [Spolygamma(i, f0) for i in range(1, k)])
        base = base * f_hat
        result += df * base / math.factorial(k)
    return Sgamma(f0) * (1 + result)

def loggamma(X):
    f0, f_hat = X.decompose()
    if isinstance(f0, int) and f0 <= 0:  # Poles of gamma
        return float("inf")
    result = GDual.constant(0.0, X.vars, X.order)
    base = 1
    for k in range(1, X.order + 1):
        # poligama funkcija reda (k-1) na f0
        df = Spolygamma(k-1, f0)
        base = base * f_hat
        result += df * base / math.factorial(k)
    return math.lgamma(f0) + result

# --- Factorials ---

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
        m, n = [K.vars, K.order] if isinstance(K, GDual) else ([N.vars, N.order] if isinstance(N, GDual) else False)
        return GDual.constant(0.0, m, n)
    return gamma(N + 1) / (gamma(K + 1) * gamma(N - K + 1))

def beta(X, Y):
    return gamma(X) * gamma(X) / gamma(X + Y)

###################### ERF ###################################


    
