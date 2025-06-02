from .gdual import *

import math
import scipy


###################### ADVANCED FUNCTIONS / PERHAPS LESS STABLE ######################

# --- Gamma ---

def gamma(X):
    f0, f_hat = X.decompose()
    if isinstance(f0, int) and f0 <= 0: # then at the pole
        return float("inf") # In hopes to get sometime later 1/inf -> 0, othervise main f not well difined
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
        df = bell_poly(k, [scipy.special.digamma(f0)] + [scipy.special.polygamma(i, f0) for i in range(1, k)])
        base = base * f_hat
        result += df * base / math.factorial(k)
    return scipy.special.gamma(f0) * (1 + result)

def loggamma(X):
    f0, f_hat = X.decompose()
    if isinstance(f0, int) and f0 <= 0:  # Poles of gamma
        return float("inf")
    result = GDual.constant(0.0, X.vars, X.order)
    base = 1
    for k in range(1, X.order + 1):
        # poligama funkcija reda (k-1) na f0
        df = scipy.special.polygamma(k-1, f0)
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

# --- integrals of upper bound ---
 
def integral_upper(f, integrand, X):
    """ returns int_a^X integrand(t)dt """
    f0, f_hat = X.decompose()
    result = f(f0)
    x = GDual.initialize({"x": f0}, X.order)
    derivs = integrand(x).univariate_derivatives("x")
    base = f_hat
    for k in range(1, X.order + 1):
        result += derivs[k - 1] * base / math.factorial(k)
        base *= f_hat
    return result

def Li(X):
    """ returns logaritmic integral """
    return integral_upper(f=lambda t: scipy.special.expi(math.log(t)), 
                          integrand=lambda T: (1 / log(T)), 
                          X=X)
   
def Ei(X):
    """ returns exponential integral """
    return integral_upper(f=lambda t: scipy.special.expi(t), 
                          integrand=lambda T: (exp(T) / T), 
                          X=X)
    
        
def Si(X):
    """ returns sine integral """
    return integral_upper(f=lambda t: scipy.special.sici(t)[0], 
                          integrand=lambda T: (sin(T) / T), 
                          X=X)
    
def Ci(X):
    """ returns cosine integral"""
    return integral_upper(f=lambda t: scipy.special.sici(t)[1], 
                          integrand=lambda T: (cos(T) / T), 
                          X=X)
    
def S(X):
    """ returns freshnel sine integral """
    return integral_upper(f=lambda t: scipy.special.fresnel(t)[0], 
                          integrand=lambda T: (sin(math.pi*T**2/2)), 
                          X=X)
    
def C(X):
    """ returns freshnel cosine integral """
    return integral_upper(f=lambda t: scipy.special.fresnel(t)[1], 
                          integrand=lambda T: (cos(math.pi*T**2/2)), 
                          X=X)




    
