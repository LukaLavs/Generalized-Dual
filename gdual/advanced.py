from .gdual import *

import math
from scipy.special import gamma as Sgamma
from scipy.special import digamma as Sdigamma
from scipy.special import polygamma as Spolygamma
import scipy.special
import scipy.stats

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

def CDFgamma(X, shape, scale):
    x0, x_hat = X.decompose()
    F = (1 / math.gamma(shape)) * (X / scale)**(shape - 1) * exp(-X / scale) / scale
    f0 = float(scipy.stats.gamma.cdf(x0, a=shape, scale=scale))
    F = [F.diff(i) for i in range(X.order + 1)]
    return f0, F

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
    derivs = integrand(x).deriviates("x")
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
    
###################### INVERSE FUNCTIONS ###################################

def inverse(f0, f_hat, N, df):
    def P(N, df): 
        # where P[(j,k)] = P(j,k) from the paper
        P = {}
        for j in range(1, N+1): 
            P[(j, j)] = df[1]**j
            for k in range(j+1, N+1):
                P[(j, k)] = 0
                for l in reversed(range(1, k - j + 1)):
                    P[(j, k)] += (l * j - k + j + l) * df[l + 1] / math.factorial(l + 1) * P[(j, k - l)]
                P[(j, k)] = P[(j, k)] * 1 / (k - j) * 1 / df[1]
        return P
    P = P(N, df)
    b_n = {} # Vector of pre-computed dummy variable values
    b_n[1] = 1/df[1]
    c_n = {} # vector of Taylor series coefficients
    c_n[1] = b_n[1] / math.factorial(1)
    for n in range(2,N+1):
        b_n[n] = 0
        for j in range(1,n): 
            b_n[n] = b_n[n] + b_n[j]/math.factorial(j) * P[(j,n)]
        b_n[n] = b_n[n] * math.factorial(n) * -1*b_n[1]**n 
        c_n[n] = b_n[n] / math.factorial(n)
    c_n[0] = f0 ## f^-1(x0)
    result = c_n[0]
    base = 1
    for k in range(1, N + 1):
        base = base * f_hat
        result += base * c_n[k]
    return result

def lambertW(X, branch=0):
    """ branch = -1 also implemented """
    f0, f_hat = X.decompose()
    if f0 < - math.exp(-1): raise ValueError("LambertW not defined")
    elif f0 > 0 and branch == -1: raise ValueError("Branch gives complex result")
    N = X.order
    f0 = float(scipy.special.lambertw(f0, k=branch).real)
    x0 = GDual.initialize({"x0": f0}, order=N)
    F = x0 * exp(x0)
    df = [F.diff((i,)) for i in range(N + 1)]
    
    return inverse(f0, f_hat, N, df)

def erfinv(X):
    """ Returns inverse of erf function """
    f0, f_hat = X.decompose()
    N = X.order
    f0 = float(scipy.special.erfinv(f0))
    x0 = GDual.initialize({"x0": f0}, order=N)
    F = erf(x0)
    df = [F.diff((i,)) for i in range(N + 1)]
    
    return inverse(f0, f_hat, N, df)
