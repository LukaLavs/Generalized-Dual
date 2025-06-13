from gdual import *

x, y = GDual.initialize({"x": 0.5, "y": 0.5}, order=9)

r = sin(x*y)
A = [0, 0]
B = [1, 1]
N = [1, 1]

import math
from itertools import product

def integrate_point(coeffs, A, B, N=None):
    # A, B definirata območje integracije
    n = len(A)
    I = 0
    for key, val in coeffs.items():
        term = val
        for i in range(n):
            a, b = A[i], B[i]
            center = (a + b) / 2
            alpha = key[i]
            # Integriraj (x - center)^alpha od a do b
            # To je: ∫_a^b (x - center)^alpha dx = (b - center)^{alpha+1} - (a - center)^{alpha+1} / (alpha+1)
            upper = (b - center)**(alpha + 1)
            lower = (a - center)**(alpha + 1)
            integral = (upper - lower) / (alpha + 1)
            term *= integral
        I += term
    return I

from itertools import product
def INTEGRATE_BOX(center, exponents, h):
    """
    Compute ∫∫...∫ Π (x_i - center_i)^α_i dx over box centered at center, with side lengths h
    The box is from (center_i - h_i/2) to (center_i + h_i/2)
    """
    result = 1
    for i in range(len(exponents)):
        alpha = exponents[i]
        a = -h[i] / 2
        b = h[i] / 2
        # ∫_{-h/2}^{h/2} x^alpha dx = (b^{α+1} - a^{α+1}) / (α+1)
        upper = b**(alpha + 1)
        lower = a**(alpha + 1)
        integral = (upper - lower) / (alpha + 1)
        result *= integral
    return result
def Integrate(func, A, B, N, m):
    n = len(N)
    h = [(B[i] - A[i]) / N[i] for i in range(n)]
    K = list(product(*[[i for i in range(N[j])] for j in range(n)]))
    I = 0
    for k in K:
        Xsi = GDual.initialize({f"x{i}": A[i] + (k[i] + 0.5)*h[i] for i in range(n)}, order=m)
        dual = func(*Xsi)
        for key, val in dual.terms.items():
            I += val * INTEGRATE_BOX([0]*n, key, h) # This line needs to be expanded
             
    return I



r = sin(x*y)
A = [0, 0, 0, 0, 0]
B = [1, 1, 1, 1, 1]
N = [5, 5, 5, 5, 5]
import time

start = time.time()
func = lambda x, y, z, w, q: tanh(log(1 + q**2 + (advanced.gamma(1 + w*sin(x*advanced.comb(y, y**2))))**2) + sinh(z*advanced.Si(z) * y**1.1))

print(Integrate(func, A=A, B=B, N=N, m=2))
end = time.time()
print(end - start, "time")


