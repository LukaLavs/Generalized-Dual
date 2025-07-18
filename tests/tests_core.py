import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generalized_dual import *
import numpy as np
import mpmath

# Basic use:
mpmath.mp.dps = 50
X = np.linspace(0, 3, 5)
Y = np.sin(X)
Z = np.array([2, 7, 2, 3, 1])
x, y, z = initialize(X, Y, Z, m=6) # Initilize three variables with max order of deriviates m
F = lambertw(x*y - nroot(z, 6)) * log(x, y)
Df = F.diff((2, 1, 2))
print("F_x^2yz^2(X, Y, Z) = ")
print(Df)

print("\n")

# Aproximate int_{[0.3, 0.7] x [0.2, 0.6]} sin(x*y) / beta(x, y) dxdy 
# with taylor polinomials of degree m on specific points
mpmath.mp.dps = 15 # default percision
A = [0.3, 0.2]  # Lower bounds
B = [0.7, 0.6]  # Upper bounds
N = [20, 20]    # Subdivisions
m = 2        # Max derivative order
integrand = lambda X, Y: sin(X*Y) / beta(X, Y)
integral_aprox = experimental.integrate(integrand, A, B, N, m)
print("Integral aproximation: ")
print(integral_aprox)

print("\n")

# Define an custum inverse function
mpmath.mp.dps = 20 # 20 digits percision. Much higher percisions are possible also
def my_inverse_func(X): # Quick note: erfinv is already included in generalized_dual module
    """ Returns inverse of erf function """
    f0, f_hat = X._decompose() # Separates non-dual part from dual part
    f0 = np.vectorize(lambda x: mpmath.erfinv(x) if (-1 <= x <= 1) else mpmath.nan)(f0) # Applies custum inverse on non-dual part
    X = GeneralizedDual._compose(f0, f_hat) # Combines new non-dual part with dual-part
    x0 = initialize(f0, m=X.m) # Initializes and variable which helps us get derivatives
    F = erf(x0) # We define an function of which inverse we are interested, with dual functions
    df = F.derivatives_along(0) # We compute the derivatives
    return inverse(X, df) # We call inverse function which uses Lagrange Inversion theorem
X = 0.4
x = initialize(X, m=7)
F = my_inverse_func(x)
print("List of [f^0, f^1, ..., f^m](X):")
disp(F.derivatives_along(0)) # Along zero-th variable (x)

print("\n")

# Acces the term in taylor expansion which is standing next to (x -X)^k(y - Y)^l
X = np.array([[1, 2], [3, 4]])
Y = np.log(X)
x, y = initialize(X, Y, m=3)
k, l = 2, 1
term = (fresnelc(x - comb(x, y))).terms[(k, l)]
print("The term is: ")
disp(term)

print("\n")

# Plot fx^2y^2, for f = abs(lambertw(x) + log(y))... with fixed y=2.3 with matplotlib
import matplotlib.pyplot as plt
import time
mpmath.mp.dps = 15
start = time.time()
X, Y = np.linspace(0, 2, 200), 2.3 * np.ones(200)
x, y = initialize(X, Y, m=4)
F = lambda X, Y: dual_abs(lambertw(X) + log(Y)) * exp(-X**2) + rising_factorial(sin(X*5), fresnelc(X*Y)/(X*Y) + 3)
fx2y2 = F(x, y).diff((2, 2), to_float=True)
end = time.time()
print(f"Execution time for a plot: {end - start:.6f} seconds.")
plt.plot(X, fx2y2, label='fx^2y^2(x, y0=2.3)')
plt.title('f(x, y) := abs(beta(lambertw(x), abs(log(y)))) + rising_factorial(sin(5x), fresnelc(xy)/(xy) + 3)')
plt.grid(True)
plt.legend()
plt.show()
inverse()
