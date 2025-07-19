import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generalized_dual import *

import numpy as np
import mpmath
from matplotlib import pyplot as plt

def build_taylor(F, *centers, to_float=False):
    if to_float:
        F = F.to_float()

    zero = F._default_zero()
    n = F.n
    output_shape = zero.shape
    coeffs_flat = zero.flatten()
    num_outputs = coeffs_flat.size

    funcs = []

    for comp_idx in range(num_outputs):
        def make_func(idx):
            def func(vars):
                # Broadcast inputs
                shapes = [np.shape(v) for v in vars if v is not None]
                out_shape = np.broadcast_shapes(*shapes) if shapes else ()

                result = np.zeros(out_shape, dtype=float)
                for key, coeff in F.terms.items():
                    term = np.ones(out_shape, dtype=float)
                    for i in range(n):
                        if vars[i] is None:
                            if key[i] != 0:
                                term = 0
                                break
                        else:
                            term *= (vars[i] - centers[i]) ** key[i]
                    coeff_i = np.asarray(coeff).flatten()[idx]
                    result += coeff_i * term
                return result
            return func
        funcs.append(make_func(comp_idx))

    funcs_array = np.array(funcs).reshape(output_shape)
    return funcs_array



def build_taylor(F, *centers, to_float=False):
    if to_float:
        F = F.to_float()

    zero = F._default_zero()
    n = F.n
    output_shape = zero.shape
    coeffs_flat = zero.flatten()
    num_outputs = coeffs_flat.size

    funcs = []

    for comp_idx in range(num_outputs):
        def make_func(idx):
            def func(vars):
                # Broadcast inputs
                shapes = [np.shape(v) for v in vars if v is not None]
                out_shape = np.broadcast_shapes(*shapes) if shapes else ()

                result = np.zeros(out_shape, dtype=float)
                for key, coeff in F.terms.items():
                    term = np.ones(out_shape, dtype=float)
                    for i in range(n):
                        if vars[i] is None:
                            if key[i] != 0:
                                term = 0
                                break
                        else:
                            term *= (vars[i] - centers[i]) ** key[i]
                    coeff_i = np.asarray(coeff).flatten()[idx]
                    result += coeff_i * term
                return result
            return func
        funcs.append(make_func(comp_idx))

    funcs_array = np.array(funcs).reshape(output_shape)
    return funcs_array


mpmath.mp.dps = 15
X = np.linspace(0, 3, 5)
Y = np.sin(X) + 1
Z = np.cos(X)

x, y, z = initialize(X, Y, Z, m=3)
F = dual_abs(lambertw(sin(x*y + z), branch=-1))
X_range = np.linspace(-2*np.pi, 2*np.pi, 300)
p = 0
vec_f = np.vectorize(lambda x: float(mpmath.fabs(mpmath.lambertw(mpmath.sin(x*Y[p] + Z[p]), -1))))
Y_f = vec_f(X_range)
point_Y = vec_f(X[p])

taylf = build_taylor(F, X, Y, Z, to_float=True)[p]



# ✅ Works with array directly
Y_taylor = taylf([X_range, None, None])


# Plotting
plt.plot(X_range, Y_f, label='True function')
plt.plot(X_range, Y_taylor, label='Taylor approximation')
plt.scatter([X[p]], [point_Y], color='red', label='Expansion point')

plt.legend()
plt.xlabel('x')
plt.ylabel('f')
plt.title('Taylor approximation of sin(x*y + z) fixing y,z')
plt.show()
