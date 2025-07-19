import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generalized_dual import *
import numpy as np
import mpmath
import matplotlib.pyplot as plt
# Basic use:


    
    
def build_taylor_at_once(F, *centers, to_float=False):
    if to_float:
        F = F.to_float()
    zero = F._default_zero()
    n = F.n

    def T(vars):
        result = zero
        for key, coeff in F.terms.items():
            term = 1
            for i in range(n):
                if vars[i] is None:
                    if key[i] != 0:
                        term = 0
                        break
                else:
                    term *= (vars[i] - centers[i]) ** key[i]
            # Support coeff being scalar or array; multiply accordingly
            result = result + coeff * term
        return result

    return T


  
def build_taylor(F, *centers, to_float=False):
    if to_float:
        F = F.to_float()
    zero = F._default_zero()
    n = F.n
    zero_flat = zero.flatten()
    output_size = zero_flat.size
    funcs = []
    for comp_idx in range(output_size):
        def make_func(idx):
            def func(vars):
                result = zero_flat[idx] * 0  # zero scalar of correct type
                for key, coeff in F.terms.items():
                    term = 1
                    for i in range(n):
                        if vars[i] is None:
                            if key[i] != 0:
                                term = 0
                                break
                        else:
                            term *= (vars[i] - centers[i]) ** key[i]
                    if term != 0:
                        result += coeff.flatten()[idx] * term
                return result
            return func
        funcs.append(make_func(comp_idx))
    funcs_array = np.array(funcs).reshape(zero.shape)
    return funcs_array

mpmath.mp.dps = 15
X = np.linspace(0, 3, 5)
Y = np.sin(X) + 1
Z = np.cos(X)
x, y, z = initialize(X, Y, Z, m=3) # Initilize three variables with max order of deriviates m
F = sin(x*y + z)
l = 4
p=l
taylf = build_taylor(F, X[p], 111, 1111, to_float=True)[l]  # Fixed: centers are scalars

X_range = np.linspace(-2*np.pi, 2*np.pi, 300)
print("fdsjfkajsdfkajsfk", taylf([X_range, None, None])) # I want thhis to work also 
Y_taylor = np.array([taylf([x, None, None]) for x in X_range])  # Fixing y,z to center

# True function values at y=Y[l], z=Z[l]
Y_f = np.sin(X_range * Y[p] + Z[p])

# Plotting
plt.plot(X_range, Y_f, label='sin(x)')
plt.plot(X_range, Y_taylor, label='Taylor approx')

val_at_1 = np.sin(X[p] * Y[p] + Z[p])
plt.scatter([X[p]], [val_at_1], color='red', label='Taylor at x=1')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Taylor approximation of sin(x * y + z) fixing y, z at center')
plt.show()





