# generalized-dual

A minimal Python library for generalized dual numbers and automatic differentiation, supporting arbitrary-order derivatives, complex numbers, and vectorized operations. Many difficult functions are implemented.

## Installation

This package is not yet available on PyPI.

To install it locally, clone the repository and install it using pip:

    git clone https://github.com/LukaLavs/Generalized-Dual.git
    cd generalized_dual
    pip install .

## Usage

```python
from generalized_dual import GeneralizedDual, initialize
from generalized_dual.functions import exp, log, sin
import numpy as np
import mpmath

mpmath.mp.dps = 100  # Set precision

x, y = initialize(np.array([0.5]), np.array([1.0]), m=2) # Prepare variables which support order m derivatives.
f = exp(x * y) + log(y) # Define a dual function and evaluate it
print(f.diff((1, 0)))  # Partial derivative f_x^1y^0
```

---

## Features

- Generalized dual numbers for multi-variable, high-order differentiation  
- Real and complex support via `mpmath`  
- Vectorized NumPy operations  
- Symbolic integration (via Taylor expansions)  
- Rich function support: `exp`, `log`, `dual_abs`, `trig`, `gamma`, `erfinv`, `lambertw`, and more
- 📖 See the full list of available functions in [FUNCTIONS.md](./FUNCTIONS.md).

---

## Advanced Examples

### ⚙️ Higher-Order Partial Derivatives

```python
from generalized_dual import *
import numpy as np
import mpmath

mpmath.mp.dps = 50 # Set percision
X = np.linspace(0, 3, 5)
Y = np.sin(X)
Z = np.array([2, 7, 2, 3, 1])
# Such choice of X, Y, Z will give us derivatives around points in (X, Y, Z)
x, y, z = initialize(X, Y, Z, m=5) # Set max order of terms to 5, since later we calculate f_x^2y_1z_2 which is of order 5
F = lambertw(x * y - nroot(z, 6)) * log(x, y)
Df = F.diff((2, 1, 2))
print("F_x^2yz^2(X, Y, Z) = ")
print(Df)
```

---

### 🧮 Approximate 2D Integral via Taylor Expansion

```python
from generalized_dual import *
import numpy as np
import mpmath

mpmath.mp.dps = 15
A = [0.3, 0.2] # A = [x_min, y_min]
B = [0.7, 0.6] # B = [x_max, y_max]
N = [20, 20] # Subdivide x interval into 20 pieces, and same for y interval
m = 2 # Approximate with multiple order 2 polinomials around points defined by A;B;N triple
integrand = lambda X, Y: sin(X * Y) / beta(X, Y)
# Now approximate \int_A^B integrand(x, y) dydx
integral_aprox = experimental.integrate(integrand, A, B, N, m)
print("Integral approximation: ")
print(integral_aprox)
```

---

### 🔁 Define a Custom Inverse Function via Lagrange Inversion

```python
from generalized_dual import *
import numpy as np
import mpmath

mpmath.mp.dps = 20
def my_inverse_func(X):
    f0, f_hat = X._decompose() # Decompose X into dual and non-dual part
    f0 = np.vectorize(lambda x: mpmath.erfinv(x) if -1 <= x <= 1 else mpmath.nan)(f0) # Apply inverse f on non-dual part
    X = GeneralizedDual._compose(f0, f_hat) # Compose it back
    x0 = initialize(f0, m=X.m) # Initialize a helper variable
    F = erf(x0) # evaluate dual f function
    df = F.derivatives_along(0) # Obtain its derivatives (gives [f^0, f^1, ..., f^m])
    return inverse(X, df) # Calls a function which applies Lagrange Inversion Theorem

x = initialize(0.4, m=7)
F = my_inverse_func(x)
print("List of [f^0, f^1, ..., f^m](X):")
disp(F.derivatives_along(0))
```

---

### 🔍 Access Specific Taylor Term

```python
from generalized_dual import *
import numpy as np

X = np.array([[1, 2], [3, 4]])
Y = np.log(X)
x, y = initialize(X, Y, m=3)
k, l = 2, 1
term = (fresnelc(x - comb(x, y))).terms[(k, l)]
print("The term is: ")
disp(term) # Term next to (x - x0)^k(y - y0)^l
# Note: terms is same size as X and Y variables
```

---

### 📈 Plot Mixed Derivative fx²y² with Custom Function

```python
from generalized_dual import *
import numpy as np
import matplotlib.pyplot as plt
import time

mpmath.mp.dps = 15
X, Y = np.linspace(0, 2, 200), 2.3 * np.ones(200)
x, y = initialize(X, Y, m=4)

F = lambda X, Y: dual_abs(lambertw(X) + log(Y)) * exp(-X**2) + \
                 rising_factorial(sin(X * 5), fresnelc(X * Y) / (X * Y) + 3)

start = time.time()
fx2y2 = F(x, y).diff((2, 2), to_float=True) # to_float=True converts from mpmath objects into floats for easier graphing
end = time.time()

print(f"Execution time for a plot: {end - start:.6f} seconds.")

plt.plot(X, fx2y2, label='∂²f/∂x²∂y² at y=2.3')
plt.title('f(x, y) := abs(lambertw(x) + log(y)) + rising_factorial(...)')
plt.grid(True)
plt.legend()
plt.show()
```

---

### 🔍 Taylor Approximation of `|LambertW(sin(xy + z))|` on Branch -1

```python
from generalized_dual import *
import numpy as np
import mpmath
import matplotlib.pyplot as plt

X = np.linspace(0, 3, 5)
Y, Z = np.sin(X) + 1, np.cos(X)
x, y, z = initialize(X, Y, Z, m=3)

F = dual_abs(lambertw(sin(x * y + z), branch=-1))
p = 1  # build_taylor will return an array of functions (same size as X, Y and Z)
# With p=1 we will select the function around (x, y, z) = (X[1], Y[1], Z[1])

X_test = np.linspace(-2 * np.pi, 2 * np.pi, 300)
f_exact = lambda x: float(mpmath.fabs(mpmath.lambertw(mpmath.sin(x * Y[p] + Z[p]), k=-1))) # Fix y and z variables to get 1-D function
Y_true = np.vectorize(f_exact)(X_test)

taylf = build_taylor(F, X, Y, Z, to_float=True)[p]
Y_taylor = taylf([X_test, None, None]) # Y and Z variables are fixed. (Fixing variables is not required, but in our case we would get 4-D function or 3-D, which would be hard to plot)

plt.plot(X_test, Y_true, label='True')
plt.plot(X_test, Y_taylor, label='Taylor')
plt.scatter(X[p], f_exact(X[p]), color='red')
plt.title(r"$|W_{-1}(\sin(xy+z))|$ Taylor Approximation")
plt.grid(); plt.legend(); plt.show()
```

---

### ✨ Another plot example

```python
from generalized_dual import *
import numpy as np
import matplotlib.pyplot as plt

X = 2
x = initialize(X, m=10)

F = sin(cos(x**2) + atan(x)) + sin(4*x)
f = lambda x: np.sin(np.cos(x**2) + np.arctan(x)) + np.sin(4*x)

taylor = build_taylor(F, X, to_float=True) # Since X is not an array, build_taylor just returns a function

X_range = np.linspace(0, 3, 100)
plt.plot(X_range, f(X_range), label='func')
plt.plot(X_range, taylor([X_range]), label='taylor')
plt.scatter(X, f(X))
plt.ylim(-3, 3)
plt.legend()
plt.show()
```

---

### 🌐 Visualize 2D Taylor Approximation in 3D

```python
from generalized_dual import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Setup
X, Y = 2, 3
x, y = initialize(X, Y, m=3) # With a goal of plotting taylor polinomial around point (x, y) = (2, 3)
F = sin(x * y) # for sin(x y) function
f = lambda x, y: np.sin(x * y) # Define a function for plotting the original
taylor = build_taylor(F, X, Y, to_float=True) # build taylor function around (X, Y)
# If X, Y were ndarrays the result would be a ndarray of functions around respected centers
# Here build_taylor returned just a function (since X, Y are not arrays)

# Grid
x_vals = np.linspace(0, 4, 100)
y_vals = np.linspace(0, 4, 100)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# Evaluate
Z_fun = f(X_grid, Y_grid)
Z_taylor = taylor([X_grid, Y_grid]) # Evaluate taylor polinom around (X, Y)
# Note: taylor([X_grid, None]) would return 1-D taylor polinomial of a function f(x, y0=3) around x0=2
Z_point = f(X, Y)

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z_fun, cmap='viridis', alpha=0.5)
ax.plot_surface(X_grid, Y_grid, Z_taylor, cmap='plasma', alpha=0.5)
ax.scatter(X, Y, Z_point, color='red', s=40)

ax.set_title("Function vs. Taylor Approximation")
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

ax.set_xlim(1.5, 2.5)
ax.set_ylim(2.5, 3.5)
ax.set_zlim(-1, 1)

legend_elements = [
    Line2D([0], [0], color='blue', lw=3, label='Function'),
    Line2D([0], [0], color='orange', lw=3, label='Taylor'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Center')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()
```

---

## Project Structure

```
generalized_dual/
│
├── core.py          # GeneralizedDual class
├── functions.py     # Math operations
├── utils.py         # Initialization, display tools
├── experimental/    # Optional features
├── __init__.py
tests/
setup.py
pyproject.toml
README.md
LICENSE
```

---

## 📚 References & Further Reading

This project was inspired and guided by the following resources:

- [Dual number — Wikipedia](https://en.wikipedia.org/wiki/Dual_number)  
- [The algebra of truncated polynomials](https://darioizzo.github.io/audi/theory_algebra.html)  
- [How to Find the Taylor Series of an Inverse Function](https://randorithms.com/2021/08/31/Taylor-Series-Inverse.html)   

If you found additional valuable resources, please consider contributing!

---

## License

MIT © 2025 Luka Lavš
