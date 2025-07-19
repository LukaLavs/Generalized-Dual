# generalized-dual

A minimal Python library for generalized dual numbers and automatic differentiation, supporting arbitrary-order derivatives, complex numbers, and vectorized operations. Many difficult functions are implemented.

## Installation

This package is not yet available on PyPI.

To install it locally, clone the repository and install it using pip:

    git clone https://github.com/LukaLavs/Automatic-Differentiation.git
    cd Automatic-Differentiation
    pip install .

## Usage

```python
from generalized_dual import GeneralizedDual, initialize
from generalized_dual.functions import exp, log, sin
import numpy as np
import mpmath

mpmath.mp.dps = 50  # Set precision

x, y = initialize(np.array([0.5]), np.array([1.0]), m=2)
f = exp(x * y) + log(y)
print(f.diff((1, 0)))  # Derivative w.r.t x
```

---

## Features

- Generalized dual numbers for multi-variable, high-order differentiation  
- Real and complex support via `mpmath`  
- Vectorized NumPy operations  
- Symbolic integration (via Taylor expansions)  
- Rich function support: `exp`, `log`, `trig`, `gamma`, `erfinv`, `lambertw`, and more

---

## Advanced Examples

### ⚙️ Higher-Order Partial Derivatives

```python
from generalized_dual import *
import numpy as np
import mpmath

mpmath.mp.dps = 50
X = np.linspace(0, 3, 5)
Y = np.sin(X)
Z = np.array([2, 7, 2, 3, 1])
x, y, z = initialize(X, Y, Z, m=6)
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
A = [0.3, 0.2]
B = [0.7, 0.6]
N = [20, 20]
m = 2
integrand = lambda X, Y: sin(X * Y) / beta(X, Y)
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
    f0, f_hat = X._decompose()
    f0 = np.vectorize(lambda x: mpmath.erfinv(x) if -1 <= x <= 1 else mpmath.nan)(f0)
    X = GeneralizedDual._compose(f0, f_hat)
    x0 = initialize(f0, m=X.m)
    F = erf(x0)
    df = F.derivatives_along(0)
    return inverse(X, df)

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
disp(term)
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
fx2y2 = F(x, y).diff((2, 2), to_float=True)
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

X = np.linspace(0, 3, 5)
Y, Z = np.sin(X) + 1, np.cos(X)
x, y, z = initialize(X, Y, Z, m=3)

F = dual_abs(lambertw(sin(x * y + z), branch=-1))
p = 1  # Center index

X_test = np.linspace(-2 * np.pi, 2 * np.pi, 300)
f_exact = lambda x: float(mpmath.fabs(mpmath.lambertw(mpmath.sin(x * Y[p] + Z[p]), k=-1)))
Y_true = np.vectorize(f_exact)(X_test)

taylf = build_taylor(F, X, Y, Z, to_float=True)[p]
Y_taylor = taylf([X_test, None, None])

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

X = 2
x = initialize(X, m=10)

F = sin(cos(x**2) + atan(x)) + sin(4*x)
f = lambda x: np.sin(np.cos(x**2) + np.arctan(x)) + np.sin(4*x)

taylor = build_taylor(F, X, to_float=True)

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
x, y = initialize(X, Y, m=3)
F = sin(x * y)
f = lambda x, y: np.sin(x * y)
taylor = build_taylor(F, X, Y, to_float=True) # build taylor function around (X, Y)
# If X, Y were ndarrays the result would be a ndarray of functions around respected centers

# Grid
x_vals = np.linspace(0, 4, 100)
y_vals = np.linspace(0, 4, 100)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# Evaluate
Z_fun = f(X_grid, Y_grid)
Z_taylor = taylor([X_grid, Y_grid]) # Evaluate taylor polinom around (X, Y)
Z_point = f(X, Y)

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid, Y_grid, Z_fun, cmap='viridis', alpha=0.5)
ax.plot_surface(X_grid, Y_grid, Z_taylor, cmap='plasma', alpha=0.5)
ax.scatter(X, Y, Z_point, color='red', s=40)

ax.set_title("Function vs. Taylor Approximation")
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

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
