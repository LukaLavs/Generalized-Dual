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

## License

MIT © 2025 Luka Lavš
