# generalized-dual

A minimal Python library for generalized dual numbers and automatic differentiation, supporting arbitrary-order derivatives, complex numbers, and vectorized operations. Many difficult functions are implemented.

## Installation

```bash
pip install generalized-dual
```

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

## Features

- Generalized dual numbers for multi-variable, high-order differentiation
- Real and complex support using `mpmath`
- Vectorized operations with NumPy arrays
- Symbolic integration (via Taylor expansions)
- Rich function support: `exp`, `log`, `trig`, `gamma`, `erfinv`, `lambertw`, etc.

## Structure

```
generalized_dual/
│
├── core.py          # GeneralizedDual class
├── functions.py     # Math operations
├── utils.py         # Initialization, display tools
├── experimental/    # Optional features
├── __init__.py
```

## License

MIT © 2025 Luka Lavš
