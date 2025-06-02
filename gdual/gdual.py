import math
from collections import defaultdict
import warnings

class GDual:
    def __init__(self, vars, order, terms=None):
        self.vars = tuple(vars)
        self.order = order
        self.terms = defaultdict(float)
        if terms:
            self.terms.update(terms)
            
    # --- Comparing values ---
    
    def __eq__(self, other):
        if isinstance(other, GDual):
            return self.terms.get((0,) * len(self.vars), 0.0) == other.terms.get((0,) * len(self.vars), 0.0)
        elif isinstance(other, (int, float)):
            return self.terms.get((0,) * len(self.vars), 0.0) == other
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, GDual):
            return self.terms.get((0,) * len(self.vars), 0.0) < other.terms.get((0,) * len(self.vars), 0.0)
        elif isinstance(other, (int, float)):
            return self.terms.get((0,) * len(self.vars), 0.0) < other
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, GDual):
            return self.terms.get((0,) * len(self.vars), 0.0) <= other.terms.get((0,) * len(self.vars), 0.0)
        elif isinstance(other, (int, float)):
            return self.terms.get((0,) * len(self.vars), 0.0) <= other
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, GDual):
            return self.terms.get((0,) * len(self.vars), 0.0) > other.terms.get((0,) * len(self.vars), 0.0)
        elif isinstance(other, (int, float)):
            return self.terms.get((0,) * len(self.vars), 0.0) > other
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, GDual):
            return self.terms.get((0,) * len(self.vars), 0.0) >= other.terms.get((0,) * len(self.vars), 0.0)
        elif isinstance(other, (int, float)):
            return self.terms.get((0,) * len(self.vars), 0.0) >= other
        else:
            return NotImplemented
        
    # --- ---
    
    @staticmethod
    def initialize(var_values: dict, order: int):
        vars = tuple(var_values.keys())
        n = len(vars)
        result = {}
        for i, var in enumerate(vars):
            terms = defaultdict(float)
            # konstanta je vrednost spremenljivke
            terms[(0,) * n] = var_values[var]
            # prvi odvod po tej spremenljivki je 1
            index = [0] * n
            index[i] = 1
            terms[tuple(index)] = 1.0
            result[var] = GDual(vars, order, terms)
        if len(vars) == 1:
            return result[vars[0]]
        return tuple(result[v] for v in vars)

    @staticmethod
    def constant(c, vars, order):
        return GDual(vars, order, {(0,) * len(vars): c})

    def __add__(self, other):
        if isinstance(other, (int, float)):
            # Dodaj konstantni GDual
            other_const = GDual.constant(other, self.vars, self.order)
            return self + other_const
        elif isinstance(other, GDual):
            result = GDual(self.vars, self.order)
            for k in set(self.terms) | set(other.terms):
                result.terms[k] = self.terms.get(k, 0.0) + other.terms.get(k, 0.0)
            return result
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # Pomnoži vse koeficiente z realno konstanto
            result_terms = {k: v * other for k, v in self.terms.items()}
            return GDual(self.vars, self.order, result_terms)
        elif isinstance(other, GDual):
            result = GDual(self.vars, self.order)
            for k1, v1 in self.terms.items():
                for k2, v2 in other.terms.items():
                    k = tuple(a + b for a, b in zip(k1, k2))
                    if sum(k) <= self.order:
                        result.terms[k] += v1 * v2
            return result
        else:
            return NotImplemented

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            return gpow(self, n)
        result = GDual.constant(1.0, self.vars, self.order)
        for _ in range(n):
            result = result * self
        return result
    
    def __neg__(self):
        return GDual(self.vars, self.order, {k: -v for k, v in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        if isinstance(other, GDual):
            return self * other.reciprocal()
        elif isinstance(other, (int, float)):
            return self * (1.0 / other)
        else:
            return NotImplemented
        
    def decompose(self):
        f0 = self.terms.get((0,) * len(self.vars), 0.0)
        f_hat = self - GDual.constant(f0, self.vars, self.order)
        return f0, f_hat
    
    def reciprocal(self):
        f0, f_hat = self.decompose()
        if f0 == 0:
            raise ZeroDivisionError("Cannot divide by zero constant term.")
        result = GDual.constant(1, self.vars, self.order)
        base = f_hat / f0
        term = GDual.constant(1.0, self.vars, self.order)  # začnemo z enko
        for k in range(1, self.order + 1):
            term = (-1)**k * (base)**k#term * (-base)  # pravilno množenje z -base
            result += term
        return result / f0

    def __radd__(self, other):
        return self + GDual.constant(other, self.vars, self.order)
    
    def __rsub__(self, other):
        return GDual.constant(other, self.vars, self.order) - self

    def __rmul__(self, other):
        return GDual.constant(other, self.vars, self.order) * self

    def __rtruediv__(self, other):
        return GDual.constant(other, self.vars, self.order) / self

    def multi_factorial(self, multi_index):
        denom = 1
        for c in multi_index:
            denom *= math.factorial(c)
        return denom

    def extract(self):
        derivs = {}
        for k, v in self.terms.items():
            if sum(k) == 0:
                derivs["f"] = v
            else:
                # Naredimo ime odvoda npr. ∂x²y če vars = (x,y)
                name = "∂"
                for var, count in zip(self.vars, k):
                    if count == 1:
                        name += var
                    elif count > 1:
                        name += f"{var}^{count}"
                derivs[name] = v * self.multi_factorial(k)
        return derivs

    def diff(self, key):
        if isinstance(key, int):
            key = (key, )
        return self.terms.get(key, 0.0) * self.multi_factorial(key)
    
    def hessian(self):
        H = []
        m = len(self.vars)
        for i in range(m):
            row = []
            for j in range(m):
                tup = tuple(2 if var == j == i else (1 if var == i or var == j else 0) for var in range(m))
                row.append(self.diff(tup))
            H.append(row)
        return H      
          
    def gradient(self):
        G = []
        m = len(self.vars)
        for i in range(m):
            tup = tuple(1 if i == var else 0 for var in range(m))
            G.append(self.diff(tup))
        return G
    
    def univariate_derivatives(self, variable):
        """ returns a list of derivates with respect to variable starting with f value"""
        D = []
        m = len(self.vars)
        index_var = self.vars.index(variable)
        for i in range(0, self.order + 1):
            tup = [0 for _ in range(m)] # variable can for example be "x" or 1
            tup[index_var] = i
            D.append(self.diff(tuple(tup)))
        return D
    
    def taylor_coeffs(self, order):
        if order > self.order:
            raise ValueError("Order can not be greater that initialized max_order")
        coeffs = {key: v * self.multi_factorial(key) for key, v in self.terms.items() if sum(key) <= order}
        return dict(sorted(coeffs.items()))    
    
    @property
    def value(self):
        return self.terms.get((0,) * len(self.vars), 0.0)

    def __repr__(self):
        if not self.terms:
            return "0"
        terms_str = []
        for k, v in sorted(self.terms.items()):
            if sum(k) == 0:
                terms_str.append(f"{v:.4g}")
            else:
                name = "".join(f"{var}^{count}" if count > 1 else var*count for var, count in zip(self.vars, k))
                terms_str.append(f"{v:.4g}{name}")
        return " + ".join(terms_str)

##########################################
##########################################
##########################################

# --- Useful functions ---

def taylor_eval(coeffs, vars):
    return sum(coeffs[key] * math.prod(vars[i]**key[i] for i in range(len(key))) for key in coeffs.keys())
    
def gsum(X):
    p = X
    for _ in range(1, len(X) + 1):
        p += X
    return p

def prod(X):
    p = X
    for _ in range(1, X.order):
        p *= X
    return p

def gabs(X):
    return X if X >= 0 else - X

def sign(X):
    return 1 if X > 0 else (-1 if X < 0 else 0)

# --- Log functions ---

def exp(X):
    f0, f_hat = X.decompose()
    result = 1
    base = 1
    for k in range(1, X.order + 1):
        base = base * f_hat
        result += base / math.factorial(k)
    return math.exp(f0) * result

def log(X, B=None):
    if B:
        if B > 0 and B != 1:
            if isinstance(B, GDual):
                return log(X) / log(B)
            return log(X) / math.log(B)
    f0, f_hat = X.decompose()
    base = f_hat / f0
    result = GDual.constant(0.0, X.vars, X.order)
    term = base
    for k in range(1, X.order + 1):
        result += ((-1) ** k) * term / k
        term = term * base
    if X <= 0:
        warnings.warn(f"Log: out of domain at point X = {f0}, returning NaN", UserWarning)
        return GDual.constant(float("nan"), X.vars, X.order) - result
    return math.log(f0) - result

def log2(X):
    return log(X, 2)

def log10(X):
    return log(X, 10)

# --- X^Y functions ---

def gpow(X, A):
    if X < 0:
        raise ValueError("Pow: out of domain")
    if isinstance(A, GDual):
        if not isinstance(X, GDual):
            return exp(A * math.log(X))
        return exp(A * log(X))
    def binomial(A, k):
        result = 1.0
        for i in range(k):
            result *= (A - i)
        return result / math.factorial(k)
    f0, f_hat = X.decompose()
    base = f_hat / f0
    result = GDual.constant(0.0, X.vars, X.order)
    term = GDual.constant(1.0, X.vars, X.order)
    for k in range(X.order + 1):
        if k > 0:
            term = term * base
        result += binomial(A, k) * term
    return (f0 ** A) * result   

def sqrt(X):
    if X > 0:
        return gpow(X, 0.5)
    raise ValueError("Sqrt: out of domain")

def cbrt(X):
    return sign(X) * gpow(gabs(X), 1/3)
    
# --- Trig ---

def sin(X):
    f0, f_hat = X.decompose()
    sinf0 = math.sin(f0)
    cosf0 = math.cos(f0)
    result = GDual.constant(0.0, X.vars, X.order)
    for k in range(X.order + 1):
        deg = 2 * k
        if deg <= X.order:
            term = (f_hat ** deg) / math.factorial(deg)
            result += ((-1) ** k) * sinf0 * term
        deg = 2 * k + 1
        if deg <= X.order:
            term = (f_hat ** deg) / math.factorial(deg)
            result += ((-1) ** k) * cosf0 * term
    return result

def cos(X):
    return sin(X + math.pi/2)

def sec(X):
    return 1 / cos(X)

def csc(X):
    return 1 / sin(X)

def tan(X):
    return sin(X) / cos(X)

def cot(X):
    return cos(X) / sin(X)

def atan(X):
    f0, f_hat = X.decompose()
    result = GDual.constant(0.0, X.vars, X.order)
    one_plus_f0_sq = 1 + f0**2
    max_order = X.order
    # Odd sum: term f̂^(2k-1)
    k = 1
    while 2 * k - 1 <= max_order:
        n = 2 * k - 1
        inner_sum = 0.0
        j = 1
        while 2 * j <= n:
            binom = math.comb(n, 2 * j)
            term = binom * (f0 ** (2 * j)) * ((-1) ** j)
            inner_sum += term
            j += 1
        coeff = (1 + inner_sum) / (one_plus_f0_sq ** n)
        term = (f_hat ** n) / n
        result += ((-1) ** (k + 1)) * coeff * term
        k += 1
    # Even sum: term f̂^(2k)
    k = 1
    while 2 * k <= max_order:
        n = 2 * k
        inner_sum = 0.0
        j = 1
        while 2 * j - 1 <= n:
            binom = math.comb(n, 2 * j - 1)
            term = binom * (f0 ** (2 * j - 1)) * ((-1) ** (j + 1))
            inner_sum += term
            j += 1
        coeff = inner_sum / (one_plus_f0_sq ** n)
        term = (f_hat ** n) / n
        result += ((-1) ** k) * coeff * term
        k += 1
    return math.atan(f0) + result

def asin(X):
    if -1 <= X <= 1:
        return atan(X / sqrt(1 - X**2))
    raise ValueError("Asin: out of domain")

def acos(X):
    if -1 <= X <= 1:
        return math.pi/2 - asin(X)
    raise ValueError("Acos: out of domain")

def acot(X):
    if X != 0:
        return atan(1 / X)
    raise ValueError("Acoth: out of domain")

# --- Hyperbolic ---

def sinh(X):
    return (exp(X) - exp(-X)) / 2

def cosh(X):
    return (exp(X) + exp(-X)) / 2

def tanh(X):
    return sinh(X) / cosh(X)

def coth(X): 
    return cosh(X) / sinh(X)

def sech(X):
    return 1 / cosh(X)

def csch(X):
    return 1 / sinh(X)

def asinh(X):
    return log(X + sqrt(X**2 + 1))

def acosh(X):
    if X >= 0:
        return log(X + sqrt(X**2 - 1))
    raise ValueError("Acosh: out of domain")

def atanh(X):
    if gabs(X) >= 1:
        f0, _ = X.decompose()
        warnings.warn(f"Atanh: out of domain at point X = {f0}, returning NaN", UserWarning)
    return 0.5 * log((1 + X) / (1 - X))

def acoth(X):
    if gabs(X) <= 1:
        f0, _ = X.decompose()
        warnings.warn(f"Acoth: out of domain at point X = {f0}, returning NaN", UserWarning)
    return 0.5 * log((X + 1) / (X - 1))

def asech(X):
    if not (0 < X < 1):
        f0, _ = X.decompose()
        warnings.warn(f"Asech: out of domain at point X = {f0}, returning NaN", UserWarning)
    return log((1 + sqrt(1 - X**2)) / X)

def acsch(X):
    if X == 0:
        f0, _ = X.decompose()
        warnings.warn(f"Acsch: out of domain at point X = {f0}, returning NaN", UserWarning)
    return log(1/X + sqrt(1/X**2 + 1))

# --- Other ---

def erf(X):
    f0, f_hat = X.decompose()
    def hermite_poly(n, x):
        if n == 0:
            return 1.0
        elif n == 1:
            return 2 * x
        else:
            Hnm2 = 1.0
            Hnm1 = 2 * x
            for i in range(2, n + 1):
                Hn = 2 * x * Hnm1 - 2 * (i - 1) * Hnm2
                Hnm2, Hnm1 = Hnm1, Hn
            return Hn
    result = GDual.constant(0.0, X.vars, X.order)
    base = 1
    for k in range(1, X.order + 1):
        df = (- 1)**(k - 1) * hermite_poly(k - 1, f0) * math.exp(- f0**2)
        base = base * f_hat
        result += df * base / math.factorial(k)
    return math.erf(f0) + 2 * result / math.sqrt(math.pi)



