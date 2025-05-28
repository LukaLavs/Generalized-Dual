import math

class NDualNumber:
    def __init__(self, coeffs, n=None):
        if n:
            # Then coeffs is a, point for derivate, construct Ndual[a, 1, 0, ..., 0]
            a = coeffs
            self.coeffs = list([a, 1] + [0] * (n - 1))
            self.n = n
        else:
            self.coeffs = list(coeffs)
            self.n = len(coeffs) - 1
    
    def __lt__(self, other):
        if isinstance(other, NDualNumber):
            return self.coeffs[0] < other.coeffs[0]
        else:
            return self.coeffs[0] < other

    def __gt__(self, other):
        if isinstance(other, NDualNumber):
            return self.coeffs[0] > other.coeffs[0]
        else:
            return self.coeffs[0] > other
        
    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def __eq__(self, other):
        if isinstance(other, NDualNumber):
            return self.coeffs == other.coeffs
        else:
            return self.coeffs[0] == other

    def __ne__(self, other):
        return not self == other
    
    def __add__(self, other):
        if isinstance(other, NDualNumber):
            return NDualNumber([a + b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            coeffs = self.coeffs[:]
            coeffs[0] += other
            return NDualNumber(coeffs)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, NDualNumber):
            return NDualNumber([a - b for a, b in zip(self.coeffs, other.coeffs)])
        else:
            coeffs = self.coeffs[:]
            coeffs[0] -= other
            return NDualNumber(coeffs)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return NDualNumber([-a for a in self.coeffs])

    def __mul__(self, other):
        if isinstance(other, NDualNumber):
            n = self.n
            result = [0.0] * (n + 1)
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    result[i + j] += self.coeffs[i] * other.coeffs[j]
            return NDualNumber(result)
        else:
            return NDualNumber([a * other for a in self.coeffs])

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        return pow(self, other)
    
    def __truediv__(self, other):
        if isinstance(other, NDualNumber):
            # Division via Newton method or recursive formula
            n = self.n
            a = self.coeffs
            b = other.coeffs
            q = [0.0] * (n + 1)
            q[0] = a[0] / b[0]
            for k in range(1, n + 1):
                s = a[k]
                for j in range(1, k + 1):
                    s -= b[j] * q[k - j] if j <= n and k - j <= n else 0.0
                q[k] = s / b[0]
            return NDualNumber(q)
        else:
            return NDualNumber([a / other for a in self.coeffs])

    def __rtruediv__(self, other):
        return NDualNumber([other]) / self

    def __repr__(self):
        return f"NDual({self.coeffs})"
    
    def derivate(self, m):
        n = self.n
        if m == "all":
            return [self.coeffs[k] * math.factorial(k) for k in range(0, n + 1)]
        if 0 <= m <= n:
            return self.coeffs[m] * math.factorial(m)
        
        
########################
#######################
########################




def to_dual(x, n):
    """ n is the number of derivates in ndual """
    return NDualNumber([x] + [0] * n)

def evaluate_f(derivs, X):
    """Evaluate f(X), where X is an NDualNumber, using Taylor expansion."""
    n = X.n
    a = X.coeffs[0]
    B = NDualNumber([0.0] + X.coeffs[1:])
    result = NDualNumber([0.0] * (n + 1))
    B_power = NDualNumber([1.0] + [0.0] * n)  # B^0 = 1
    for k in range(n + 1):
        term_coeff = derivs[k](a) / math.factorial(k)
        term = B_power * term_coeff
        result = result + term
        B_power = B_power * B  # B^k → B^{k+1}
    return result

# --- Trigonometry ---

def sin(X):
    # X as a dual number
    n = X.n
    derivs = []
    sign = -1
    for k in range(0, n + 1):
        if k % 2 == 0:
            sign *= -1
            derivs.append(lambda x, s=sign: s * math.sin(x))
        else:
            derivs.append(lambda x, s=sign: s * math.cos(x))
    return evaluate_f(derivs, X)

def cos(X):
    n = X.n
    derivs = []
    sign = 1
    for k in range(n + 1):
        if k % 2 == 0:
            derivs.append(lambda x, s=sign: s * math.cos(x))
            sign *= -1
        else:
            derivs.append(lambda x, s=sign: s * math.sin(x))
    return evaluate_f(derivs, X)

def tan(X):
    return sin(X) / cos(X)

def cot(X):
    return cos(X) / sin(X)

# --- Hyperbolic functions ---

def cosh(X):
    return (exp(X) + exp(-X)) / 2

def sinh(X):
    return (exp(X) - exp(-X)) / 2

def tanh(X):
    return sinh(X) / cosh(X)

def coth(X):
    return cosh(X) / sinh(X)

# --- Inverse Hyperbolic Functions ---

def asinh(X):
    return log(X + sqrt(X*X + 1))

def acosh(X):
    if X >= 0:
        return log(X + sqrt(X*X - 1))
    raise ValueError(f"Error in domain for acosh: X = {X.coeffs[0]}")

def atanh(X):
    if abs(X) < 1:
        return 0.5 * log((1 + X) / (1 - X))
    raise ValueError(f"Error in atanh domain: X = {X.coeffs[0]}")

def acoth(X):
    if abs(X) > 1:
        return 0.5 * log((X + 1) / (X - 1))
    raise ValueError(f"Error in acoth domain: X = {X.coeffs[0]}")

# --- Exponential functions --- 

def exp(X):
    n = X.n
    derivs = [math.exp] * (n + 1)
    return evaluate_f(derivs, X)

def expm1(X):
    return exp(X) - 1
    
def sqrt(X):
    return pow(X, 0.5)

def Sfalling_factorial(X, n):
    """ Does not work if n is a dual number"""
    p = X
    for i in range(1, n):
        p *= X - i
    return p

def pow(X, Y):
    if isinstance(X, NDualNumber) and X.coeffs[0] == 0:
        n = X.n
        if isinstance(Y, (int, float)):
            if Y == 0:
                return to_dual(1, n)
            if (Y - n) >= 0:
                derivs = [lambda x, i=i: x**(Y - i) * Sfalling_factorial(Y, i) 
                            for i in range(0, n + 1)]
                return evaluate_f(derivs, X)
            raise ValueError("pow is not defined: Derivates don't exist")
        if isinstance(Y, NDualNumber):
            if Y.coeffs[0] == X.coeffs[0] == 0:
                raise ValueError("pow not defined")
    return exp(Y * log(X))

def cbrt(X):
    return pow(X, 1/3)
        
# --- Logarithms ---

def log(X, base="e"):
    if base != "e" and base > 0 and base != 1:
        return log(X) / math.log(base)
    n = X.n
    derivs = [lambda x: math.log(x)] + [lambda x, s=(-1)**(k-1), f=math.factorial(k-1), k=k: s * f * x**(-k) for k in range(1, n+1)]
    return evaluate_f(derivs, X)

def log1p(X):
    return log(1 + X)

def log2(X):
    return log(X, 2)

def log10(X):
    return log(X, 10)

# --- Basic ---
def abs(X):
    return X if X >= 0 else - X

def prod(L):
    p = 1
    for X in L:
        p *= X
    return p

def sum(L):
    p = 0
    for X in L:
        p += X
    return p

def dist(L):
    return sqrt(sum(X**2 for X in L))

def sumprod(L1, L2):
    return sum(X * Y for (X, Y) in zip(L1, L2))
