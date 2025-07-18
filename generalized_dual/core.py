import mpmath
import numpy as np
from collections import defaultdict

class GeneralizedDual:
    
    # == Definition and Basic Arithmetic == 
    
    def __init__(self, terms=None, n=1, m=1):
        """ n -> number of variables 
            m -> order of interest
            p -> percision
        """
        self.terms = defaultdict(lambda: 0)
        if terms: self.terms.update(terms)
        self.n = n
        self.m = m
        
    def _decompose(self):
        f0 = self.terms.get((0,) * self.n, 0)
        f_hat = self - f0
        return f0, f_hat
    
    @staticmethod
    def _compose(f0, f_hat):
        result = f_hat + f0
        return result
    
    @staticmethod
    def _constant(C, size_like):
        key0 = (0,)*size_like.n
        one = size_like.terms[key0] * 0 + 1
        return GeneralizedDual(terms={key0: C*one}, n=size_like.n, m=size_like.m)
    
    @staticmethod
    def _safe_inversion(f0):
        """ returns 1 / f0, handles 1/0 cases
        """
        try: return 1 / f0
        except ZeroDivisionError: 
            return np.vectorize(lambda x: 1 / x if x != 0 else mpmath.nan, otypes=[object])(f0)
        
    @staticmethod
    def _mp_comb(n, k):
        return mpmath.gamma(n+1) / (mpmath.gamma(k+1) * mpmath.gamma(n - k + 1))
    
    def _default_zero(self):
        key0 = (0,)*self.n
        zero = self.terms[key0] * 0
        return zero
    
    def __add__(self, other):
        if isinstance(other, GeneralizedDual):
            result_terms = dict()
            for key in set(self.terms) | set(other.terms):
                result_terms[key] = self.terms.get(key, 0) + other.terms.get(key, 0)
            return GeneralizedDual(result_terms, self.n, self.m)
        else:
            key0 = (0,)*self.n
            result_terms = self.terms.copy()
            result_terms[key0] = self.terms[key0] + other
        return GeneralizedDual(result_terms, self.n, self.m)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return GeneralizedDual(terms={key: - val for key, val in self.terms.items()}, n=self.n, m=self.m)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return - self + other
    
    def __mul__(self, other):
        if isinstance(other, GeneralizedDual):
            result = GeneralizedDual._constant(0, size_like=other)
            for k1, v1 in self.terms.items():
                for k2, v2 in other.terms.items():
                    k = tuple(a + b for a, b in zip(k1, k2))
                    if sum(k) <= self.m:
                        result.terms[k] += v1 * v2
            return result
        else:
            result = GeneralizedDual({key: val * other for key, val in self.terms.items()}, n=self.n, m=self.m)
            return result
        
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, n):
        if isinstance(n, int):
            result = GeneralizedDual._constant(1, size_like=self)
            for _ in range(n):
                result = result * self
            return result
        raise NotImplemented("Use dual_pow or nroot instad.")
        
    def _reciprocal(self):
        """ calculates GeneralizedDual^-1
        """
        f0, f_hat = self._decompose()
        f_inv = GeneralizedDual._safe_inversion(f0)
        base = f_hat * f_inv    
        base_power = 1
        result = GeneralizedDual._constant(1, size_like=self)
        term = GeneralizedDual._constant(1, size_like=self)
        for k in range(1, self.m + 1):
            base_power *= base
            term = (-1)**k * base_power
            result += term
        return result * f_inv
        
    def __truediv__(self, other):
        if isinstance(other, GeneralizedDual):
            return self * other._reciprocal()
        else:
            return self * (1 / other)
    
    def __rtruediv__(self, other):
        return self._reciprocal() * other
    
    # == Comparisons ==
    
    def __eq__(self, other):
        key0 = (0,)*self.n
        if isinstance(other, GeneralizedDual):
            return np.all(self.terms[key0] == other.terms[key0])
        else:
            return np.all(self.terms[key0] == other)
        
    def __ne__(self, other):
        key0 = (0,)*self.n
        if isinstance(other, GeneralizedDual):
            return np.all(self.terms[key0] != other.terms[key0])
        else:
            return np.all(self.terms[key0] != other)
    
    def __lt__(self, other):
        key0 = (0,)*self.n
        if isinstance(other, GeneralizedDual):
            return np.all(self.terms[key0] < other.terms[key0])
        else:
            return np.all(self.terms[key0] < other)
        
    def __le__(self, other):
        key0 = (0,)*self.n
        if isinstance(other, GeneralizedDual):
            return np.all(self.terms[key0] <= other.terms[key0])
        else:
            return np.all(self.terms[key0] <= other)
        
    def __gt__(self, other):
        key0 = (0,)*self.n
        if isinstance(other, GeneralizedDual):
            return np.all(self.terms[key0] > other.terms[key0])
        else:
            return np.all(self.terms[key0] > other)
        
    def __ge__(self, other):
        key0 = (0,)*self.n
        if isinstance(other, GeneralizedDual):
            return np.all(self.terms[key0] >= other.terms[key0])
        else:
            return np.all(self.terms[key0] >= other)
    
    # == Display ==
               
    def __repr__(self):
        disp_vec = np.vectorize(lambda x: mpmath.mp.nstr(x, mpmath.mp.dps))
        disp_terms = {key: disp_vec(val) for key, val in self.terms.items()}
        return f"{type(self).__name__}({disp_terms})"

    # == Basic Methods ==
    
    def diff(self, key, to_float=None):
        if isinstance(key, int): key = (key,)
        result = self.terms.get(key, self._default_zero()) * mpmath.fprod([mpmath.factorial(k) for k in key])
        if to_float:
            result = np.vectorize(lambda x: float(x) if not mpmath.isnan(x) else float("nan"))(result)
        return result    
    
    def gradient(self):
        G = []
        for i in range(self.n):
            key = tuple(1 if var == i else 0 for var in range(self.n))
            G.append(self.diff(key))
        return G
    
    def hessian(self):
        H = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                key = tuple(2 if var == j == i else (1 if var == i or var == j else 0) for var in range(self.n))
                row.append(self.diff(key))
            H.append(row)
        return H    
    
    def derivatives_along(self, var):
        D = []
        for i in range(self.m + 1):
            key = tuple(i if var == j else 0 for j in range(self.n))
            D.append(self.diff(key))
        return D