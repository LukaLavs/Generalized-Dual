import numpy as np
from .utils import initialize

def integrate(func, A, B, N, m):
    def integrate_box(alpha, h):
        # ∫_{-h/2}^{h/2} x^alpha dx = (b^{alpha+1} - a^{alpha+1}) / (alpha+1), separable product form
        a = -np.array(h) / 2
        b = -a
        return np.prod([(b[i]**(a_+1) - a[i]**(a_+1)) / (a_+1) for i, a_ in enumerate(alpha)])
    h = [(B[i] - A[i]) / N[i] for i in range(len(N))]
    grids = np.meshgrid(*[np.linspace(A[i] + h[i]/2, B[i] - h[i]/2, N[i]) for i in range(len(N))], indexing="ij")
    dual_vars = initialize(*grids, m=m)
    dual_eval = func(*dual_vars)
    return sum(
        np.sum(v * integrate_box(alpha, h))
        for alpha, v in dual_eval.terms.items()
    )