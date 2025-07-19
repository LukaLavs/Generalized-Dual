import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from generalized_dual import *

import numpy as np
import mpmath
from matplotlib import pyplot as plt




import mpmath
import numpy as np
import matplotlib.pyplot as plt

# # Set desired precision for mpmath (used for high-accuracy math operations)
# mpmath.mp.dps = 15

# # Generate input data: X is linspace, Y = sin(X) + 1, Z = cos(X)
# X = np.linspace(0, 3, 5)
# Y = np.sin(X) + 1
# Z = np.cos(X)

# # Initialize dual-number variables for automatic differentiation up to order m
# x, y, z = initialize(X, Y, Z, m=3)

# # Define the target function: |LambertW(sin(x*y + z))| on branch -1
# F = dual_abs(lambertw(sin(x * y + z), branch=-1))

# # Select the index `p` where we want to center our Taylor expansion
# p = 1  # Taylor will be centered at (X[p], Y[p], Z[p])

# # Create a dense set of x-values to evaluate and plot the true and Taylor-approximated functions
# X_test = np.linspace(-2 * np.pi, 2 * np.pi, 300)

# # Define the true function (non-dual, scalar) for comparison
# F_not_dual = lambda x: float(mpmath.fabs(mpmath.lambertw(mpmath.sin(x * Y[p] + Z[p]), k=-1)))
# Y_true = np.vectorize(F_not_dual)(X_test)

# # Evaluate the true function at the expansion center
# X_point = X[p]
# Y_point = F_not_dual(X_point)

# # Build Taylor approximations centered at each (X[i], Y[i], Z[i])
# taylor_funcs = build_taylor(F, X, Y, Z, to_float=True)

# # Select the Taylor function centered at the desired point (X[p], Y[p], Z[p])
# taylf = taylor_funcs[p]

# # Evaluate the Taylor approximation as a function of x, holding y and z fixed at center values
# Y_taylor = taylf([X_test, None, None])

# # Plot both the true function and its Taylor approximation
# plt.plot(X_test, Y_true, label='True function')
# plt.plot(X_test, Y_taylor, label='Taylor approximation')
# plt.scatter(X_point, Y_point, color='red', label='Expansion point')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Taylor approximation of |LambertW(sin(x*y + z))| (branch -1)')
# plt.legend()
# plt.grid(True)
# plt.show()




# mpmath.mp.dps = 15
# X = 2
# x = initialize(X, m=10)


# F = sin(cos(x**2) + atan(x)) + sin(4*x)
# f = lambda x: np.sin(np.cos(x**2) + np.atan(x)) + np.sin(4*x)

# taylor = build_taylor(F, X, to_float=True)

# X_range = np.linspace(0, 3, 100)
# Y_fun = f(X_range)
# Y_taylor = taylor([X_range])
# X_point = X
# Y_point = f(X)

# plt.plot(X_range, Y_fun, label='func')
# plt.plot(X_range, Y_taylor, label='taylor')
# plt.scatter(X_point, Y_point)
# plt.ylim(-3, 3)
# plt.show()











# from generalized_dual import *
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# # Setup
# X, Y = 2, 3
# x, y = initialize(X, Y, m=3)
# F = sin(x * y)
# f = lambda x, y: np.sin(x * y)
# taylor = build_taylor(F, X, Y, to_float=True) # build taylor function around (X, Y)
# # If X, Y were ndarrays the result would be a ndarray of functions around respected centers

# # Grid
# x_vals = np.linspace(0, 4, 100)
# y_vals = np.linspace(0, 4, 100)
# X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# # Evaluate
# Z_fun = f(X_grid, Y_grid)
# Z_taylor = taylor([X_grid, Y_grid]) # Evaluate taylor polinom around (X, Y)
# Z_point = f(X, Y)

# # Plot
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X_grid, Y_grid, Z_fun, cmap='viridis', alpha=0.5)
# ax.plot_surface(X_grid, Y_grid, Z_taylor, cmap='plasma', alpha=0.5)
# ax.scatter(X, Y, Z_point, color='red', s=40)

# ax.set_title("Function vs. Taylor Approximation")
# ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

# legend_elements = [
#     Line2D([0], [0], color='blue', lw=3, label='Function'),
#     Line2D([0], [0], color='orange', lw=3, label='Taylor'),
#     Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Center')
# ]
# ax.legend(handles=legend_elements, loc='upper left')

# plt.tight_layout()
# plt.show()






