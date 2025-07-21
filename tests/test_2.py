import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




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