import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
print("----------------------------------- Task 3 - Quadratic curves and surfaces -----------------------------------")
# ----------------------------------------------------------------------------------------------------------------------

# Define the variables
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Define the equation
equation = 2*x1**2 - x2**2 + 2*x3**2 - 10*x1*x2 - 4*x1*x3 + 10*x2*x3 - 1

# Solve for x3 in terms of x1 and x2
solutions = sp.solve(equation, x3)

print(solutions)

# f(x1, x2) = x3
def x3(x1, x2):
    sol1 = x1 - (5*x2)/2 - (27*x2**2 + 2)**0.5/2
    sol2 = x1 - (5*x2)/2 + (27*x2**2 + 2)**0.5/2
    return sol1, sol2


x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)

X, Y = np.meshgrid(x, y)
Z = x3(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot the two solutions with distinct colormaps
ax.contour3D(X, Y, Z[0], 50, cmap='Blues')
ax.contour3D(X, Y, Z[1], 50, cmap='Reds')

# Set axis labels
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

# Create proxy artists for the legend with the custom colors
proxy1 = plt.Line2D([0], [0], linestyle='none', c='blue', marker='.')
proxy2 = plt.Line2D([0], [0], linestyle='none', c='red', marker='.')

ax.legend([proxy1, proxy2], ['Solution 1', 'Solution 2'], numpoints=1)

plt.show()