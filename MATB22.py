import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import sympy as sp


# ---------------------------------------------------------------------------------------------------------------------
print("\n-------------------------- Task 1 - Least squares, normal equation, minimization ---------------------------")
# ---------------------------------------------------------------------------------------------------------------------

# ***
# *** Solve this task by setting up and solving the normal equations in Python.
# ***

A = np.array([[1, 1, 2],
              [1, 2, 1],
              [2, 1, 1],
              [2, 2, 1]])

b = np.array([1, -1, 1, -1])


# calculates the
def normalEquation(matrix, vector):
    # A^{T}
    matTrans = matrix.transpose()

    # AA^{T}
    matNew = np.matmul(matTrans, matrix)

    # (AA^{T})^{-1}
    matInv = np.linalg.inv(matNew)

    # A^{T}y
    vecNew = np.matmul(matTrans, vector)

    # (AA^{T})^{-1}A^{T}y
    approx = np.matmul(matInv, vecNew)

    return approx


# ***
# *** Use scipy.optimize.fmin to directly solve the minimization problem. Compare results.
# ***


def error(x, matrix, vector):
    # Function to take a vector, x, and return the norm, ||Ax-b||
    Ax = np.dot(matrix, x)
    diff = Ax - vector
    return np.linalg.norm(diff)


initialGuess = np.array([0, 0, 0])  # Can be anything

normEqRes = normalEquation(A, b)
fminRes = fmin(error, initialGuess, args=(A, b), disp=0)

print("The approximation by the normal equation: ", normEqRes)
print("The approximation by fmin: ", fminRes)

# ***
# *** Replace b by b(a) = (1, a, 1, a)^{T} and plot the residual for each r(a) = ∥Ax(a) − b(a)∥ versus a.
# *** Note the dependence of x on a in this subtask.
# ***

a = range(-20, 21)


def b(values):
    # creates a simple vector with changing elements in second and fourth index.
    # Iterates over all values in the range of a and provides a new output vector for each.
    newVectors = np.empty((len(values), 4))

    i = 0
    for val in values:
        newVectors[i] = np.array([1, val, 1, val])
        i += 1

    return newVectors


def r(values, matrix):
    # Makes b dependent on a
    bNew = b(values)

    # Makes a dependent on a
    x = np.empty((len(bNew), 3))

    i = 0
    for vec in bNew:
        x[i] = normalEquation(A, vec)
        i += 1

    # Finding the residuals using the norm
    residuals = np.empty(len(values))

    for i in range(len(values)):
        residuals[i] = error(x[i], matrix, bNew[i])

    return residuals


plt.figure()
plt.plot(a, r(a, A))
plt.xlabel("a")
plt.ylabel("Residuals")
plt.title("Residuals vs a")
plt.grid(True)
plt.show()

# ***
# *** Does this curve have a zero?
# *** If yes, what does this mean in the context of solving over-determined linear systems?
# ***

# The curve does not have a zero. Given that the curve has a minimum at a = 0, it implies that for there does not
# exist a number for a such that the system is consistent.


# ---------------------------------------------------------------------------------------------------------------------
print("\n------------------------- Task 2 - Eigenvalues, eigenvectors, recurrence relations -------------------------")
# ---------------------------------------------------------------------------------------------------------------------

# ---------- Do the iterates converge? If so, determine the limit. ----------

# Equation-system
A = np.array([[1, 3, 2],
              [-3, 4, 3],
              [2, 3, 1]])

# Finding eigenvalues and their respective eigenvectors.
eigenValues, eigenVectors = np.linalg.eig(A)

# Solving for coefficients.
initialCondition = np.array([8, 3, 12])
coefficients = np.linalg.solve(eigenVectors, initialCondition)
print("\nCoefficients are: ", coefficients)


def z_n(n):
    return np.array(
        [sum(coefficients[i] * np.exp(-eigenValues[i] * n) * eigenVectors[j][i] for i in range(3)) for j in range(3)])


# Testing initial condition fulfilled.
if any(np.round(z_n(0)) != initialCondition):
    raise Exception("Incorrect coefficients")

# Testing for convergence.
print("\nLimit for z_n")
for n in range(0, 1000, 100):
    print(z_n(n))
print("Conclusion: each iterate diverges.")
print("\t\t\tThe limits are a -> -oo, b -> -oo, c -> oo")

# ---------- Do the Normalized iterates converge? If so, determine the limit. ----------
def v_n(n):
    current_z = initialCondition
    for j in range(n):
        next_z = np.dot(A, (current_z / np.linalg.norm(current_z)))
        current_z = next_z
    return current_z

print("\nLimit for v_n")
for n in range(0, 50, 5):
    print(v_n(n))
print("Conclusion: Each iterate converges.")
print("\t\t\tThe limits are a -> 2.75, b -> 0.92, c -> 2.75")

# ---------- Which property do the limits z and v have (if they exist)? ----------
# The property that z has is that it diverges due to the presence of an eigenvalue greater than 1 in magnitude.
# The property that v has is that while the magnitude of z might grow unbounded, its direction stabilizes.
# This means that the iterates, when normalized, will tend to a specific direction in 3D space.
# Over time, as z grows, its direction (v) becomes stable if there's a dominant eigenvalue directing its growth.

# ---------- Compute for every normalized iterate the quantity q = v^{t}Av. Determine the limit q. ----------
q_n = lambda n: np.dot(v_n(n).T, np.dot(A, v_n(n)))
for n in range(0, 50, 5):
    print(q_n(n))
print("Conclusion: The serie converges.")
print("\t\t\tThe limit is q_n -> 64")

# ---------- What quantity does q represent with respect to the matrix A? ----------

# It is a quadratic form wrt A.

# ---------- How many iterates do you need to fulfil ∥vn − v∥ < ε for ε = 10−8? ----------
def iterationsNeeded(epsilon, func):
    limit = func(50)
    biggerThanEpsilon = True
    i = 0
    while biggerThanEpsilon:
        diff = np.abs(func(i) - limit)
        if np.size(diff) > 1:
            biggerThanEpsilon = all(diff >= epsilon)
        else:
            biggerThanEpsilon = diff >= epsilon
        i += 1

    return i


print("\nWe need ", iterationsNeeded(10 ** (-8), v_n), " iterations for v_n to be bigger than ε.")

# ---------- Vary ε between 0.1 and 10−14, plot the number of iterates versus ε in a semi log plot ----------
epsilonRange = np.linspace(10 ** (-14), 0.1, 20)

iterations_v = np.zeros(20)
iterations_q = np.zeros(20)
for i, epsilon in enumerate(epsilonRange):
    iterations_v[i] = iterationsNeeded(epsilon, v_n)
    iterations_q[i] = iterationsNeeded(epsilon, q_n)

plt.grid(True, which="both")
plt.semilogy(epsilonRange, iterations_v, label="v")
plt.semilogy(epsilonRange, iterations_q, label="q")
plt.xlabel("ε")
plt.ylabel("Number of Iterates")
plt.legend()
plt.show()

# v_n converges faster


# ----------------------------------------------------------------------------------------------------------------------
print("\n---------------------------------- Task 3 - Quadratic curves and surfaces ----------------------------------")
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