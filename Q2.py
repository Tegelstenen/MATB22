import numpy as np
import matplotlib.pyplot as plt

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
