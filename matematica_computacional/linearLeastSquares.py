

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import mibiblioteca as bib


'''
# define true model parameters
x = np.linspace(-1, 1, 100) # intervalo sobre el cual efectuamos el experimento
a, b, c = 1, 2, 150
y_exact = a + b * x + c * x**2

# simulate noisy data
m = 20
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 +2*np.random.randn(m)

# fit the data to the model using linear least square
A = np.vstack([X**0, X**1, X**2]) # see np.vander for alternative
sol, r, rank, sv = la.lstsq(A.T, Y)
'''
'''
At = np.array([X**0,X**1,X**2])
auxMat = np.matmul(At,At.T)
np.reshape(Y,(m,1))
b = np.matmul(At,Y)
sol = bib.GaussElimPiv(auxMat,b)
'''

'''
y_fit = sol[0] + sol[1] * x + sol[2] * x**2
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'r', lw=2, label='True value $y = 1 + 2x + 3x^2$')
ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.legend(loc=2)
plt.show()
'''


'''
# Define true model parameters
x = np.linspace(-1, 1, 100)  # Intervalo sobre el cual efectuamos el experimento
a, b, c = 1, 2, 150
y_exact = a + b * x + c * x**2

# Simulate noisy data
m = 20
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 + 2 * np.random.randn(m)

# Construcción explícita de la matriz A usando arrays de numpy
A = np.array([np.ones(m), X, X**2]).T
print("Matriz A:\n", A)

# Resolvemos el sistema usando np.linalg.solve
# Primero calculamos la matriz A^T A y el vector A^T Y
AtA = np.dot(A.T, A)
AtY = np.dot(A.T, Y)

# Solucionamos para los coeficientes del polinomio
sol = np.linalg.solve(AtA, AtY)
print("Solución (coeficientes del polinomio):", sol)

# Evaluamos el polinomio ajustado
y_fit = sol[0] + sol[1] * x + sol[2] * x**2

# Plot results
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exact, 'r', lw=2, label='True value $y = 1 + 2x + 3x^2$')
ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.legend(loc=2)
plt.show()

'''


# Define true model parameters
x = np.linspace(-1, 1, 100)  # Intervalo sobre el cual efectuamos el experimento
a, b, c = 1, 2, 150
y_exacto = a + b * x + c * x**2

# Simulate noisy data
m = 20
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 + 2 * np.random.randn(m)

# Construcción explícita de la matriz A usando arrays de numpy
A = np.array([np.ones(m), X, X**2]).T
print("Matriz A:\n", A)

# Calculamos la matriz A^T A y el vector A^T Y
AtA = np.dot(A.T, A)
AtY = np.dot(A.T, Y)

print("Matriz A^T A:\n", AtA)
print("Vector A^T Y:\n", AtY)

# Resolvemos el sistema usando np.linalg.solve
sol = np.linalg.solve(AtA, AtY)
print("Solución (coeficientes del polinomio):", sol)

# Evaluamos el polinomio ajustado
y_ajustado = sol[0] + sol[1] * x + sol[2] * x**2

# Plot results
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
ax.plot(x, y_exacto, 'r', lw=2, label='True value $y = 1 + 2x + 150x^2$')
ax.plot(x, y_ajustado, 'b', lw=2, label=f'Least square fit $y = {sol[0]:.2f} + {sol[1]:.2f}x + {sol[2]:.2f}x^2$')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.legend(loc=2)
plt.show()
