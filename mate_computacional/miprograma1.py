import numpy as np

import mibiblioteca1 as bib
# Paso 1: Generar una matriz aleatoria A y el vector b
np.random.seed(0)  # Para reproducibilidad
A = np.random.uniform(-10, 10, (10, 10))
x_exacta = np.ones((10, 1))  # Solución exacta x = [1, 1, ..., 1]
b = np.dot(A, x_exacta)

# Paso 2: Imprimir la matriz de coeficientes y el vector de términos independientes
print("Matriz de coeficientes A:")
print(A)
print("\nVector de términos independientes b:")
print(b)

# Paso 3: Resolución por eliminación gaussiana simple
# Matriz aumentada
Ab = np.append(A, b, axis=1)
print("\nMatriz aumentada [A|b]:")
print(Ab)

# Escalonar la matriz aumentada
bib.escalonaSimple(Ab)
print("\nMatriz aumentada escalonada:")
print(Ab)

# Resolver el sistema
A1 = Ab[:, :10]
b1 = Ab[:, 10]
b1 = b1.reshape(b1.shape[0], 1)
x = bib.sustRegresiva(A1, b1)

print("\nSolución del sistema:")
print(x)

# Calcular la norma suma del residuo
residuo = b - np.dot(A, x)
norma_suma_residuo = np.sum(np.abs(residuo))

print("\nNorma suma del residuo:")
print(norma_suma_residuo)