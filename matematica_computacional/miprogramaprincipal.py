import numpy as np
import mibiblioteca1 as bib
import time
#B = np.array([[2.1,3.5,4.],[4.,5.,6.],[7.,8.,9.]])
#b = np.array([[9],[15],[24]])
#x = bib.GaussElimSimple(B,b)
#print(x)
#res = np.matmul(B,x)-b
#print(res)

#A = np.random.rand(10,10)
#print(A)
#B=np.arange(-10,10,1)
#print(B)
#b = np.random.rand(10,1)
#print(b)
#initt = time.time()
#x = bib.GaussElimSimple(A,b)
#endt = time.time()
#ttransc = endt - initt
#print(ttransc)

#A=2*np.random.rand(5,5)-1
#print("B=\n",B)
#bib.escalonaSimple(A)
#print("B=\n",A)

#print("B=\n",B)


# Crear una matriz A de 10x10 con valores aleatorios
#A = np.random.rand(10, 10)
#print("Matriz A:")
#print(A)

# Crear un vector b de 10x1 con valores aleatorios
#b = np.random.rand(10, 1)
#print("Vector b:")
#print(b)

# Medir el tiempo de ejecución de la eliminación de Gauss
#initt = time.time()
#x = bib.GaussElimSimple(A, b)
#endt = time.time()

# Tiempo transcurrido
#ttransc = endt - initt
#print("Tiempo transcurrido:")
#print(ttransc)

# Solución del sistema
#print("Solución del sistema:")
#print(x)
import numpy as np
import mibiblioteca as bib
#import time
'''
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
##############################3
import numpy as np
import mibiblioteca1 as bib
#import time
from tabulate import tabulate

# Paso 1: Generar una matriz aleatoria A y el vector b
#np.random.seed(0)  # Para reproducibilidad
A = np.random.uniform(-10, 10, (10, 10))
x_exacta = np.ones((10, 1))  # Solución exacta x = [1, 1, ..., 1]
b = np.dot(A, x_exacta)

# Paso 2: Imprimir la matriz de coeficientes y el vector de términos independientes
print("Matriz de coeficientes A:")
print(tabulate(A, tablefmt="fancy_grid"))

print("\nVector de términos independientes b:")
print(tabulate(b, tablefmt="fancy_grid"))

# Paso 3: Resolución por eliminación gaussiana simple
# Matriz aumentada
Ab = np.append(A, b, axis=1)
print("\nMatriz aumentada [A|b]:")
print(tabulate(Ab, tablefmt="fancy_grid"))

# Escalonar la matriz aumentada
bib.escalonaSimple(Ab)
print("\nMatriz aumentada escalonada:")
print(tabulate(Ab, tablefmt="fancy_grid"))

# Resolver el sistema
A1 = Ab[:, :10]
b1 = Ab[:, 10]
b1 = b1.reshape(b1.shape[0], 1)
x = bib.sustRegresiva(A1, b1)

print("\nSolución del sistema:")
print(tabulate(x, tablefmt="fancy_grid"))

# Calcular la norma suma del residuo
residuo = b - np.dot(A, x)
norma_suma_residuo = np.sum(np.abs(residuo))

print("\nNorma suma del residuo:")
print(norma_suma_residuo)
'''

# Ejemplo de uso extraido de  https://es.wikipedia.org/wiki/Factorizaci%C3%B3n_QR
A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
Q,R = bib.QRdecomposition(A)
print("Q:\n", Q)
print("R:\n", R)

# Comprobación de la ortogonalidad de Q
aux = np.matmul(Q, Q.transpose()) - np.eye(Q.shape[0])
print("Auxiliar (Q * Q^T - I):\n", aux)
norma = np.linalg.norm(aux)
print("Norma de la matriz auxiliar:", norma)

# Comprobación de R
A_reconstruida = np.matmul(Q, R)
print("A reconstruida:\n", A_reconstruida)
error = np.linalg.norm(A - A_reconstruida)
print("Error de reconstrucción de A:", error)


# Generamos las  matrices aleatorias y verificamos los  resultados
sizes = [10, 20]
for size in sizes:
    A = np.random.rand(size, size)
    Q, R = bib.QRdecomposition(A)

    # Comprobación de la ortogonalidad de Q
    aux = np.matmul(Q, Q.transpose()) - np.eye(Q.shape[0])
    norma = np.linalg.norm(aux)
    print(f"Norma de la matriz auxiliar para tamaño {size}x{size}: {norma}")

    # Comprobación de R
    A_reconstruida = np.matmul(Q, R)
    error = np.linalg.norm(A - A_reconstruida)
    print(f"Error de reconstrucción de A para tamaño {size}x{size}: {error}")
    
 
# Generamos la descomposicion de  matrices aleatorias y verificar resultados con qr propia de 
# Numpy, claro esta es  diferente al trabajado en clase 
sizes = [10, 20]
for size in sizes:
    # Generar una matriz aleatoria de tamaño 'size' x 'size'
    A = np.random.rand(size, size)
    # Realizar la descomposición QR utilizando np.linalg.qr
    Q, R = bib.QRdecomposition1(A)

    # Comprobación de la ortogonalidad de Q
    aux = np.matmul(Q, Q.T) - np.eye(Q.shape[0])
    norma = np.linalg.norm(aux)
    print(f"Norma de la matriz auxiliar para tamaño {size}x{size} (Q * Q^T - I): {norma}")

    # Comprobación de la matriz R
    # Reconstruir la matriz original A multiplicando Q y R
    A_reconstruida = np.matmul(Q, R)
    # Calcular el error de reconstrucción (diferencia entre A y A_reconstruida)
    error = np.linalg.norm(A - A_reconstruida)
    print(f"Error de reconstrucción de A para tamaño {size}x{size}: {error}")

