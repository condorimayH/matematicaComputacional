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

''' 

'''
# Definir parámetros del modelo real
x = np.linspace(-1, 1, 100)  # Intervalo sobre el cual efectuamos el experimento
a, b, c = 1, 2, 150
y_exact = a + b * x + c * x**2

# Simular datos con ruido
m = 20
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 + 2 * np.random.randn(m)

# Construcción explícita de la matriz A usando arrays de numpy
A = np.array([np.ones(m), X, X**2]).T
print("Matriz A:\n", A)

# Calcular la matriz A^T A y el vector A^T Y
AtA = np.dot(A.T, A)
AtY = np.dot(A.T, Y)

print("Matriz A^T A:\n", AtA)
print("Vector A^T Y:\n", AtY)

# Resolver el sistema usando np.linalg.solve
#sol = np.linalg.solve(AtA, AtY)
#print("Solución (coeficientes del polinomio):", sol)


# Resolver el sistema usando eliminación Gaussiana con pivoteo parcial
b_columna = AtY.reshape(-1, 1)  # Asegurarse de que b sea una columna
sol = bib.GaussElimPiv(AtA, b_columna)

print("Solución (coeficientes del polinomio):", sol.flatten())

# Evaluar el polinomio ajustado
y_fit = sol[0] + sol[1] * x + sol[2] * x**2

# Graficar los resultados
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(X, Y, 'go', alpha=0.5, label='Datos simulados')
ax.plot(x, y_exact, 'r', lw=2, label='Valor real $y = 1 + 2x + 150x^2$')
ax.plot(x, y_fit, 'b', lw=2, label=f'Ajuste de mínimos cuadrados $y = {sol[0][0]:.2f} + {sol[1][0]:.2f}x + {sol[2][0]:.2f}x^2$')
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.legend(loc=2)
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt
import mibiblioteca as bib

# Paso 1: Definir parámetros del modelo real
# Generamos 100 puntos igualmente espaciados en el intervalo [-1, 1]
x = np.linspace(-1, 1, 100)  

# Coeficientes del polinomio real
a, b, c = 1, 2, 150  

# Evaluamos el polinomio exacto usando los coeficientes
y_real = a + b * x + c * x**2

# Paso 2: Simular datos con ruido
# Número de datos simulados
m = 20  
# Generamos datos X aleatorios en el intervalo [-1, 1]
X = 1 - 2 * np.random.rand(m)  

# Calculamos Y usando el polinomio real y añadiendo ruido aleatorio
Y = a + b * X + c * X**2 + 2 * np.random.randn(m)
# Paso 3: Construcción explícita de la matriz A usando arrays de numpy
# Matriz A con términos constantes, lineales y cuadráticos
A = np.array([np.ones(m), X, X**2]).T  
print("Matriz A:\n", A)

# Paso 4: Calcular la matriz A^T A y el vector A^T Y
AtA = np.dot(A.T, A)
AtY = np.dot(A.T, Y)
print("Matriz A^T A:\n", AtA)
print("Vector A^T Y:\n", AtY)

# Paso 5: Resolver el sistema usando eliminación Gaussiana con pivoteo parcial

# Asegurarse de que AtY sea una columna
b_columna = AtY.reshape(-1, 1)  

# Utilizamos una función personalizada GaussElimPiv para resolver el sistema
sol = bib.GaussElimPiv(AtA, b_columna)
print("Solución (coeficientes del polinomio):", sol.flatten())

# Paso 6: Evaluar el polinomio ajustado
# Calculamos los valores ajustados del polinomio
y_ajustado = sol[0] + sol[1] * x + sol[2] * x**2
# Paso 7: Graficar los resultados
fig, ax = plt.subplots(figsize=(12, 4))
# Graficar los datos simulados con ruido (puntos verdes)
ax.plot(X, Y, 'go', alpha=0.5, label='Datos simulados')
# Graficar el polinomio real (línea roja)
ax.plot(x, y_real, 'r', lw=2, label='Valor real $y = 1 + 2x + 150x^2$')
# Graficar el polinomio ajustado por mínimos cuadrados (línea azul)
ax.plot(x, y_ajustado, 'b', lw=2,
label=f'Ajuste de mínimos cuadrados $y = {sol[0][0]:.2f} + {sol[1][0]:.2f}x + {sol[2][0]:.2f}x^2$')
# Etiquetas de los ejes
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
# Mostrar la leyenda
ax.legend(loc=2)
# Mostrar el gráfico
plt.title('AJUSTE POR MINIMOS CUADRADOS EN MODELOS: Y = a + bX + cX²')
plt.show()
