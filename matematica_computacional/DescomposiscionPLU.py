import numpy as np

def operacionFila(A, fm, fp, factor):
    # filafm = filafm - factor * filafp
    # A[fm, :] hace referencia a la fila fm de la matriz A
    # factor * A[fp, :] es la fila fp multiplicada por el factor
    # Se resta factor * fila fp de la fila fm en la matriz A
    A[fm, :] = A[fm, :] - factor * A[fp, :]

def intercambiaFil(A, fi, fj):
    # Intercambia las filas fi y fj en la matriz A
    # A[[fi, fj], :] selecciona las filas fi y fj
    # A[[fj, fi], :] reasigna las filas intercambiadas
    A[[fi, fj], :] = A[[fj, fi], :]

def PLU_decomposition(A):
    n = A.shape[0]  # Número de filas de la matriz A
    P = np.eye(n)   # Matriz identidad de tamaño n (matriz de permutación)
    L = np.zeros((n, n))  # Matriz de ceros de tamaño n x n (matriz triangular inferior)
    U = A.copy()    # Hacemos una copia de A para no modificar la original (matriz triangular superior)

    for k in range(n):  # Iteramos sobre cada columna
        # Encontramos el pivote (el valor absoluto más grande en la subcolumna)
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if pivot != k:
            # Si el pivote no está en la fila actual, intercambiamos las filas
            intercambiaFil(P, k, pivot)
            intercambiaFil(U, k, pivot)
            if k >= 1:
                intercambiaFil(L, k, pivot)

        for j in range(k + 1, n):
            # Calculamos el multiplicador para la fila j
            L[j, k] = U[j, k] / U[k, k]
            # Realizamos la eliminación gaussiana
            operacionFila(U, j, k, L[j, k])

    # Establecemos 1 en la diagonal de L
    np.fill_diagonal(L, 1)
    return P, L, U  # Devolvemos las matrices P, L y U

def sustRegresiva(A, b):
    # Resuelve un sistema escalonado superior A*x = b
    N = b.shape[0]  # Número de filas de b
    x = np.zeros((N, 1))  # Vector solución inicializado a ceros
    for i in range(N-1, -1, -1):  # Iteramos de la última fila a la primera
        x[i, 0] = (b[i, 0] - np.dot(A[i, i+1:N], x[i+1:N, 0])) / A[i, i]
        # Calculamos x[i] usando sustitución regresiva
    return x  # Devolvemos el vector solución

def sustProgresiva(A, b):
    # Resuelve un sistema escalonado inferior A*x = b
    N = b.shape[0]  # Número de filas de b
    x = np.zeros((N, 1))  # Vector solución inicializado a ceros
    for i in range(0, N):  # Iteramos de la primera fila a la última
        x[i, 0] = (b[i, 0] - np.dot(A[i, 0:i], x[0:i, 0])) / A[i, i]
        # Calculamos x[i] usando sustitución progresiva
    return x  # Devolvemos el vector solución

def PLU_solucion(A, b):
    # Resuelve el sistema A*x = b usando descomposición PLU
    P, L, U = PLU_decomposition(A)  # Obtenemos las matrices P, L y U
    Pb = np.dot(P, b)  # Multiplicamos la matriz de permutación P por el vector b
    y = sustProgresiva(L, Pb)  # Resolvemos L*y = P*b usando sustitución progresiva
    x = sustRegresiva(U, y)  # Resolvemos U*x = y usando sustitución regresiva
    return P, L, U, x  # Devolvemos P, L, U y el vector solución x


def comprobar_PLU(A, P, L, U):
    # Comprueba si PA = LU
    PA = np.dot(P, A)  # Calculamos P*A
    LU = np.dot(L, U)  # Calculamos L*U
    print("Matriz PA:")
    print(PA)  # Mostramos PA
    print("Matriz LU:")
    print(LU)  # Mostramos LU
    return np.allclose(PA, LU)  # Verificamos si PA es aproximadamente igual a LU

def comprobar_solucion(A, x, b):
    # Comprueba si Ax = b
    Ax = np.dot(A, x)  # Calculamos A*x
    print("Ax calculado:")
    print(Ax)  # Mostramos A*x
    print("b original:")
    print(b)  # Mostramos b original
    return np.allclose(Ax, b)  # Verificamos si A*x es aproximadamente igual a b

# Ejemplo de uso
#A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], dtype=float)
#b = np.array([[5], [-2], [9]], dtype=float)
A = np.array([[0, -1, 4], [2, 1, 1], [1, 1, -2]], dtype=float)
b = np.array([[5], [-2], [9]], dtype=float)
# Descomposición y solución


P, L, U = PLU_decomposition(A)
print("Matriz P:") 
print(P) 
print("Matriz L:") 
print(L)
print("Matriz U:") 
print(U)
# Ejemplo de uso 

P, L, U, x = PLU_solver(A, b) 
 
print("Solución del sistema AX = b:", x)

# Verificación de la descomposición PLU
es_PLU_correcto = comprobar_PLU(A, P, L, U)
print("¿PA = LU? ", es_PLU_correcto)

# Verificación de la solución del sistema
es_solucion_correcta = comprobar_solucion(A, x, b)
print("¿AX = b? ", es_solucion_correcta)
