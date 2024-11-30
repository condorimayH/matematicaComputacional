import numpy as np
import numpy.linalg as la

def projection(u, v):
    aux = np.dot(u, v) / np.dot(v, v)
    return aux * v

def QRdecomposition(A):
    Q = A.copy().astype(float)
    R = np.zeros((A.shape[1], A.shape[1]), dtype=float)

    N = Q.shape[1]
    for col in range(N):
        sum = np.zeros_like(Q[:, col]) # vector 
        for j in range(col):
            if la.norm(Q[:, j]) != 0: #no es cero
                sum = sum + projection(Q[:, col], Q[:, j])
                R[j, col] = np.dot(Q[:, j], Q[:, col])
        Q[:, col] = Q[:, col] - sum
        R[col, col] = la.norm(Q[:, col], 2)
        if R[col, col] != 0:
            Q[:, col] = Q[:, col] / R[col, col]

    return (Q, R)

def sustRegresiva(R, c):
    N = c.shape[0]
    x = np.zeros((N, 1))
    for i in range(N-1, -1, -1):
        x[i, 0] = (c[i, 0] - np.dot(R[i, i+1:], x[i+1:, 0])) / R[i, i]
    return x

# Generar una matriz aleatoria y un vector b aleatorio
#Snp.random.seed(0)
A = np.random.rand(4, 4)
b = np.random.rand(4, 1)
print("Matriz A aleatoria:\n", A)
print("Vector b aleatorio:\n", b)

# Aplicar descomposición QR
Q, R = QRdecomposition(A)
print("Q:\n", Q)
print("R:\n", R)

# Resolver Q^T * b
Qt_b = np.dot(Q.T, b)
print("Q^T * b:\n", Qt_b)

# Resolver el sistema RX = Q^T * b usando sustitución regresiva
x = sustRegresiva(R, Qt_b)
print("Solución del sistema AX = b:\n", x)

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
