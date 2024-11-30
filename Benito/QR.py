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
        sum = np.zeros_like(Q[:, col])
        for j in range(col):
            if la.norm(Q[:, j]) != 0:
                sum = sum + projection(Q[:, col], Q[:, j])
                R[j, col] = np.dot(Q[:, j], Q[:, col])
        Q[:, col] = Q[:, col] - sum
        R[col, col] = la.norm(Q[:, col], 2)
        if R[col, col] != 0:
            Q[:, col] = Q[:, col] / R[col, col]

    return (Q, R)

# Generar una matriz aleatoria
np.random.seed(0)  # Fijar la semilla para reproducibilidad
A = np.random.rand(4, 4)  # Matriz aleatoria de 4x3
print("Matriz A aleatoria:\n", A)

# Aplicar descomposición QR
Q, R = QRdecomposition(A)
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
