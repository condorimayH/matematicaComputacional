
import numpy as np
import numpy.linalg as la
'''
def operacionFila(A,fm,fp,factor): # filafm = filafm - factor*filafm 
    A[fm,:] = A[fm,:]- factor*A[fp,:]


def escalonaSimple(A):
    nfil = A.shape[0]
    ncol = A.shape[1]

    for j in range(0,nfil):
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio)
            #   print(A)

def sustRegresiva(A,b): # resuelve sistema escalonada
    N = b.shape[0] # A y b deben ser array numpy bidiemmsional
    x = np.zeros((N,1))
    for i in range(N-1,-1,-1):
        x[i,0] = (b[i,0]-np.dot(A[i,i+1:N],x[i+1:N,0]))/A[i,i]
    return x #array bidimensional

def GaussElimSimple(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaSimple(Ab)
    A1 = Ab[: ,0:Ab.shape[1]-1].copy()
    b1 = Ab[: ,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x # array bidimensional
        
'''
'''
def projection(u, v):
    aux = np.dot(u, v) / np.dot(v, v)
    return aux * v

def QRdecomposition(A):
    Q = A.copy()
    R = np.zeros((Q.shape[1], Q.shape[1]))

    N = Q.shape[1]
    for col in range(N):
        for j in range(col):
            R[j, col] = np.dot(Q[:, j], Q[:, col])
            Q[:, col] = Q[:, col] - projection(Q[:, col], Q[:, j])
        R[col, col] = la.norm(Q[:, col], 2)
        Q[:, col] = Q[:, col] / R[col, col]

    return Q, R

# Ejemplo de uso
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Q, R = QRdecomposition(A)
print("Q:\n", Q)
print("R:\n", R)
'''

'''
import numpy as np
import numpy.linalg as la

def projection(u, v):
    aux = np.dot(u, v) / np.dot(v, v)
    return aux * v

def QRdecomposition(A):
    Q = np.zeros_like(A, dtype=float)
    R = np.zeros((A.shape[1], A.shape[1]), dtype=float)

    N = A.shape[1]
    for col in range(N):
        Q[:, col] = A[:, col]
        for j in range(col):
            R[j, col] = np.dot(Q[:, j], A[:, col])
            Q[:, col] = Q[:, col] - R[j, col] * Q[:, j]
        R[col, col] = la.norm(Q[:, col], 2)
        Q[:, col] = Q[:, col] / R[col, col]

    return Q, R

# Ejemplo de uso
A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
Q, R = QRdecomposition(A)
print("Q:\n", Q)
print("R:\n", R)

# Comprobación de la ortogonalidad de Q
aux = np.matmul(Q, Q.transpose()) - np.eye(Q.shape[0])
print("Auxiliar (Q * Q^T - I):\n", aux)
norma = np.linalg.norm(aux)
print("Norma de la matriz auxiliar:", norma)

# Comprobación de R
A_reconstructed = np.matmul(Q, R)
print("A reconstruida:\n", A_reconstructed)
error = np.linalg.norm(A - A_reconstructed)
print("Error de reconstrucción de A:", error)
'''


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

    return Q, R

# Otra forma de calcular es utilizar 
# La función np.linalg.qr utiliza internamente el algoritmo de Householder 
def QRdecomposition1(A):
    Q, R = np.linalg.qr(A) 
    return Q, R



