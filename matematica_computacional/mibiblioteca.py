
import numpy as np
import numpy.linalg as la

def operacionFila(A,fm,fp,factor): # filafm = filafm - factor*filafm 
    A[fm,:] = A[fm,:]- factor*A[fp,:]

def intercambiaFil(A,fi,fj):
    A[[fi,fj],:] = A[[fj,fi],:]


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


def sustProgresiva(A,b):   #Resuelve un sistema escalonado
    N = b.shape[0] # A y b deben ser array numpy bidimensional
    x = np.zeros((N,1))
    for i in range(0,N):
        x[i,0] = (b[i,0]-np.dot(A[i,0:i],x[0:i,0]))/A[i,i]
    return x # Array bidimensional

 
def GaussElimSimple(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaSimple(Ab)
    A1 = Ab[: ,0:Ab.shape[1]-1].copy()
    b1 = Ab[: ,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x # array bidimensional
      
def escalonaConPiv(A):
    nfil = A.shape[0]
    ncol = A.shape[1]
    
    for j in range(0,nfil):
        imax = np.argmax(np.abs(A[j:nfil,j]))
        intercambiaFil(A,j+imax,j)
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio)
        print("Matriz escalonada pivoteada con pivot A:\n",A)
        
def GaussElimPiv(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaConPiv(Ab)
    print("Matriz Eliminacion pivoteada aumentada Ab:\n",Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    print("Matriz pivoteda A1:\n",A1)
    b1 = Ab[:,Ab.shape[1]-1].copy()
    print("vector b1:\n",b1)
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x # Array bidimensional      


def LUdescomp(A): # A debe ser matriz cuadrada
    L = np.zeros_like(A)
    U = A.copy()

    nfil = A.shape[0]
    ncol = A.shape[1]
    
    for j in range(0,nfil):
        for i in range(j+1,nfil):
            ratio = U[i,j]/U[j,j]
            L[i,j] = ratio
            operacionFila(U,i,j,ratio)

    np.fill_diagonal(L,1)
    return (L,U)

#def LDLtDescomp(A): # A debe ser matriz cuadrada

def projection(u,v): #projection numpy vectors u onto v
    aux = np.dot(u,v)/np.dot(v,v)
    return aux*v

def QRdecomposition(A): # Ortonormalize columns of numpy array
    Q = A.copy()
    R = np.zeros_like() 
    
    N = Q.shape[1]
    for col in range(N):
        sum = np.zeros_like(Q[:,col])
        for j in range(col):
            sum = sum + projection(Q[:,col],Q[:,j]) 
        Q[:,col] = Q[:,col] - sum

    for col in range(N):
        norm = la.norm(Q[:,col],2)
        Q[:,col] = Q[:,col]/norm 
    return Q,R 

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

    return (Q, R)

# Otra forma de calcular es utilizar 
# La función np.linalg.qr utiliza internamente el algoritmo de Householder 
def QRdecomposition1(A):
    Q, R = np.linalg.qr(A) 
    return Q, R
'''


def PLU_decomposicion(A):
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
def PLU_solucion(A, b):
    # Resuelve el sistema A*x = b usando descomposición PLU
    P, L, U = PLU_decomposicion(A)  # Obtenemos las matrices P, L y U
    Pb = np.dot(P, b)  # Multiplicamos la matriz de permutación P por el vector b
    y = sustProgresiva(L, Pb)  # Resolvemos L*y = P*b usando sustitución progresiva
    x = sustRegresiva(U, y)  # Resolvemos U*x = y usando sustitución regresiva
    return P, L, U, x  # Devolvemos P, L, U y el vector solución x
