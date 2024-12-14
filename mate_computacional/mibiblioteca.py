
from turtle import shape
import numpy as np
import numpy.polynomial as P
import scipy.linalg as la

def operacionFila(A,fm,fp,factor): # filafm = filafm - factor*filafp
    A[fm,:] = A[fm,:] - factor*A[fp,:]

def intercambiaFil(A,fi,fj):
    A[[fi,fj],:] = A[[fj,fi],:]
#def operacionFil(A,fm,fp,factor): # filafm = filafm - factor*filafm 
 #   A[fm,:] = A[fm,:]- factor*A[fp,:]


def escalonaSimple(A):
    nfil = A.shape[0]
    ncol = A.shape[1]

    for j in range(0,nfil):
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio)
            print(A)

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
    print("Matriz aumentada:\n",Ab)
    escalonaSimple(Ab)
    print("Matriz aumentada escalonada:\n",Ab)
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
        print("Matriz escalonada pivoteada con pivot :\n",A)
        
def GaussElimPiv(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaConPiv(Ab)
    print("Matriz Eliminacion pivoteada aumentada :\n",Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    print("Matriz pivoteda:\n",A1)
    b1 = Ab[:,Ab.shape[1]-1].copy()
    print("vector:\n",b1)
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
    R = np.zeros_like(Q,dtype=float) 
    N = Q.shape[1]
    for col in range(N):
        sum = np.zeros_like(Q[:,col])
        for j in range(col):
            sum = sum + projection(Q[:,col],Q[:,j]) 
        Q[:,col] = Q[:,col] - sum

    for col in range(N):
        norm = la.norm(Q[:,col],2)
        Q[:,col] = Q[:,col]/norm 

    for col in range(N):
        for row in range(col+1):
            R[row,col] = np.dot(Q[:,row],A[:,col])
    return Q,R



