
import numpy as np
from numpy.linalg import norm, cond, solve

def hilbert_matrix(n):
    A = np.zeros((n,n))
    for i in range(1,n+1):
        for j in range(1,n+1):
            A[i-1,j-1]=1/(i+j-1)
    return A

print("{:15s}{:25s}{:20s}".format(" n","cond","error"))
print("-"*50)

solutions=[]

for i in range(4,17):
    x = np.ones(i)
    H = hilbert_matrix(i)
    b = H.dot(x)

    c = cond(H,2);
    xx = solve(H,b)
    err = norm(x-xx,np.inf)/norm(x,np.inf)
    solutions.append(xx)

    print("{:2d} {:20e} {:20e}".format(i,c,err))

#print("B=\n",B)
def escalonaSimple(A):
    nfil = A.shape[0]
    ncol = A.shape[1]
    
    for j in range(0,nfil):
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio)
            