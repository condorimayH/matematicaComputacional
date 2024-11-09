import numpy as np
import mibiblioteca as bib
import time
import random
#A = np.array([[0.0,1.0,4],[4.0,5.0,6],[7.0,1.0,9.0]])
#b = np.array([[9.0],[15.0],[24.0]])

#x = bib.GaussElimSimple(B,b)
#print(x)
#res = np.matmul(B,x)-b
#print(res)

#A = np.random.rand(10,10)
#B= np.random.rand(-10, 10, 100)
#b = np.random.rand(10,1)
#print(b)
#initt = time.time()
#x = bib.GaussElimSimple(A,b)
#endt = time.time()
#ttransc = endt - initt
#print(ttransc)
''' 
A=2*np.random.rand(3,3)-1
b = np.array([[9],[15],[2]])
print("A=\n",A)
bib.escalonaSimple(A)
bib.escalonaConPiv(A)
print("A=\n",A)
bib.GaussElimPiv(A,b)
print("A=\n",A)

A = np.random.rand(3,3)
print("\nMatriz A:")
print(A)

b = np.random.rand(3,1)
print("\nVector de términos independientes b:")
print(b)
'''
'''
initt = time.time()
x1 = bib.GaussElimSimple(A,b)
endt = time.time()
ttransc = endt - initt
print(ttransc)

bib.escalonaSimple(A)
print("\n Metodo Escalonada simple de A:")
print(A)


bib.escalonaConPiv(A)
print("\n metododo Escalonada Con pivoteo de A:")
print(A)

x2=bib.GaussElimPiv(A,b)
print("\n Eliminacion con Gauss con pivote de A:")
print(A)

print("Solucion de Gauss simple:\n",x1)
res1=np.matmul(A,x1)-b
print("Residuo1:\n",res1)


print("Solucion por Gauss con pivote:\n",x2)
res2=np.matmul(A,x2)-b
print("Residuo2:\n",res2)
'''

'''
Ab = np.append(A, b, axis=1)
print("\nMatriz aumentada [A|b]:")
print(Ab)
# Resolver el sistema
A1 = Ab[:, :3]
b1 = Ab[:, 3]
b1 = b1.reshape(b1.shape[0], 1)
x = bib.sustRegresiva(A1, b1)

print("\nSolución del sistema:")
print(x)
'''

################################
''' 
A = np.random.rand(3,3)
print("\nMatriz A:")
print(A)

b = np.random.rand(3,1)
print("\nMatriz b:")
print(b)'''

'''
np.random.seed(0)  # Para reproducibilidad
A = np.random.uniform(1, 5, (10, 10))
x_exacta = np.ones((10, 1))  # Solución exacta x = [1, 1, ..., 1]
b = np.dot(A, x_exacta)
LU = bib.LUdescomp(A)
print("\nMatriz A:")
print(A)
print("\nMatriz b:")
print(b)
print("\nMatriz LU:")
print(LU)
L = LU[0]
U = LU[1]
print("\n Matriz L: ",L)
print("\n Matriz U:",U)
result = np.matmul(L,U)
print("\n result:")
print("\n Matriz U:",U)
diff = A-result
print (diff)


x3=bib.sustProgresiva(L,b)
print("\n sust. progresiva:")
print(x3)

x4=bib.sustRegresiva(U,x3)
print("\n sust. regresiva:")
print(x4)


res=np.matmul(A,x4)-b
print("\n res:",res)
resnorma = np.linalg.norm(res,1)
print("\n Resnorma:",resnorma)
# Calcular la norma suma del residuo
#residuo = b- np.dot(A, x4)
#norma_suma_residuo = np.sum(np.abs(residuo))-

'''
#A = np.random.rand(3,3)
#b = np.random.rand(3,1)
A = np.array([[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4., 24.0, -41.0]])

print(A)
#print(b)
Q = bib.QRdecomposition(A) 
print(Q)

aux = np.matmul(Q,Q.transpose())-np.eye(3,3)
print(aux)
norma = np.linalg.norm(aux)
print(norma)


