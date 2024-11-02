import numpy as np
import mibiblioteca1 as bib
import time
import random
#B = np.array([[2.1,3.5,4.],[4.,5.,6.],[7.,8.,9.]])
#b = np.array([[9],[15],[24]])
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

A=2*np.random.rand(5,5)-1
b = np.array([[9],[15],[2],[5],[6]])
print("A=\n",A)
bib.escalonaSimple(A)
bib.escalonaConPiv(A)
print("A=\n",A)
bib.GaussElimPiv(A,b)
print("A=\n",A)