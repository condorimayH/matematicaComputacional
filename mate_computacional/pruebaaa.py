import numpy as np
import mibiblioteca as bib
import matplotlib.pyplot as plt
#Metodo de diferencias finiytas para PVF -u''(x)=-5,u(0)=1,u(1)=2
a=0;b=1 #[a,b] es el dominio
N=5 # nmumero de nodos interiores de la red
ua=1;ub=2#condicione sde frontera
xx = np.linspace(a,b,200)
yy = -5*xx*xx/2+7*xx/2+1  #solucion exacta
dx=(b-a)/(N+1)
nodos = np.linspace(a,b,N+2)
A=np.eye(N,k=1)-2*np.eye(N)+np.eye(N,k=1)
print(A)
b = -5*dx*dx*np.ones((N,1))
b[0,0]= b[0,0]-ua
b[N-1,0]= b[N-1,0]-ub
#print(b)
#U = bib.GaussElimPiv(A,b)
U = bib.GaussElimPiv(A,b)
U = np.append(U,np.array([[ub]]),axis=0)
U = np.append([[ua]],U,axis=0)
#print("puntos aproximados en los nodos U:")
#print(U)
fig, ax = plt.subplots(figsize =(10,8))
ax.plot( nodos,U, 'go', lw=2, label='Splucion aproximada')
ax.plot(xx,yy,lw=2,label='Solucion exacta')
plt.show()