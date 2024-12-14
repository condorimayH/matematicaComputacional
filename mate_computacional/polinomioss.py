
import numpy as np
from numpy import polynomial as P
from matplotlib import pyplot as plt

'''

def interpola(x,y):
    l0 = P.Polynomial.fromroots([x[1],x[2],x[3]])
    L0 = l0/l0(x[0])
    l1 = P.Polynomial.fromroots([x[0],x[2],x[3]])
    L1 = l1/l1(x[1])
    l2 = P.Polynomial.fromroots([x[0],x[1],x[3]])
    L2= l2/l2(x[2])
    l3 = P.Polynomial.fromroots([x[0],x[1],x[2]])
    L2= l3/l3(x[3])
    p = y[0]*L0+y[1]*L1+y[2]*L3+y[2]*L3
    return p

xdata = [1,2,3,5]
ydata = [2, 1, 2,1]
pol = interpola(xdata,ydata)
print(pol(xdata))
'''


def interpola(x,y):
    n = len(x)
    xx= np.array(x)
    p = P.Polynomial([0])
    for i in range(n):
            mask = np.ones(n,dtype=bool)
            mask[i] = 0
            
            li = P.Polynomial.fromroots(xx[mask])
            Li = li/li(x[i])
            p  = p + y[i]*Li
    return p


xdata = np.array([1,2,3,5,7])
#ydata = [2, 1, 2,1,8]
ydata = np.sin(xdata)
pol = interpola(xdata,ydata)
print(pol(xdata))

# grafica
a = xdata.min()
b = ydata.max()
xx = np.linspace(a,b,200) 
yy = pol(xx)

yexacto = np.sin(xx)
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(xx,yexacto,'g',lw=2,label='solucion exacta')
ax.plot(xx,yy,'b',lw=2,label='polinomio interpolante')
ax.plot(xdata,ydata,'ro',alpha = 0.5,label='Datos')
ax.legend(loc=2)
ax.set_xlabel(r"$x$",fontsize=10)
ax.set_ylabel(r"$y$",fontsize=10)
plt.grid()
plt.show()

