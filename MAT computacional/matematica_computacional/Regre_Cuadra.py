import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import bibliotecaX as bib
import os
os.system("cls")

# PARAMETROS DEL MODELO VERDADERO
m = 50                          # Numero de datos  
x = np.linspace(-1, 1, m)       # intervalo sobre el cual efectuamos el experimento
a = -10
b = 2
c = 150
y_exact = a + b * x + c * x**2

# SIMULAR DATOS CON RUIDO
Ruido = 7
X = 1 - 2 * np.random.rand(m)
Y = a + b * X + c * X**2 + Ruido*np.random.randn(m)

# AJUSTE DE DATOS ( con QR y GaussElimSimple)
A = np.vstack([X**0, X**1, X**2]) 
AA = np.matmul(A,A.T)
bb = np.matmul(A,Y.T)
QR = bib.QRdecomp(AA)
Q = QR[0]
R = QR[1]
QQ = Q.T
Qb = np.matmul(QQ,bb)
Ri = np.linalg.inv(R)
sol=np.matmul(Ri,Qb)
#sol = bib.GaussElimSimple(R,Qb.T)

# COEFICIENTES a b c REDONDEADOS
nn=2                            # Num. decimales a redondear
a1 = round(sol[0],nn)           # Intercepto al eje Y
b1 = round(sol[1],nn)           # Coef. de X
c1 = round(sol[2],nn)           # Coef. de X²

y_fit = sol[0] + sol[1] * x + sol[2] * x**2
fig, ax = plt.subplots(figsize=(12, 4))

# COEFICIENTES R y R²
Sy = np.var(y_exact)                              # Varianza poblacional de y_exact
Syy = np.var(y_fit)                               # Varianza poblacional de y_fit
Cov_pob = np.cov(y_exact,y_fit, bias=True)[0,1]   # Covarianza poblacional
CR1 = Cov_pob/np.sqrt(Sy*Syy)                     # Coeficiente de correlacion R
CR2 = CR1**2                                      # Coeficiente de determinacion R²
nnn=5                                             # Num. decimales a redondear
R1 = round(CR1,nnn)
R2 = round(CR2,nnn)

# GRAFICAR
ax.plot(X, Y, 'g.', alpha=0.7, label='Datos simulados')
ax.plot(x, y_exact, 'r', lw=2, label='Ec. exacta: 'f'y = {a} + {b}x + {c}x²')
ax.plot(x, y_fit, 'b', lw=2, label='Ec. ajust.: 'f'y = {a1} + {b1}x + {c1}x²')
ax.set_xlabel(r"$eje X$", fontsize=18)
ax.set_ylabel(r"$eje Y$", fontsize=18)
plt.axhline(0, color='black',linewidth=1, ls='--')              #Eje X
plt.axvline(0, color='black',linewidth=1, ls='--')              #Eje Y
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)     #Grid
ax.legend(loc=2)

# Anadir más texto a la leyenda
Agrega = '\n'.join((
    r'n='f'{m}',
    r'ruido='f'{Ruido}',
    r'R='f'{R1}',
    r'R²='f'{R2}'))

# Posicionar el texto adicional
ax.text(1, 0.05, Agrega, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

plt.title('AJUSTE POR MINIMOS CUADRADOS EN MODELOS: Y = a + bX + cX²')
plt.show()
