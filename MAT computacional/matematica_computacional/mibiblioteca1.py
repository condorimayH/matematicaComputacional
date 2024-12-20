import numpy as np

def operacionFila(A, fm, fp, factor):
    A[fm, :] = A[fm, :] - factor * A[fp, :]

def escalonaSimple(A):
    nfil = A.shape[0]
    ncol = A.shape[1]
    for j in range(0, nfil):
        for i in range(j + 1, nfil):
            ratio = A[i, j] / A[j, j]
            operacionFila(A, i, j, ratio)

def sustRegresiva(A, b):
    N = b.shape[0]
    x = np.zeros((N, 1))
    for i in range(N - 1, -1, -1):
        x[i, 0] = (b[i, 0] - np.dot(A[i, i + 1:N], x[i + 1:N, 0])) / A[i, i]
    return x

def GaussElimSimple(A, b):
    Ab = np.append(A, b, axis=1)
    escalonaSimple(Ab)
    A1 = Ab[:, 0:Ab.shape[1] - 1].copy()
    b1 = Ab[:, Ab.shape[1] - 1].copy()
    b1 = b1.reshape(b.shape[0], 1)
    x = sustRegresiva(A1, b1)
    return x


