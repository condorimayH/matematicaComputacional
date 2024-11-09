import numpy as np
A = np.array([[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4., 24.0, -41.0]])
print(A)
n = A.shape[1] #number of columns in the matrix. shape[0] give row

for j in range(n):
# To orthogonalize the vector in column j with respect to the previous vectors, subtract from
#it its projection onto each of the previous vectors. 
    for k in range(j):
        A[:, j] = A[:, j] - np.dot(A[:, k], A[:, j]) * A[:, k]
        A[:, j] = A[:, j] / np.linalg.norm(A[:, j])

print(A)
#Qt = Q.transpose()



