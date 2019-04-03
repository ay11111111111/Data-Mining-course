import numpy as np
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

print(2*A)
print(A+B)
print(A*B)

print(A.shape)
A = A[:, np.newaxis]
B = B[np.newaxis, :]

print(A.shape)
print(B.shape)
print(A)
print(B)
print(np.dot(A, B))
print(np.dot(B, A))
