# ExLSsvd.py  -- version 1995-11-27
import numpy as np

A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
b = np.array([2, 1, 1, 1]).T
[m,n] = np.shape(A)
[U,sv,V] = np.linalg.svd(A)
c = U.T@b
c1 = np.copy(c[range(n)])
c2 = np.copy(c[range(n,m)])
z1 = c1/sv
x = V@z1
sigma2 = c2.T@c2/(m-n)
S = V@np.diag(sv**(-2))@V.T
Mcov = sigma2*S
