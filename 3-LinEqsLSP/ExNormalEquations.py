# ExNormalEquations.py  -- version 2003-11-12
import numpy as np
from functions import trilinv

A=np.asarray([[1, 1], [1, 2], [1, 3], [1, 4]])
b=np.transpose(np.asarray([2, 1, 1, 1]))
[m,n] = np.shape(A)
C = A.T@A
c = (A.T@b).reshape(-1,1)
bb = (b.T@b).reshape(-1,1)
Cbar = np.vstack((np.hstack((C,c)), np.hstack((c.T, bb))))
Gbar = np.linalg.cholesky(Cbar)
G = np.copy(Gbar[:2,:2])
z = np.transpose(Gbar[2,:2])
rho = Gbar[2,2]
x = np.linalg.solve(G.T,z)
sigma2 = rho**2/(m-n)
T = trilinv(G) 
# Inversion of triangular matrix
S = T.T*T
Mcov = sigma2*S

