# ExLSQR.py  -- version 1995-11-25
import numpy as np
from functions import trilinv

A=np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
b=np.array([2, 1, 1, 1]).reshape(-1,1)
[m,n] = np.shape(A)
[Q,R] = np.linalg.qr(A,mode='complete')
R1 = np.copy(R[:n,:n])
Q1 = np.copy(Q[:,:n])
Q2 = np.copy(Q[:,n:m])
x = np.linalg.solve(R1,Q1.T@b)
r = Q2.T@b
sigma2 = ((r.T@r)/(m-n))[0][0]
T = trilinv(np.copy(R1).T)
S = T.T@T
Mcov = sigma2*S
