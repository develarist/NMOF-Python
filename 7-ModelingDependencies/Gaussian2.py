# Gaussian2.py  -- version 2011-01-16
import numpy as np
import scipy as sp
import pandas as pd
from pandas.plotting import scatter_matrix

p = 5     # number of assets
N = 500   # number of obs
X = np.random.randn(N,p)
X = X @ (np.diag(1/np.std(X,0)))
X = X - np.ones((N,1))*np.mean(X,0)
# check
scatter_matrix(pd.DataFrame(X), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(X,rowvar=False),4))

## induce linear correlation
rho = 0.7   # correlation between any two assets

# set correlation matrix
M = np.ones((p,p)) * rho
np.fill_diagonal(M, 1)

# compute cholesky factor
C = np.linalg.cholesky(M).T

# induce correlation, check
Xc = X @ C
scatter_matrix(pd.DataFrame(Xc), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(Xc,rowvar=False),4))

## rank-deficient case
np.linalg.matrix_rank(Xc)
Xc[:,p-1] = Xc[:,:(p-1)]@np.random.rand(p-1,1).flatten()
np.linalg.matrix_rank(Xc)
# check
scatter_matrix(pd.DataFrame(Xc), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(Xc,rowvar=False),4))
M = np.corrcoef(Xc,rowvar=False)
np.linalg.matrix_rank(M)
np.linalg.cholesky(M).T

## eigen decomposition
[D,V] = np.linalg.eig(M)
D = np.diag(D)
C = np.real(V@np.sqrt(D))
C = C.T
Xcc = X @ C
scatter_matrix(pd.DataFrame(Xcc), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(Xcc,rowvar=False),4))

## eigen v. svd
# eigen decomposition
M = np.corrcoef(X,rowvar=False)
[D,V1] = np.linalg.eig(M)
D = np.diag(D)
C = np.real(V1@np.sqrt(D))
C = C.T

# svd
[U,S,V2] = np.linalg.svd(X)

# ratio of sing values squared to eigenvalues
print(((S**2)/(N-1)) / np.sort(np.diag(D))[::-1])