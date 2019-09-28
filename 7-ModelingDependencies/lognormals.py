# lognormals.py -- version 2010-01-08
## uncorrelated Gaussian variates
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

p = 3; N = 200; X = np.random.randn(N,p)
X = X @ (np.diag(1/np.std(X,0)))
X = X - np.ones((N,1))*np.mean(X,0)
plt.figure(1)
scatter_matrix(pd.DataFrame(X), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(X,rowvar=False),4))

## induce linear correlation
rho = 0.5; M = np.ones((p,p)) * rho;
np.fill_diagonal(M, 1)
C = np.linalg.cholesky(M); Xc = X @ C;
plt.figure(2)
scatter_matrix(pd.DataFrame(Xc), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(Xc,rowvar=False),4))

## make exp
Z = np.exp(Xc)
plt.figure(3)
scatter_matrix(pd.DataFrame(Z), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(Z,rowvar=False),4))

## change variance of Xc
sd = 5
Xc[:,0] = sd*Xc[:,0]
Z = np.exp(Xc)
plt.figure(4)
scatter_matrix(pd.DataFrame(Z), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(Z,rowvar=False),4))