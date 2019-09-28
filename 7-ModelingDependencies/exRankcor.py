# exRankcor.py -- version 2010-01-09
## example rank correlation
import numpy as np
import scipy as sp
import pandas as pd
from pandas.plotting import scatter_matrix

p = 3; N = 1000; 
X = np.random.randn(N,p)
X = X @ (np.diag(1/np.std(X,0)))
X = X - np.ones((N,1))*np.mean(X,0)
scatter_matrix(pd.DataFrame(X), figsize=(6,6), diagonal='hist')
print(np.round(np.corrcoef(X,rowvar=False),4))

## induce rank correlation: Spearman
rhoS = 0.9   # correlation between any two assets

# set rank correlation matrix
Mrank = np.ones((p,p)) * rhoS
np.fill_diagonal(Mrank, 1)

# compute corresponding linear correlation matrix
M = 2*np.sin(np.pi/6*Mrank)

# compute cholesky factor
C = np.linalg.cholesky(M).T

# induce correlation, check
Xc = X @ C
scatter_matrix(pd.DataFrame(Xc), figsize=(6,6), diagonal='hist')

# check
np.corrcoef(Xc,rowvar=False)
print(sp.stats.spearmanr(Xc)[0])

sd = 5; Xc[:,0] = sd * Xc[:,0]
Z = np.exp(Xc)
# check
np.corrcoef(Z,rowvar=False)
print(sp.stats.spearmanr(Z)[0])
