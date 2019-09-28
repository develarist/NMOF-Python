# exEmpCor.py -- version 2011-01-04
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

NO = 200 # empirical number of obs
Y1 = np.random.randn(NO,1)
Y2 = np.random.randn(NO,1)
Y2[Y2<0.3] = -0.5 # make Y2 non-Gaussian

# create CDFs
sortedY1 = np.sort(Y1,0)
sortedY2 = np.sort(Y2,0)

# resample
N = 1000 # (re)sample size
Z = np.random.randn(N,2); rho = 0.9;
M = np.asarray([[1, rho], [rho, 1]])
Z = Z@np.linalg.cholesky(M).T
U = norm.cdf(Z); U = np.ceil(NO*U).astype(int)-1;

#check
np.corrcoef(Y1,Y2,rowvar=False)
np.corrcoef(sortedY1[U[:,0]],
            sortedY2[U[:,1]],rowvar=False)

# histograms and scatter of original Y1 and Y2
plt.subplot(231), plt.hist(Y1)
plt.subplot(232), plt.hist(Y2)
plt.subplot(233), plt.scatter(Y1[U[:,0]],Y2[U[:,1]])

# histograms and scatter of resampled/correlated Y1 and Y2
plt.subplot(234), plt.hist(Y1[U[:,0]])
plt.subplot(235), plt.hist(Y2[U[:,1]])
plt.subplot(236), plt.scatter(sortedY1[U[:,0]],sortedY2[U[:,1]])
plt.show()