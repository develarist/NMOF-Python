# Time-stamp: <2018-02-05>
import numpy as np
from numpy.random import randn
import scipy.linalg as linalg
import timeit

m = 10000; n = 10; trials = 100;  
X = randn(m,n); y = randn(m,1);

# QR
start = timeit.default_timer()
for r in range(trials):
    sol1 = linalg.lstsq(X,y)[0]
stop = timeit.default_timer()
print('Elapsed time is {0:.6f}'.format(stop-start),'seconds.')

# form (X'X) and (X'y)
start = timeit.default_timer()
for r in range(trials):
    sol2 = linalg.solve((X.T@X),(X.T@y))
stop = timeit.default_timer()
print('Elapsed time is {0:.6f}'.format(stop-start),'seconds.')

# check
print(np.max(np.abs(sol1[:]-sol2[:])))