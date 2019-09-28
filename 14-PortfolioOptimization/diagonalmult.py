# diagonalmult.py -- version 2018-09-28
import numpy as np
import timeit

# set size of matrix
N = 500

# create matrices
D = np.diag(np.random.rand(N))
M = np.random.rand(N, N)

# compute product / compare time
start = timeit.default_timer()
Z1 = D @ M @ D
stop = timeit.default_timer()
print("Elapsed time is %s seconds." % (stop-start))

start = timeit.default_timer()
Z2 = np.multiply((np.diag(D).reshape(-1,1) @ np.diag(D).reshape(1,-1)), M)
stop = timeit.default_timer()
print("Elapsed time is %s seconds." % (stop-start))

# check difference betweeen matrices
print(np.max(np.max(np.abs(Z1 - Z2),1)))
