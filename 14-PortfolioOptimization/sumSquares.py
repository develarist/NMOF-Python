# sumSquares.py -- version 2018-09-28
import numpy as np
import timeit

trials = 1000
n = 50000
ee = np.random.randn(n,1)

start = timeit.default_timer()
z1 = np.sum(ee ** 2) # [np.sum(ee ** 2) for i in range(trials)]
stop = timeit.default_timer()
print("Elapsed time is %s seconds." % (stop-start))

start = timeit.default_timer()
z2 = (ee.T @ ee)[0][0]
stop = timeit.default_timer()
print("Elapsed time is %s seconds." % (stop-start))

start = timeit.default_timer()
z3 = np.sum(ee.conj()*ee, axis=0)[0]
stop = timeit.default_timer()
print("Elapsed time is %s seconds." % (stop-start))
