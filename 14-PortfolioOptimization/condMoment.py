# condMoment.py -- version 2010-12-31
import numpy as np
import timeit

rp = np.random.randn(10000)
trials = 1000

#%% squared
start = timeit.default_timer()
a = [rp @ rp for i in range(trials)][0]
stop = timeit.default_timer()
print('Elapsed time is {0:.6f}'.format(stop-start),'seconds.')

start = timeit.default_timer()
b = [np.sum(rp ** 2) for i in range(trials)][0]
stop = timeit.default_timer()
print('Elapsed time is {0:.6f}'.format(stop-start),'seconds.')

print('{0:.4f}'.format(np.max(np.abs(a-b))))

#%% other exponent
e = 4
start = timeit.default_timer()
for i in np.arange(trials):
    r = np.copy(rp)
    r = [r*rp for j in range(1,e)]
    a = np.sum(r)
stop = timeit.default_timer()
print('Elapsed time is {0:.6f}'.format(stop-start),'seconds.')

start = timeit.default_timer()
b = [np.sum(rp ** e) for i in range(trials)][0]
stop = timeit.default_timer()
print('Elapsed time is {0:.6f}'.format(stop-start),'seconds.')

print('{0:.4f}'.format(np.max(np.abs(a-b))))
