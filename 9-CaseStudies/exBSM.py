# exBMS.py -- version 2010-12-08
import timeit
import numpy as np
from functions import callBSM
from functions import callBSMMC
from functions import callBSMMC2
from functions import callBSMMC3
from functions import callBSMMC4
from functions import callBSMMC5
from functions import pricepaths

S     = 100
X     = 100
tau   = 1/2
r     = 0.03
q     = 0.05
v     = 0.2**2

# MC parameters
M = 1
N = 100000

#%% analytic solution
start = timeit.default_timer()
call = callBSM(S,X,tau,r,q,v)
stop = timeit.default_timer()
time = np.round(stop-start,2)
print('The analytic solution.\ncall price {0:.2f}'.format(call),
      '.   It took {0:.1f}'.format(time),'seconds.\n')


#%% MC
start = timeit.default_timer()
[call, payoff] = callBSMMC(S,X,tau,r,q,v,M,N)
stop = timeit.default_timer()
time = np.round(stop-start,2)
SE = np.std(payoff)/np.sqrt(N)
print('\nMC 1 (vectorized).\ncall price', np.round(call,2),'.   lower band ', 
      np.round(-2*SE+call,2),'   upper band ', np.round(2*SE+call,2),
      '.   width of CI ', np.round(4*SE,2),'.   It took', time,'seconds.\n')


#%% pathwise
start = timeit.default_timer()
[call, Q] = callBSMMC2(S,X,tau,r,q,v,M,N)
stop = timeit.default_timer()
time = np.round(stop-start,2)
SE = np.sqrt(Q/N)/np.sqrt(N)
print('\nMC 2 (loop).\ncall price', np.round(call,2),'.   lower band ', 
      np.round(-2*SE+call,2),'.   upper band ', np.round(2*SE+call,2),
      '.   width of CI ', np.round(4*SE,2),'.   It took', time,'seconds.\n')


#%% variance reduction: antithetic
start = timeit.default_timer()
[call, Q] = callBSMMC3(S,X,tau,r,q,v,M,N)
stop = timeit.default_timer()
time = np.round(stop-start,2)
SE = np.sqrt(Q/N)/np.sqrt(N)
print('\nMC 3 (loop), antithetic variates.\ncall price', np.round(call,2),
      '.   lower band ', np.round(-2*SE+call,2),'.   upper band ', 
      np.round(2*SE+call,2),'.   width of CI ', np.round(4*SE,2),
      '.   It took', time,'seconds.\n')


#%% variance reduction: antithetic
start = timeit.default_timer()
[call, payoff] = callBSMMC4(S,X,tau,r,q,v,M,N)
stop = timeit.default_timer()
time = np.round(stop-start,2)
SE = np.std(payoff)/np.sqrt(N)
print('\nMC 4 (vectorized), antithetic variates.\ncall price', 
      np.round(call,2),'.   lower band ', np.round(-2*SE+call,2),
      '.   upper band ', np.round(2*SE+call,2),'.   width of CI ', 
      np.round(4*SE,2),'.   It took', time,'seconds.\n')


#%% variance reduction: control variate
start = timeit.default_timer()
[call, Q] = callBSMMC5(S,X,tau,r,q,v,M,N)
stop = timeit.default_timer()
time = np.round(stop-start,2)
SE = np.sqrt(Q/N)/np.sqrt(N)
print('\nMC 5 (loop), control variate.\ncall price', np.round(call,2),
      '.   lower band ', np.round(-2*SE+call,2),'.   upper band ', 
      np.round(2*SE+call,2),'.   width of CI ', np.round(4*SE,2),
      '.   It took', time,'seconds.\n')