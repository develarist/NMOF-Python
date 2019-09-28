# returns.py -- version 2018-09-28
#%% generate artificial price data:
# R = returns, P = prices
import numpy as np
import matplotlib.pyplot as plt

ns = 100 # number of scenarios
na = 10  # number of assets
R = 1 + np.random.randn(ns, na) * 0.01
P = np.cumprod( np.vstack((100*np.ones((1,na)),R)),0 )
plt.plot(P), plt.show()

#%% discrete returns
# compute returns: rets should be equal to R
rets1 = P[1:,:] / P[:len(P)-1,:]
# ... or
rets2 = np.diff(np.log(P),1,axis=0) / P[:len(P)-1,:] + 1
np.max(np.max(np.abs(rets1-R),0)) # not 'exactly' equal
np.max(np.max(np.abs(rets2-R),0)) # not 'exactly' equal
np.max(np.max(np.abs(rets1-rets1),0)) # 'exactly' equal

#%% log-returns
rets3 = np.diff(np.log(P),1,axis=0)
# ... almost like discrete returns
plt.plot(rets1 - rets3 - 1), plt.show()
