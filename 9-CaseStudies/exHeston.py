import numpy as np
from functions import callHestonMC
 
# exampleHeston.py -- version 2011-01-16
S     = 100       # spot price
q     = 0.02      # dividend yield
r     = 0.03      # risk-free rate 
X     = 110       # strike
tau   = 0.2       # time to maturity
k     = 1         # mean reversion speed (kappa in paper)
sigma = 0.6       # vol of vol
rho   = -0.7      # correlation
v0    = 0.2**2     # current variance
vT    = 0.2**2     # long-run variance (theta in paper)

# --solution by integration (taken from Chapter 15)
call = callHestoncf(S,X,tau,r,q,v0,vT,rho,k,sigma)

M = 200
N = 200000
[call,Q ]= callHestonMC(S,X,tau,r,q,v0,vT,rho,k,sigma,M,N)
SE = np.sqrt(Q/N)/np.sqrt(N)
print(np.round([-2*SE+call, call, 2*SE+call, 4*SE],4))
