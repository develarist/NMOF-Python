import numpy as np
from scipy.stats import norm

# EuropeanOptionSimulation.py  -- version 2011-01-06

# parameters
S0  = 100
X   = 100
rf  = .05
sigma = .2
T   = .25
ns  = 100000 # number of Monte Carlo samples

# Black-Scholes price for European call
d1   = (np.log(S0/X)+(rf+sigma**2/2)*T) / (sigma * np.sqrt(T))
d2   = d1 - sigma*np.sqrt(T)
c0BS = S0*norm.cdf(d1) - X*np.exp(-rf*T) * norm.cdf(d2)

# MC Simulation
rs   = np.random.randn(ns,1)*sigma*np.sqrt(T) + (rf - sigma**2/2)*T
STs  = S0*np.exp(rs)
cTs  = np.fmax(STs-X,0)
c0MC = np.mean(cTs)*np.exp(-rf*T)

# "ingredients"
ex   = STs>X    # Boolean: exercise yes/no

print('Simulation Results: \n======================\n')
print('E(stock price): {0:.4f}'.format(np.mean(STs)), 
      ' (theor.: {0:.3f}'.format(S0*np.exp(rf*T)),')')
print('prob. exercise: {0:.4f}'.format(np.mean(ex)), 
      '   (theor.: {0:.3f}'.format(norm.cdf(d2)),')')
print('PV(E(paymt X)): {0:.4f}'.format(np.mean(ex)*X*np.exp(-rf*T)), 
      '  (theor.: {0:.3f}'.format(X*np.exp(-rf*T) * norm.cdf(d2)),')')
print('PV(E(paymt S)): {0:.4f}'.format(np.mean(ex)*np.mean(STs[ex])
      *np.exp(-rf*T)),'  (theor.: {0:.3f}'.format(S0*norm.cdf(d1)),')')
print('======================\ncall price:    {0:.4f}'.format(c0MC), 
      '   (theor.: {0:.3f}'.format(c0BS),')\n')


