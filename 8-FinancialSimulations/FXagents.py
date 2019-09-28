# FXagents.py  -- version 2011-01-06
import numpy as np

# -- parameters
N_agents = 50   # number of agents
N_days   = 500  # number of days
N_IPD    = 20   # number of interactions per day

P_fund   = 100  # fundamental price
sigma_p  = 0.1  # additional price volatility

g        = 1    # adj. speed chartists; 0 < g <=1
nu       = .01  # adj. speed fundamentalisits;   0 < nu <= 1
delta    = .25  # probability convincing
epsilon  = .01  # random type change    

# -- initial setting
N_I    = N_days * N_IPD  # total number of interactions
isFund = np.random.rand(N_agents,1)<.5 # type of investor
P      = float('NaN')*np.ones((N_I,1))      # intra period prices; initial values
P[:2] = P_fund + np.random.randn(2,1)*sigma_p
w = float('NaN')*np.ones((N_I,1))           # perceived fraction of fundamentalists

E_change_F = np.zeros(N_I)
E_change_C = np.zeros(N_I)

# -- emergence over time    
for i in np.arange(2,N_I):
    a = np.random.permutation(range(N_agents))
    if np.random.rand(1)[0] < delta:  # recruitment
        isFund[a[1]] = isFund[a[0]]

    if np.random.rand(1)[0] < epsilon: # individual change of opinion
        isFund[a[2]] = ~isFund[a[2]]

    w[i] = np.mean(isFund)  # perceived fraction of fundamentalists

    # expected price changes and new price
    E_change_F[i] = (P_fund - P[i-1]) * nu
    E_change_C[i] = (P[i-1]-P[i-2]) * g
    change = w[i] * E_change_F[i]  + (1-w[i]) * E_change_C[i]
    P[i] = np.abs((P[i-1] + change) +  np.random.randn(1)*sigma_p)

# -- extract end of period prices 
t = np.arange(N_IPD,N_I,N_IPD)
S = P[t]
