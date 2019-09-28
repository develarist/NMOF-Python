import numpy as np


#%%
"NMOF chapter 3 functions"
import matplotlib.pyplot as plt
from matplotlib import rc

#------------------------------------------------------------------------------
def converged(x0,x1,tol):
    # converged.py  -- version 2007-08-10
    res = np.all( abs(x1-x0) / (abs(x0) + 1) < tol )
    return res

#------------------------------------------------------------------------------
def OmegaGridSearch(A,winf=0.9,wsup=1.8,npoints=21):
    # OmegaGridSearch.py  -- version  2010-10-28
    # Grid search for optimal relaxation parameter
    D = np.diag(np.diag(A))
    L = np.tril(A,-1)
    U = np.triu(A,1)
    omegavec = np.linspace(winf,wsup,npoints)
    s = []
    for k in range(npoints):
        w = omegavec[k]
        M = D + w*L
        N = (1-w)*D - w*U
        v = np.linalg.eig(np.linalg.inv(M)*N)[0]
        s.append(np.max(np.abs(v)))
    smin = np.min(s)
    i = np.where(s==smin)[0][0]
    w = omegavec[i]
    plt.plot(omegavec,s,'o',markersize=5,color='grey')
    plt.xticks((winf, 1, w, wsup), (winf, 1, w, wsup))
    plt.ylabel(r'$\rho$', rotation=0)
    plt.xlabel(r'$\omega$')
    plt.show()
    r = smin
    return w,r

def SOR(x1,A,b,omega=1.2,tol=1e-4,maxit=70):
    # SOR.py  -- version 2010-10-25
    # SOR for Ax = b 
    it = 0
    n = len(x1)
    x0 = -x1
    while not(converged(x0,x1,tol)):
       x0 = x1
       for i in range(n):
          x1[i,0] = omega*( b[i,0]-(A[i,:]@x1)[0] ) / A[i,i] + x1[i,0]
       it=it+1
       if it>maxit:
           raise ValueError('Maxit in SOR')
    nit = it
    return x1,nit

#------------------------------------------------------------------------------
def lu3diag(p,d,q):
    # lu3diag.py  -- version 2018-11-30
    n = len(d); l = np.zeros(n-1); u = np.copy(l);
    u = np.insert(u,0,d[0])
    for i in np.arange(1,n):
        l[i-1] = p[i-1]/u[i-1]
        u[i]   = d[i] - l[i-1]*q[i-1]
    return l,u

#------------------------------------------------------------------------------
def solve3diag(l,u,q,b):
    # solve3diag.py  -- version 1999-05-11
    # Back and forward substitution for tridiagonal system
    n = len(b)
    for i in range(1,n):
        b[i] = np.copy(b[i] - l[i-1] * b[i-1])
    b[n-1] = b[n-1] / u[n-1]
    for i in np.arange(n-2,-1,-1):
        b[i] = ( b[i] - q[i] * b[i+1] ) / u[i]
    return b

#------------------------------------------------------------------------------
def trilinv(L):
    # trilinv.py  -- version 1995-03-25
    for i in range(len(L[:,0])):
        for j in range(i-1):
            L[i,j] = -L[i,np.arange(j,i-1)]*L[np.arange(j,i-1),j] / L[i,i]
        L[i,i] = 1/L[i,i]
    return L

#%%
"NMOF chapter 5 functions"
#------------------------------------------------------------------------------
def AmericanCallDiv(S0,X,r,T,sigma,D,TD,M):
    # AmericanCallDiv.py -- version 2010-12-28
    # compute constants
    f7 = 1
    dt = T/M
    v = np.exp(-r * dt)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1 /u
    p = (np.exp(r * dt) - d)/(u - d)
    
    # adjust spot for dividend
    S0 = S0 - D * np.exp(-r * TD)
    
    # initialize asset prices at maturity (period M)
    S = np.zeros((M + 1,1))
    S[f7 -1] = S0 * d**M
    for j in np.arange(M):
        S[f7+j] = S[f7+j - 1] * u / d
    
    # initialize option values at maturity (period M)
    C = np.fmax(S - X, 0)
    
    # step back through the tree
    for i in np.arange(M-1,-1,-1):
        # compute present value of dividend (dD)
        t = T * i / M
        dD = D * np.exp(-r * (TD-t))
        for j in np.arange(i+1):
            C[f7+j -1] = v * ( p * C[f7+j] + (1-p) * C[f7+j -1])
            S[f7+j -1] = S[f7+j -1] / d
            if t > TD:
                C[f7+j -1] = np.fmax(C[f7 + j -1], S[f7+j -1] - X)
            else:
                C[f7+j -1] = np.fmax(C[f7 + j -1], S[f7+j -1] + dD - X)
    C0 = (C[f7 -1])[0]
    return C0

#------------------------------------------------------------------------------
def AmericanPut(S0,X,r,T,sigma,M):
    # AmericanPut.py -- version 2010-12-28
    f7 = 1; dt = T / M; v = np.exp(-r * dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # initialize asset prices at maturity (period M)
    S = np.zeros((M + 1,1))
    S[f7-1] = S0 * d**M
    for j in np.arange(1,M+1):
        S[f7+j-1] = S[f7+j - 2] * u / d
    
    # initialize option values at maturity (period M)
    P = np.fmax(X - S, 0)
    
    # step back through the tree
    for i in np.arange(M-1,-1,-1):
        for j in np.arange(i+1):
            P[f7+j-1] = v * (p * P[f7+j] + (1-p) * P[f7+j-1])
            S[f7+j-1] = S[f7+j-1] / d
            P[f7+j-1] = np.fmax(P[f7 + j-1],X - S[f7+j-1])
    P0 = P[f7-1][0]
    return P0

#------------------------------------------------------------------------------
def EuropeanCall(S0,X,r,T,sigma,M):
# EuropeanCall.py -- version 2010-12-28
# compute constants
    f7 = 1
    dt = T / M
    v = np.exp(-r * dt)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1 /u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # initialize asset prices at maturity (period M)
    S = np.zeros((M + 1,1))
    S[f7-1] = S0 * d**M
    for j in range(M):
        S[f7+j] = S[f7+j - 1] * u / d
    
    # initialize option values at maturity (period M)
    C = np.fmax(S - X, 0)
    
    # step back through the tree
    for i in np.arange(M-1,-1,-1):
        for j in np.arange(i+1):
            C[f7+j-1] = v * (p * C[f7+j] + (1-p) * C[f7+j-1])
    C0 = C[f7-1]
    return C0

#------------------------------------------------------------------------------
def EuropeanCallBE(S0,X,r,T,sigma,M):
# EuropeanCallBE.py  -- version: 2011-03-14
# compute constants
    dt = T / M
    u = np.exp(sigma*np.sqrt(dt))
    d = 1 /u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # initialise asset prices at maturity (period M)
    C = np.fmax(S0*d**(np.arange(M,-1,-1).reshape(-1,1))*(
            u**(np.arange(0,M+1).reshape(-1,1))) - X,0)
    
    # log/cumsum version
    csl = np.cumsum(np.log(np.insert(np.arange(1,M+1),0,1))).reshape(-1,1)
    tmp = csl[M] - csl - csl[np.arange(M,-1,-1)] + np.log(p)*(
            np.arange(0,M+1).reshape(-1,1)) + np.log(1-p)*(
            np.arange(M,-1,-1).reshape(-1,1))
    C0 = np.exp(-r*T)*np.sum(np.multiply(np.exp(tmp), C))
    return C0

#------------------------------------------------------------------------------
def EuropeanCallGreeks(S0,X,r,T,sigma,M):
    # EuropeanCallGreeks.py -- version 2010-12-28
    # compute constants
    f7 = 1
    dt = T/M
    v = np.exp(-r * dt)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1 /u
    p = (np.exp(r * dt) - d)/(u - d)
    
    # initialize asset prices at maturity (period M)
    S = np.zeros((M + 1,1))
    S[f7-1] = S0 * d**M
    for j in np.arange(M):
        S[f7+j] = S[f7+j - 1] * u / d
    
    # initialize option values at maturity (period M)
    C = np.fmax(S - X, 0)
    
    # step back through the tree
    for i in np.arange(M-1,-1,-1):
        for j in np.arange(i+1):
            C[f7+j -1] = v * ( p * C[f7+j + 1 -1] + (1-p) * C[f7+j -1])
            S[f7+j -1] = S[f7+j -1] / d
        if i==2:
          #gamma
          gammaE = ((C[2+f7 -1] - C[1+f7 -1]) / (S[2+f7 -1] - S[1+f7 -1]) - (
                  C[1+f7 -1] - C[0+f7 -1]) / (S[1+f7 -1] - S[0+f7 -1]))/ (
                    0.5 * (S[2+f7 -1] - S[0+f7 -1]))
          #theta (aux)
          thetaE = C[1+f7 -1]
        if i==1:
            #delta
            deltaE = (C[1+f7 -1] - C[0+f7 -1]) / (S[1+f7 -1] - S[0+f7 -1])
        if i==0:
            #theta (final)
            thetaE = (thetaE - C[0+f7 -1]) / (2 * dt)
    C0 = C[f7 -1]
    return C0,deltaE,gammaE,thetaE

#%%
"NMOF chapter 6 functions"
from scipy.stats import skew,kurtosis,norm

#------------------------------------------------------------------------------
def bootstrap(r,n,b=1):
    # bootstrap.py  -- version 2011-01-06
    #   r ... original data
    #   n ... length of bootstrap sample x
    #   b ... block length

    [T, k] = np.shape(r)
    if b==1:  # simple bootstrap 
        j = np.ceil(np.random.rand(n,1)*T -1).astype(int).flatten()
        x = np.copy(r[j,:])
    else:   # circular block bootstrap
        nb = np.ceil(n/b).astype(int) # number of bootstraps
        js = np.floor(np.random.rand(nb,1)*T).astype(int) # starting points - 1
        x = float('NaN')*np.zeros((nb*b,k))
        for i in range(nb):
            j = (np.mod(js[i]+np.arange(0,b), T)).tolist()  # positions in original data
            s = np.arange(0,b) + (i -1)*b
            x[s,:] = np.copy(r[j,:])
        if (nb*n) > n:               # correct length if nb*b > n 
            np.delete(x,np.s_[n +1::],0)
    return x

#------------------------------------------------------------------------------
def CornishFisher(r,alpha):
    # CornishFisher.py  -- version 2011-01-06

    S = skew(r)
    K = kurtosis(r)
    u = norm.ppf(alpha)
    Omega = u + S/6*(u**2 -1) + (K -3)/24 *(
            u**3 - 3*u) - S**2/36 * (2*u**3 -5*u)
    q = np.mean(r,0) + Omega * np.std(r,0)
    return q

#------------------------------------------------------------------------------
def CornishFisherSimulation(mu,sigma,skew,kurt,N):
    # CornishFisherSimulation.py  version 2011-01-06
    alpha = np.random.rand(N,1)
    u = norm.ppf(alpha)
    Omega = (u + skew/6*(u**2 -1) + (kurt-3)/24 *(u**3 - 3*u) 
              - skew**2/36 * (2*u**3 -5*u))
    X = mu + sigma * Omega
    return X

#------------------------------------------------------------------------------
def GaussianBoxMullerSim(N_samples=1):
    # GaussianBoxMullerSim.py  -- version 2011-01-06

    u = np.random.rand(N_samples,2)
    z1 = np.multiply( np.sqrt(-2*np.log(u[:,0])), 
                     np.cos(2 * np.pi * u[:,1]) )[0]
    z2 = np.multiply( np.sqrt(-2*np.log(u[:,0])), 
                     np.sin(2 * np.pi * u[:,1]) )[0]
    return z1,z2

#------------------------------------------------------------------------------
def GaussianPolarSim(N_samples=1):
    # GaussianPolarSim.py  -- version 2011-01-06

    a = np.random.rand(N_samples,2) * 2 - 1
    s = np.copy(a[:,0]**2 + a[:,1]**2).reshape(-1,1)
    
    # --check for "at-or-within-unit-circle" criterion
    outside = (s > 1).flatten()
    while np.sum(outside) > 0:
        a[outside] = np.random.rand(np.sum(outside),2) * 2 - 1
        s[outside] = (a[outside,0]**2 + a[outside,1]**2).reshape(-1,1)
        outside = s > 1
    
    # --perform transformation
    f  = np.sqrt(-2*np.log(s)/s)
    z1 = f * a[:,0].reshape(-1,1)
    z2 = f * a[:,1].reshape(-1,1)
    return z1,z2

#------------------------------------------------------------------------------
def KDE(x,xi,Kernel='Gaussian'):
    # KDE.py  -- version 2011-01-06
    #   kernel density estimation
    #   x ... points at which to estimate the density
    #   xi .. original sample
    #   h ... bandwidth, smoothing parameter
    #   Kernel .. name of kernel

    h=(4/(3*len(xi)))**(.2) *np.std(xi)
    
    case = Kernel.upper()
    if case == 'GAUSSIAN':
        K = lambda y: np.exp(-y**2/2) / np.sqrt(2*np.pi)
    elif case == 'UNIFORM':
        K = lambda y: (np.abs(y) <1 )/2
    elif case == 'TRIANGULAR':
        K = lambda y: np.fmax(0,(1-np.abs(y)))
    elif case == 'QUADRATIC' or case == 'EPANECHNIKOV':
        K = lambda y: np.fmax(0, 0.75*(1-y**2))
    else:
        raise ValueError('Kernel not recognised')
        
    f_x = np.zeros((np.shape(x)))
    n = len(xi)
    for j in range(len(x)):
        y = (x[j] - xi[:])/h
        f_x[j] = np.ones((1,n)) @ K(y) / (h*n)
    return f_x

#------------------------------------------------------------------------------
def LinearCongruential(a,c,m,seed,N=1):
    # LinearCongruential.py  -- version 2011-01-06
    #   linear congruential random number generator
    #   a, c, m ... parameters
    #   seed ...... seed
    #   N ......... number of samples

    # -- initialize
    u = float('NaN')*np.zeros((N+1))
    u[0] = seed
    if u[0]<1:
        u[0] = np.floor(u[0]*m).astype(int)
    
    # -- generate variates
    for i in range(1,N+1):
        u[i] = np.mod( (a*u[i-1] + c) , m)
    u = np.delete(u,0,0)
    u = u/m
    return u

#------------------------------------------------------------------------------
def NormalAcceptReject(n):
    # NormalAcceptReject.py  -- version 2011-01-06
    # 	generate n standard normal variates x
    x = float('NaN')*np.zeros((n,1))
    for i in range(n):
        while np.isnan( x[i] ):
            rand = np.random.rand(1)
            z = -np.log(rand)
            r = norm.pdf(z) / ( 1.32*np.exp(-z) );
            if rand < r:
                x[i] = z * np.sign(rand - 0.5);
    return x

#------------------------------------------------------------------------------
def PibySimulation(N_samples):
    # PibySimulation.py  -- version 2011-01-06
    
    a = np.random.rand(N_samples,2)*2 - 1
    r = np.sqrt(a[:,0]**2 + a[:,1]**2)
    within_unit_circle = (r<=1)
    piEst = 4 * np.sum(within_unit_circle)/N_samples
    return piEst

#------------------------------------------------------------------------------
def rearrange(x,s):
    # rearrange.py  -- version 2011-01-06
    #   x ... data sample
    #   s ... length of segment to be moved
    
    n  = len(x)
    i_start = np.ceil(np.random.rand(1)[0]*(n-s +1)-1).astype(int)
    i  = i_start + np.arange(0,(s)) # elements to be moved
    t  = np.ceil(np.random.rand()*(n-s +1)-1).astype(int) # target position
    chunk = np.copy(x[i])
    x = np.delete(x,i,0)
    x = np.hstack((x[:t:].reshape(-1,1), chunk.reshape(-1,1),  
                   x[t::].reshape(-1,1))) # insert at new position
    return x

#------------------------------------------------------------------------------
def RouletteWheel(prop,N=1):
    # RouletteWheel.py  -- version 2011-01-06
    # roulette wheel selection
    # prop ... propensities for choosing an element ( > 0 )
    # N ...... number of draws

    # -- compute (cumulative) probabilities
    prob = np.fmax(0,prop)/np.sum(np.fmax(prop,0))
    cum_prob = np.cumsum(prob)
    
    # -- perform draws
    w = float('NaN')*np.zeros((1,N))
    for i in np.arange(N):
        u = np.random.rand()
        w[0,i] = next(j for j in range(len(cum_prob)) 
            if u < cum_prob[j])
    return w

#------------------------------------------------------------------------------
def TaylorThompson(X,N,m):
    # TaylorThompson.py  -- version 2011-01-06
    #   X ... original sample
    #   N ... number of new samples to be drawn
    #   m ... number of neighbors to be used

    [xr, xc] = np.shape(X)
    # -- compute Euclidean distances
    B = X @ X.T
    ED = np.sqrt(np.tile(np.diag(B).reshape(-1,1),(1,xr)
        ) + np.tile(np.diag(B).reshape(1,-1),(xr,1)) - 2*B)
    
    # -- limits for weights
    m = min(xr,m)
    uLim = 1/m + np.asarray([-1, 1]) * np.sqrt(3*(m-1)/m**2)
    
    # -- draw samples
    Xs = np.zeros((N,xc))
    for s in range(N):
        j = np.ceil(np.random.rand()*xr).astype(int)
        # -- m nearest neighbors:
        dnn = np.sort(ED[:,j]).reshape(-1,1) 
        inn = np.argsort(ED[:,j]).tolist()
        Xnn = X[inn[0:m],:]
        # -- weights
        u = np.random.rand(1,m)[0] * (uLim[1]-uLim[0]) + uLim[0]
        # -- form linear combinations
        Xnn_bar = (np.ones((1,m)) @ Xnn / m)[0]
        e = u @ (Xnn - np.tile(Xnn_bar,(m,1)))
        Xs[s,:] = Xnn_bar + e
    return Xs

#%%
"NMOF chapter 7 functions"
from scipy.stats import norm
from scipy.stats import multivariate_normal

#------------------------------------------------------------------------------
def CopulaSim(n,copula,p):
    # CopulaSim.py  -- version 2011-01-06
    #   n ....... number of samples
    #   copula .. name of copula (Frank,Clayton,Gumbel or Gaussian)
    #   p ....... parameter of the copula

    # marginal

    Finv = lambda u: norm.ppf(u)
    # densities
    case = copula.upper()
    if case == 'FRANK':         # parameter: p<>0
        c = lambda u,v,p : ( p*(1-np.exp(-p)) * np.exp(-p*(u+v)) 
            / (np.exp(-p*(u+v)) - np.exp(-p*u) - np.exp(-p*v) 
            + np.exp(-p))**2 )
    if case == 'CLAYTON':       # parameter: p>= -1; p~=0
        c = lambda u,v,p : ( (1+p) / ((u*v)**(1+p) *(u**(-p) 
            + v**(-p)-1)**(2 +1/p) ) )
    if case == 'GUMBEL':        # parameter:  p >= 1
        c = lambda u,v,p : ( ((-np.log(u))**(p-1) ** (-np.log(v))**(p-1) 
            * ((-np.log(u))**p + (-np.log(v))**p)**(1/p-2)) 
            / (u*v*np.exp( ((-np.log(u))**p + (-np.log(v))**p)**(1/p))) )
    else: # Gaussian # parameter: -1 < p < 1
        c = lambda u,v,p : ( 1/np.sqrt(1-p**2) * np.exp((norm.ppf(u)**2 
            + norm.ppf(v)**2)/(2) + (2*p*norm.ppf(u)*norm.ppf(v) 
            - norm.ppf(u)**2-norm.ppf(v)**2) / (2*(1-p**2))) )
    # Metropolis
    u = np.nan*np.zeros((n,2))
    u[0,:] = np.random.rand(1,2)
    f_i = c(u[0,0],u[0,1],p)
    for i in range(n):
        while np.any(np.isnan(u[i,:])):
            u_new = np.random.rand(2)
            f_new = c(u_new[0],u_new[1],p)
            if np.random.rand() < f_new/f_i:
                u[i,:] = u_new
                f_i = f_new
    x = Finv(u)
    return u,x,Finv,c

#------------------------------------------------------------------------------
def DiscreteMC(prob,N,x_0=None):
#   DiscreteMC.py -- version 2011-01-06
#   discrete Markov Chain with N samples
#   prob: transition variables
    
    if x_0==None:
        uc_prob = prob**50
        x_0 = RouletteWheel(uc_prob[0,:],1)
    x = np.zeros(N+1,dtype=int)
    x[0] = x_0
    for i in range(N):
        x[i+1] = RouletteWheel(prob[x[i],:],1)
    x = np.delete(x,0)
    return x

#------------------------------------------------------------------------------
def DiscreteMC2(prob_1,prob_2,N,x_0=None):
#   DiscreteMC2.py  -- version 2011-01-06
#   discrete Markov Chain with N samples
#   prob: transition variables
    
    if x_0==None:
        x_0 = np.empty(2)
        uc_prob_1 = ( prob_2 * prob_1 )**50
        uc_prob_2 = ( prob_1 * prob_2 )**50
        x_0[0]    = RouletteWheel(uc_prob_1[0,:],1)
        x_0[1]    = RouletteWheel(uc_prob_2[0,:],1)
    
    x = np.zeros((N+1,2),dtype=int)
    x[0,:] = x_0
    for i in range(N):
        x[i+1,0] = RouletteWheel(prob_1[x[i,1],:],1)
        x[i+1,1] = RouletteWheel(prob_2[x[i,0],:],1)
    x = np.delete(x,0,axis=0)
    return x

#------------------------------------------------------------------------------
def MetropolisMVNormal(n,d,s,rho):
    # MetropolisMVNormal.py  -- version 2011-01-06
    #   generate n standard normal variates
    #   n .... number of sample
    #   d .... number of dimensions
    #   s .... step size for new variate
    #   rho .. correlation matrix; if scalar all have same corr.)

    if np.isscalar(rho):
        R = np.identity(d)*(1-rho) + np.ones(d)*rho
    else:
        R = rho
    mu = np.zeros(d)
    x = np.nan*np.zeros((n,d))
    x[0,:] = np.random.randn(1,d)
    f_i =  multivariate_normal.pdf(x[0,:],mu,R)
    for i in range(n-1):
        while np.any(np.isnan(x[i+1,:])):
            x_new = x[i,:] + 2*(np.random.rand(1,d)-.5)*s
            f_new = multivariate_normal.pdf(x_new,mu,R)
            if np.random.rand() <  f_new / f_i:
                x[i+1,:] = x_new
                f_i  = f_new
    return x

#%%
"NMOF chapter 8 functions"

#------------------------------------------------------------------------------
def ARMAsim(mu,phi,theta,sigma,T,Tb=0):
    # ARMAsim.py  -- version 2011-01-06
    
    # -- prepare parameters
    phi = phi[:];   theta = theta[:];
    p   = len(phi); q     = len(theta);
    T_compl = T + Tb + np.fmax(p,q)
    lag_p = np.arange(1,p+1)
    lag_q = np.arange(1,q+1)
    
    # -- initialise vectors
    r = np.ones(T_compl) * mu
    e = np.random.randn(T_compl) * sigma
    
    # -- simulate returns
    for t in np.arange( (np.fmax(p,q)+1), T_compl +1):
       r[t -1] = mu + r[t-lag_p -1] @ phi + e[t-lag_q -1] @ theta + e[t -1]
    
    # -- discard initial values & compute prices
    e = np.delete(e,np.arange(T_compl-T))
    r = np.delete(r,np.arange(T_compl-T))
    S = np.exp(np.cumsum(r))
    return S,r,e

#------------------------------------------------------------------------------
def ARsim(T,mu,sigma,phi):
    # ARsim.py  -- version 2011-01-06
    #   simulation of AR(p) process
    p = len(phi)
    e = np.random.randn(2*T+p,1) * sigma
    r = np.ones(2*T+p) * mu /(1-np.sum(phi[:]))
    for t in p+np.arange(2*T):
        r[t] = mu + r[t-np.arange(p)] @ phi[:] + e[t]
    r = np.delete(r,np.arange(p+T))
    e = np.delete(e,np.arange(p+T))
    return r,e

#------------------------------------------------------------------------------
def bootstrapPrice(S,n,b,S_0=None):
    # bootstrapPrice.py  -- version 2011-01-06
    """ returns one bootstrap price path
        S ..... original price series
        n ..... length of bootstrap sample x
        b ..... block length
        S_0 ... initial price for simulation (S(T) if not provided) """
    if S_0 == None:
        S_0 = S[-1,:]
    r    = np.diff(np.log(S),axis=0)               # log returns
    r_bs = bootstrap(r,n,b)
    r_bs_c = np.cumsum(np.vstack((np.log(S_0),r_bs)), 0)
    S_bs = np.exp(r_bs_c)
    return S_bs,r_bs

#------------------------------------------------------------------------------
def GARCHsim(mu,a0,a1,b1,T,Tb=0):
    # GARCHsim.py  -- version 2011-01-06
    # Tb=0: no "before" periods to swing in
    # -- initialize variables
    z = np.random.randn(T+Tb)
    e = np.zeros(T+Tb)
    h = np.zeros(T+Tb)
    h[0] = a0/(1-(a1+b1))
    e[0] = z[0] * np.sqrt(h[0])
    # -- generate sample variances and innovations
    for t in range(1,(T+Tb)):
        h[t] = a0 + a1 * e[t-1]**2 + b1 * h[t-1]
        e[t] = z[t] * np.sqrt(h[t])
    # -- remove excess observations from initialisation phase
    e = np.delete(e,np.arange(Tb))
    h = np.delete(h,np.arange(Tb))
    # -- compute returns
    r = e + mu
    return r,e,h

#------------------------------------------------------------------------------
def MAsim(T,mu,sigma,theta):
    # MAsim.py -- version 2011-01-06
    #   simulation of MA(q) process
    q = len(theta)
    e = np.random.randn(T+q) * sigma
    r = np.zeros(T+q)
    for t in q+np.arange(T):
        r[t] = mu + e[t-np.arange(q +1)] @ np.insert(theta[:],0,1)
    r = np.delete(r,np.arange(q))
    e = np.delete(e,np.arange(q))
    return r,e

#------------------------------------------------------------------------------
def Simulate1overN(mu,sigma,rho,N_samples,N_stocks):
    # Simulate1overN.py  -- version 2011-01-06
    #   mu, sigma ...: drift and volatility (same for all stocks)
    #   rho .........: linear correlation
    #   N_samples ...: number of samples
    #   N_stocks ....: maximum number of stocks
    
    CovMat = np.identity(N_stocks) * sigma**2 + (
        np.ones((N_stocks,N_stocks)) - np.identity(N_stocks)
        ) * sigma**2 * rho
    e = np.random.randn(N_samples,N_stocks)@ np.linalg.cholesky(CovMat).T
    r = mu + e
    
    # compute mean return for equally weighted portfolio
    # of first 1 ... N_stocks stocks
    r_Pf   = float('NaN')*np.zeros((N_samples,N_stocks))
    for i in range(N_stocks):
        w = (np.ones((i +1,1))/(i +1))
        r_Pf[:,i] = (r[:,:i +1:] @ w).flatten()
    return r_Pf

#------------------------------------------------------------------------------
def SimulateStauPtau(S_0,drift,vol,X,rSafe,T,tau=None,NSample=10000):
    # SimulateStauPtau.py  -- version 2011-01-06
    """ simulate portfolio with  1 stock and 1 put at time tau
        S_0       initial stock price with return ~ N(drift,vol^2)
        X         strike of put
        rSafe     riskfree rate of return
        T         time to maturity of put
        tau       point of valuation; if not provided: tau = T
        NSample   number of samples for the simulation """
    if tau == None:
        tau = T  
    else: 
        tau = np.fmin(T,tau)
    # -- simulate stock prices at tau
    r = np.random.randn(NSample)*(vol*np.sqrt(tau)) + drift * tau
    S_tau = S_0 * np.exp(r)
    # -- compute put prices at tau; time left to maturity (T-tau)
    T_left = T - tau
    p_tau = BSput(S_tau,X,rSafe,0,T_left,vol)
    # -- value of portfolio at time tau
    PF_tau  = S_tau + p_tau
    return PF_tau,S_tau,p_tau

#------------------------------------------------------------------------------
def SimulateStk(mu,sigma,N_samples):
    # SimulateStk.py  -- version 2018-09-29
    # mu, sigma ....: drift and volatility
    # N ............: number of samples
    SimStk = {}
    SimStk['mu'] = mu
    SimStk['sigma'] = sigma
    SimStk['r'] = np.random.randn(N_samples,1)*sigma + mu
    SimStk['S'] = np.exp(SimStk['r'])
    SimStk['meanr'] = np.mean(SimStk['r'])
    SimStk['stdr'] = np.std(SimStk['r'])
    SimStk['meanS'] = np.mean(SimStk['S'])
    SimStk['muS'] = np.exp(mu+(sigma**2)/2)
    SimStk['medS'] = np.median(SimStk['S'])
    
    r = SimStk['r']; S_T = SimStk['S']
    # display results
    print('        mean:', np.round(SimStk['meanr'],2), 
          ' (mu   =', mu,')\n')
    print('  volatility:', np.round(SimStk['stdr'],2), 
          ' (E(r) =', sigma, ')\n')
    print('  exp. price:', np.round(SimStk['meanS'],2), 
          ' (E(S) =', np.round(np.exp(mu+(sigma**2)/2),2),')\n')
    print('median price:', np.round(SimStk['medS'],2), 
          ' (M(S) =', np.round(np.exp(mu),2),')\n')
    return SimStk

#------------------------------------------------------------------------------
def Timmermann(T,r,mu,sigma,n):
    # Timmermann.py  -- version 2011-01-06
    g_true = mu + np.random.randn(T+n) * sigma
    mu_hat = np.zeros(T+n)
    var_hat = np.zeros(T+n)
    log_div = np.zeros(T+n)
    mu_hat[n]  = np.mean(g_true[:n])
    var_hat[n] = np.var(g_true[:n])
    log_div[n] = 0
    weight = (n-1)/n
    for t in np.arange( (n+1), (T+n) ):
        mu_hat[t]  = weight * mu_hat[t-1]  + (1/n) * g_true[t]
        var_hat[t] = weight * var_hat[t-1] + (1/n) * ( weight 
            * ( g_true[t]-mu_hat[t-1])**2)
        log_div[t] = log_div[t-1] + g_true[t]
    # -- discard first n observations (preceeding observations)
    mu_hat = np.delete(mu_hat,np.arange(n))
    var_hat = np.delete(var_hat,np.arange(n))
    log_div = np.delete(log_div,np.arange(n))
    # -- compute prices
    Div   = np.exp(log_div)
    g_est = np.exp(mu_hat + var_hat/2)
    g_est = np.fmin(g_est,(1 + r -.0001))
    # -- price under rational expectations
    S_RE = np.multiply( Div , np.divide(g_est, (1+r-g_est)) )
    # -- fundamental price under known true parameters
    S_FT = np.multiply( Div, (np.exp(mu + sigma**2/2)/(1+r-np.exp(mu+sigma**2/2))) )
    return S_RE,Div,g_est,S_FT

#%%
" NMOF chapter 9 functions"
import numpy as np
import scipy as sp
from scipy.stats import norm

#------------------------------------------------------------------------------
def bridge(t0,ttau,z0,ztau,M):
    # bridge.py -- version 2010-12-08
    dt = (ttau-t0)/M
    vt = np.linspace(t0,ttau,M+1).T
    vz = np.insert(np.cumsum(np.random.randn(M,1) * np.sqrt(dt)),0,0)
    b = z0 + vz - np.multiply( 
            ((vt - t0)/(ttau - t0)), (vz[M] - ztau + z0) )
    return b

#------------------------------------------------------------------------------
def callBSM(S,X,tau,r,q,v):
    # callBSM.py -- version 2010-12-08
    """ S   = spot
        X   = strike
        tau = time to mat
        r   = riskfree rate
        q   = dividend yield
        v   = volatility^2 """
    d1 = ( np.log(S/X) + (r - q + v/2)*tau ) / np.sqrt(v*tau)
    d2 = d1 - np.sqrt(v*tau)
    call = S*np.exp(-q*tau)*norm.cdf(d1) - X*np.exp(-r*tau)*norm.cdf(d2)
    return call

#------------------------------------------------------------------------------
def callBSMMC(S,X,tau,r,q,v,M,N):
    # callBSMMC.py -- version 2010-12-10
    """ S   = spot
        X   = strike
        tau = time to mat
        r   = riskfree rate
        q   = dividend yield
        v   = volatility^2 
        M   = time steps
        N   = number of paths """
    S = pricepaths(S,tau,r,q,v,M,N)
    payoff = np.fmax((S[-1,:]-X),0)
    payoff = np.exp(-r*tau) * payoff
    call = np.mean(payoff)
    return call,payoff

#------------------------------------------------------------------------------
def callBSMMC2(S,X,tau,r,q,v,M,N):
    # callBSMMC2.py -- version 2010-12-10
    """ S   = spot
        X   = strike
        tau = time to mat
        r   = riskfree rate
        q   = dividend yield
        v   = volatility^2 
        M   = time steps
        N   = number of paths """
    dt = tau/M
    g1 = (r - q - v/2)*dt
    g2 = np.sqrt(v*dt)
    sumPayoff = 0
    T = 0
    Q = 0
    s = np.log(S)
    for n in range(N):
        z = g1 + g2*np.random.randn(M)
        z = np.cumsum(z)+s
        Send = np.exp(z[len(z)-1])
        payoff = np.fmax(Send-X,0)
        sumPayoff = payoff + sumPayoff
        # compute variance
        if n > 0:
            n += 1
            T = T + payoff
            Q = Q + (1/(n*(n-1))) * (n*payoff - T)**2
        else:
            T = payoff
    call = np.exp(-r*tau) * (sumPayoff/N)
    return call,Q

#------------------------------------------------------------------------------
def callBSMMC3(S,X,tau,r,q,v,M,N):
    """ S   = spot
        X   = strike
        tau = time to mat
        r   = riskfree rate
        q   = dividend yield
        v   = volatility^2 
        M   = time steps
        N   = number of paths """
    dt = tau/M
    g1 = (r - q - v/2)*dt
    g2 = np.sqrt(v*dt)
    sumPayoff = 0
    T = 0
    Q = 0
    s = np.log(S)
    for n in range(N):
        ee = g2 * np.random.randn(M)
        z = g1 + ee
        z = np.cumsum(z)+s
        Send = np.exp(z[len(z)-1])
        payoff = np.fmax(Send-X,0)
        z = g1 - ee
        z = np.cumsum(z)+s
        Send = np.exp(z[len(z)-1])
        payoff = payoff + np.fmax(Send-X,0)
        payoff = payoff/2
        sumPayoff = payoff + sumPayoff
        # compute variance
        if n>0:
            n += 1
            T = T + payoff
            Q = Q+ (1/(n*(n-1))) * (n*payoff - T)**2
        else:
            T = payoff
    call = np.exp(-r*tau) * (sumPayoff/N)
    return call,Q

#------------------------------------------------------------------------------
def callBSMMC4(S,X,tau,r,q,v,M,N):
    # callBSMMC4.py -- version 2010-12-08
    """ S   = spot
        X   = strike
        tau = time to mat
        r   = riskfree rate
        q   = dividend yield
        v   = volatility^2 
        M   = time steps
        N   = number of paths """
    dt = tau/M
    g1 = (r - q - v/2)*dt
    g2 = np.sqrt(v*dt)
    s = np.log(S)
    ee = g2 * np.random.randn(M,N)
    z = np.cumsum(g1+ee,0)+s # cumsum(...,1) in case of M=1!
    S = np.exp(z[len(z)-1,:])
    payoff = np.fmax(S-X,0)
    z = np.cumsum(g1-ee,0)+s # cumsum(...,1) in case of M=1!
    S = np.exp(z[len(z)-1,:])
    payoff = payoff + np.fmax(S-X,0)
    payoff = payoff/2
    payoff = np.exp(-r*tau) * payoff
    call = np.mean(payoff)
    return call,payoff

#------------------------------------------------------------------------------
def callBSMMC5(S,X,tau,r,q,v,M,N):
    # callBSMMC5.py -- version 2010-12-10
    """ S   = spot
        X   = strike
        tau = time to maturity
        r   = riskfree rate
        q   = dividend yield
        v   = volatility^2
        M   = time steps
        N   = number of paths """
    dt = tau/M
    g1 = (r - q - v/2)*dt; g2 = np.sqrt(v*dt);
    sumPayoff = 0; T = 0; Q = 0;
    s = np.log(S)
    # determine beta
    nT = 2000
    sampleS = S*np.exp(g1*tau + np.sqrt(v*tau) * np.random.randn(nT,1))
    sampleO = np.exp(-r*tau) * np.fmax(sampleS - X, 0)     
    aux     = sp.linalg.lstsq(
              np.hstack((np.ones((nT,1)), sampleS)),sampleO)[0]
    beta    = -aux[1][0]              
    expS    = S*np.exp((r-q)*tau) # expected stock price
    # run paths
    for n in range(N):
        z = g1 + g2*np.random.randn(M,1)
        z = np.cumsum(z)+s
        Send   = np.exp(z[-1])
        payoff = np.fmax(Send-X,0) + beta*(Send-expS)
        sumPayoff = payoff + sumPayoff
        #compute variance
        if n>0:
            n += 1
            T = T + payoff
            Q = Q + (1/(n*(n-1))) * (n*payoff - T)**2
        else:
            T = payoff
    call = (np.exp(-r*tau) * (sumPayoff/N))
    return call,Q

#------------------------------------------------------------------------------
def callBSMQMC(S,K,tau,r,q,v,N):
    # callBSMQMC.py -- version 2010-12-08
    """ S   = spot
        X   = strike
        tau = time to mat
        r   = riskfree rate
        q   = dividend yield
        v   = volatility^2 """
    g1 = (r - q - v/2)*tau
    g2 = np.sqrt(v*tau)
    U  = VDC(np.arange(1,N+1),7)
    ee = g2 * norm.ppf(U) # norminv(U)
    z  = np.log(S) + g1 + ee
    S  = np.exp(z)
    payoff = np.exp(-r*tau) * np.fmax(S-K,0)
    call = np.sum(payoff)/N
    return call

#------------------------------------------------------------------------------
def callHestonMC(S,X,tau,r,q,v0,vT,rho,k,sigma,M,N):
# callHestonMC.py -- version 2011-01-08
    """ S     = spot
        X     = strike
        tau   = time to maturity
        r     = riskfree rate
        q     = dividend yield
        v0    = initial variance
        vT    = long run variance (theta in Heston's paper)
        rho   = correlation
        k     = speed of mean reversion (kappa in Heston's paper)
        sigma = vol of vol
        M     = time steps
        N     = number of paths """
    dt = tau/M
    sumPayoff = 0
    C = np.vstack(( np.asarray([1, rho]), np.asarray([rho, 1]) ))
    C = np.linalg.cholesky(C).T
    T = 0
    Q = 0
    for n in range(N):
        ee = np.random.randn(M,2)
        ee = ee @ C
        vS = np.log(S)
        vV = v0
        for t in range(M):
            # --update stock price
            dS = (r - q - vV/2)*dt + np.sqrt(vV)*ee[t,0]*np.sqrt(dt)
            vS = vS + dS
            # --update squared vol
            aux = ee[t,1]
            # --Euler scheme
            dV = k*(vT-vV)*dt + sigma*np.sqrt(vV)*aux*np.sqrt(dt)
            # --absorbing condition
            if (vV + dV) < 0:
                vV = 0
            else:
                vV = vV + dV
            # --zero variance: some alternatives (omitted)
        Send = np.exp(vS)
        payoff = np.fmax(Send-X,0)
        sumPayoff = payoff + sumPayoff
        #compute variance
        if n>0:
            n += 1
            T = T + payoff
            Q = Q + (1/(n*(n-1))) * (n*payoff - T)**2
        else:
            T = payoff
    call = np.exp(-r*tau) * (sumPayoff/N)
    return call,Q

#------------------------------------------------------------------------------
def CPPIgap(S, m, G, r_c, gap=1):
    # CPPIgap.py version -- 2018-09-29
    """ S   .. stock price series t = 0 .. T
        G   .. guaranteed payback amount
        m   .. multiplier
        r_c .. cumulated save return over entire horizon
        gap .. readjustment frequency; if blank: 1 = always """
    
    # -- initial setting
    T = len(S)-1
    t = np.arange(T +1)
    
    V    = np.zeros(T +1)
    V[0] = G
    
    F = G*np.exp(-r_c*((T-t)/T))
    
    C = np.zeros((T +1))
    B = np.zeros((T +1))
    n = np.zeros((T +1))     # number of risky assets
    
    # -- development over time
    E = np.zeros(T)
    for tau in range(T):       # tau = t+1
        C[tau]      = V[tau]-F[tau]
        if (tau % gap) == 0: # re-adjust now
            E[tau] = np.fmin(m * C[tau], V[tau])
            n[tau] = E[tau] / S[tau]
            B[tau] = V[tau] - E[tau]
        else:
            n[tau] = n[tau -1]
            E[tau] = V[tau] - B[tau]
        
        B[tau +1] = B[tau]*np.exp(r_c/T)
        V[tau +1] = n[tau]*S[tau +1] + B[tau +1]
        
    return V, C, B, F, E

#------------------------------------------------------------------------------
def digits(k,b):
    # digits.py -- version 2010-12-08
    nD = 1 + np.floor(np.log(np.max(k))/np.log(b)).astype(int) # required digits
    nN = [len(k) if type(k)!= int else 1][0]
    dd = np.zeros((nN,nD))        
    for i in np.arange(nD-1,-1,-1):
        dd[:,i] = np.mod(k,b)
        if i>0:
            k = np.fix(k/b)
    return dd

#------------------------------------------------------------------------------
def pricepaths(S,tau,r,q,v,M,N):
    # pricepaths.py -- version 2010-12-08
    """ S   = spot
        tau = time to mat
        r   = riskfree rate
        q   = dividend yield
        v   = volatility^2
        M   = time steps
        N   = number of paths """
    dt = tau/M
    g1 = (r - q - v/2)*dt
    g2 = np.sqrt(v * dt)
    aux = np.cumsum(np.vstack((np.log(S)*np.ones((1,N)), 
        g1 + g2 * np.random.randn(M,N))),axis=0)
    paths = np.exp(aux)
    return paths

#------------------------------------------------------------------------------
def VaRHill(r,m,a):
# VaRHill.py -- version 2011-01-06
    """ r ... historical returns
        m ... number of largest losses used
        a ... probability VaR is exceeded """

    # -- adjusted parameters where necessary
    if m < 1:
        m = np.ceil(m*len(r))
    if np.any(a) > 1:
        a = a/100
    if np.any(a) > .5:
        a = 1-a
    
    # --compute Hill estimator
    r_order = np.sort(r,0)
    ksi = np.sum(np.log(r_order[:m]/r_order[m+1]))/m
    
    # --compute the VaR and ES
    VaR = r_order[m+1] * ( (m/len(r)) / a ) **ksi
    ES = VaR / (1-ksi)
    return VaR, ES, ksi

#------------------------------------------------------------------------------
def VDC(k,b):
    # VDC.py -- version 2010-09-29
    nD = (1 + np.floor(np.log(np.max(k))/np.log(b))).astype(int) # required digits
    nN = [len(k) if type(k)!= int else 1][0] # number of VDC numbers
    vv = np.zeros((nN,nD))         
    for i in np.arange(nD-1,-1,-1):
        vv[:,i] = np.copy(np.mod(k,b))
        if i>0:
            k = np.fix(k/b)
    ex = (b ** np.arange(nD-1,-1,-1))[0]
    vv = np.divide(vv,np.ones((nN,1))*ex)
    vv = np.sum(vv,1)
    return vv

