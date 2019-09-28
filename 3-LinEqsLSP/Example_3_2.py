# Example 3.2 NMOF
import numpy as np
from functions import OmegaGridSearch
from functions import SOR

#np.random.seed(1999) #np.random.randn('state',999)
n = 100; m = 30;
X = np.hstack((np.ones((n,1)), np.random.randn(n,m-1)))
A = X.T@X
x = np.ones((m,1)); b = A@x; x1 = 2*x;
maxit = 80; tol = 1e-4;
wopt,_ = OmegaGridSearch(A)
omega = 1
[sol1,nit1] = SOR(x1,A,b,omega,tol,maxit)
omega = wopt
[sol2,nit2] = SOR(x1,A,b,omega,tol,maxit)
print('\n nit1 = ', nit1, 'nit2 = ', nit2, '\n')

# nit1 = 14;   nit2 = 10;