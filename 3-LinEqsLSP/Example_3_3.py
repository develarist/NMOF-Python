# Example 3.3
import timeit
import numpy as np
import scipy.linalg as linalg
from scipy.sparse import spdiags  
from functions import lu3diag
from functions import solve3diag

n = 500; m = 100000;
c = np.arange(1,n+1) / 3; d = np.ones(n); x = np.ones((n,1));
p = -c[1:n]; q = c[0:n-1];
A = spdiags(np.hstack((np.append(p, np.nan).reshape(-1,1), 
            d.reshape(-1,1), np.insert(q,0,np.nan).reshape(-1,1))).T,
            np.arange(-1,2),n,n,format=None)
b = (A@x).flatten()
#
start = timeit.default_timer()
A = A.toarray()
L = linalg.lu(A)[1]
U = linalg.lu(A)[2]
#for k in np.arange(m):
s1 = linalg.solve(U,(linalg.solve(L,b)))
stop = timeit.default_timer()
print('\n Sparse Matlab {0:.6f}'.format(np.fix(stop)),'seconds.')

start = timeit.default_timer()
l,u = lu3diag(p,d,q)
#for k in np.arange(m):
s2 = solve3diag(l,u,q,b)
stop = timeit.default_timer()
print('\n Sparse code {0:.6f}'.format(np.fix(stop)),'seconds.')

#Sparse Matlab 5 (sec)
#    Sparse code 9 (sec)