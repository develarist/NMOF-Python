# meanVar.py -- version 2011-05-16
#%% compute a mean--variance efficient portfolio (long-only)
import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt

# generate artificial returns data
ns = 60 # number of scenarios
na = 10 # number of assets
R = 0.005 + np.random.randn(ns, na) * 0.015
Q = cvx.matrix(2 * np.cov(R.T))
m = np.mean(R,0).reshape(1,-1)
rd = 0.0055 # required return

c = cvx.matrix(0.0, (na,1))
A = cvx.matrix(1.0, (1,na))                       # equality constraints
a = cvx.matrix(1.0)
B = cvx.matrix(np.vstack((-m, -np.identity(na)))) # inequality constraints
b = cvx.matrix(np.vstack((np.array(-rd), np.zeros((na,1)))))

cvx.solvers.options['show_progress'] = False 
w = cvx.solvers.qp(Q,c,B,b,A,a)
w = np.asarray(w['x'])

# check constraints
print(np.sum(w)), print(np.all(w>=0)), print((m @ w)[0])

#%% compute and plot a whole frontier (long-only)
npoints = 500
lamda = np.sqrt(1-np.linspace(0.99,0.05,npoints)**2)
B = cvx.matrix(-np.identity(na))
b = cvx.matrix(0.0, (na,1))
for i in range(npoints):
    Q = cvx.matrix(2*lamda[i] * np.cov(R.T))
    c = cvx.matrix(-(1-lamda[i]) * m.T)
    w = np.asarray(cvx.solvers.qp(Q,c,B,b,A,a)['x'])
    plt.plot(np.sqrt(w.T@np.cov(R.T)@w), m@w,'r.')
plt.xlabel('Volatility')
plt.ylabel('Expected portfolio return')
plt.show()
