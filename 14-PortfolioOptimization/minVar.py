# minVar.py -- version 2010-12-12
import numpy as np
import cvxopt as cvx

# generate artificial returns data
ns = 60     # number of scenarios
na = 10     # number of assets
R = 0.005 + np.random.randn(ns,na) * 0.015
Q = cvx.matrix(2 * np.cov(R.T))

# set up
c = cvx.matrix(0.0, (na,1))
A = cvx.matrix(1.0, (1,na))
a = cvx.matrix(1.0)
B = cvx.matrix(-np.identity(na))
b = cvx.matrix(0.0, (na,1))

# solution
cvx.solvers.options['show_progress'] = False 
w = cvx.solvers.qp(Q,c,B,b,A,a)
w = np.asarray(w['x'])

# check constraints
print(np.sum(w))
print(np.all(w>=0))