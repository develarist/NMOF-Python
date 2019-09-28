# exVaRThreshold.py -- version 2018-09-29
import numpy as np
from scipy import stats
from functions import VaRHill
from functions import quantile
from functions import CornishFisherSimulation

N     = 1250
Nruns = 1000
alpha = np.logspace(0,1,18) ** 2 / 1e3 # [.001 .005 .01 .025 .05 .075 .1]
mOverNSet = np.array([ .01, .025, .05, .1])
sizemOverNSet = len(mOverNSet)+1
mSet = (np.ceil(mOverNSet*N)).astype(int)

## ..... normal distribution
VaR_normal = np.nan*np.zeros((Nruns,len(alpha),len(mSet)+1))
for run in range(Nruns):
    r = np.random.randn(N)
    for j in range(len(mSet)):
        m = mSet[j]
        VaR_normal[run,:,j] = VaRHill(r,m,alpha)[0]
    VaR_normal[run,:,-1] = np.quantile(r,alpha)

## ..... student t distribution
VaR_student = np.nan*np.zeros((Nruns,len(alpha),len(mSet)+1))
for run in range(Nruns):
    r = stats.t.ppf(np.random.rand(N,1),5)
    for j in range(len(mSet)):
        m = mSet[j]
        VaR_student[run,:,j] = VaRHill(r,m,alpha)[0]
    VaR_student[run,:,-1] = np.quantile(r,alpha)

## ...... via Cornish Fisher approximation
VaR_CF = np.nan*np.zeros((Nruns,len(alpha),len(mSet)+1))
for run in range(Nruns):
    r = CornishFisherSimulation(0,1,-.3,3,N)
    for j in range(len(mSet)):
        m = mSet[j]
        VaR_CF[run,:,j] = VaRHill(r,m,alpha)[0]
    VaR_CF[run,:,-1] = np.quantile(r,alpha)

