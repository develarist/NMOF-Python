# exUniforms.py -- version 2011-01-07
# generate normals, check correlations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform

X = np.random.randn(1000,4)
np.corrcoef(X,rowvar=False)

# desired linear correlation
M = np.asarray([[1.0,  0.7,  0.6,  0.6],
    [0.7,  1.0,  0.6,  0.6],
    [0.6,  0.6,  1.0,  0.8],
    [0.6,  0.6,  0.8,  1.0]])

# adjust correlations for uniforms
M = 2 * np.sin(np.pi/6 * M)

# induce correlation, check correlations
C = np.linalg.cholesky(M)
Xc = X @ C
np.corrcoef(Xc,rowvar=False)

# create uniforms, check correlations
Xc[:,np.arange(2,4)] = norm.cdf(Xc[:,np.arange(2,4)])
np.corrcoef(Xc,rowvar=False)

# plot results (marginals)
plt.figure(figsize=(8,6))
for i in range(1,5):
    plt.subplot(2,2,i)
    plt.hist(Xc[:,i-1])
    plt.title('X%s' %i)
plt.show()