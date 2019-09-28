# spearman.py  version -- 2011-01-10
import numpy as np
import scipy.stats as stats

Y = np.random.randn(20)
Z = np.random.randn(20)
print(stats.spearmanr(Y,Z)[0])
#
indexY = np.argsort(Y)
indexZ = np.argsort(Z)
ranksY = np.argsort(indexY)
ranksZ = np.argsort(indexZ)
print(stats.spearmanr(ranksY,ranksZ)[0])