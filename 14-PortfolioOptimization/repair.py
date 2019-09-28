# repair.py -- version 2018-09-30
import numpy as np

# --compute eigenvectors/-values
[D,V] = np.linalg.eig(C)
D = np.diag(D)

# --replace negative eigenvalues by zero
D = np.fmax(D, 0)

# --reconstruct correlation matrix
CC = V @ D @ V.T

# --rescale correlation matrix
S       = (1 / np.sqrt(np.diag(CC))).reshape(-1,1) # S = 1 ./ sqrt(diag(CC))
SS      = S @ S.T
C       = CC*SS
