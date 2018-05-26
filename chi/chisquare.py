import numpy as np
from scipy.stats import chisquare

# Ideas:
#
#    | 2 | 3 | 6
# FP
# FN

# Observed distribution
M = np.array([[63, 52, 43],
              [69, 48, 40]])

# Total number of observations
Total = np.zeros((M.shape[0]+1, M.shape[1]+1), dtype=int)
Total[:M.shape[0], :M.shape[1]] = M
Total[-1,:-1] = [M[:,j].sum() for j in range(M.shape[1])]
R = [M[j].sum() for j in range(M.shape[0])]
Total[:,-1] = R + [sum(R)]
N = M.sum()

# Expected distribution
Mexp = np.zeros(M.shape)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        Mexp[i,j] = M[i].sum() * M[:,j].sum() / N

print("Observed samples:")
print(M)
print("Total:")
print(Total)
print("Expected samples:")
print(Mexp)
print("Column sums:")
print([M[:,i].sum() for i in range(M.shape[1])])

XSq = ((M - Mexp)**2) / Mexp
print("Chi Squared:", XSq)
print("Chi Squared Distance:", XSq.sum())
DF = (M.shape[0]-1) * (M.shape[1]-1)
print("Degrees of freedom:", DF)

