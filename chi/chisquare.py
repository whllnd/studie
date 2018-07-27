import numpy as np
from scipy.stats import chisquare

# Ideas:
#
#    | 2 | 3 | 6
# FP
# FN

# Observed distribution
#M = np.array([[63, 52, 43],
#              [69, 48, 40]])
#M = np.array([[62, 296],
#              [43, 168]])
#M = np.array([[7, 33],    # 6 hours, FN
#              [4, 8],     #          FP
#              [19, 87],   #          TN
#              [16, 62]])  #          TP
#M = np.array([[6, 33],    # 12 hours, FN
#              [6, 18],    #           FP
#              [17, 77],   #           TN
#              [17, 62]])  #           TP
M = np.array([[11, 58],    # 2 hours, FN
              [4, 26],    #          FP
              [19, 69],   #          TN
              [12, 37]])  #          TP


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
print("Chi Squared:")
print(XSq)
print("Chi Squared Distance:", XSq.sum())
DF = (M.shape[0]-1) * (M.shape[1]-1)
print("Degrees of freedom:", DF)

