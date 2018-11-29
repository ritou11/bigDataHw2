from scipy.sparse import csr_matrix
from scipy.sparse import diags
import pickle as pkl
from time import time
from utils import HwGlobal as hg
import numpy as np
from scipy.sparse.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set_style('ticks')
resFile = open('output/exp3_LM.log', 'w')

lbd = 1e-2
k = 50
alpha = 5e-4
eps = 1e3
init_uv = 1e-2
MAX_ITER = 400

t1 = time()

with open('output/testMatrix.pkl', 'rb') as f:
    testMatrix = pkl.load(f)
with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f)
with open('output/testMatrix.pkl', 'rb') as f:
    testMatrix = pkl.load(f)
mask = (trainMatrix != 0).astype(int)
trainRowMeans = list()
for i in range(hg.N):
    trainRowMeans.append(trainMatrix.getrow(i).sum() / trainMatrix.getrow(i).nnz)
trainMean = diags(trainRowMeans, shape=(hg.N, hg.N))
trainMatrixLM = trainMatrix - trainMean * mask

t2 = time()
print('Load data: %.2fms' % ((t2 - t1) * 1000), file=resFile)

n = testMatrix.count_nonzero()
N = hg.N
A = (trainMatrix != 0).astype(int)
mask = (testMatrix != 0).astype(int)
U = np.matrix(np.random.rand(N,k)) * init_uv
V = np.matrix(np.random.rand(N,k)) * init_uv
rmseList = list()
jList = list()
# Step 0
UVT = U * V.T
AUVT = A.multiply(UVT)
AUVTX = AUVT - trainMatrixLM
rmse = norm(mask.multiply(UVT) + trainMean * mask - testMatrix) / np.sqrt(n)
J = 0.5 * norm(AUVTX)**2 + lbd * np.linalg.norm(U)**2 + lbd * np.linalg.norm(V)**2
jList.append(J)
rmseList.append(rmse)

t0 = t3 = time()
print('Init values: %.2fms' % ((t3 - t2) * 1000), file=resFile)

for i in range(MAX_ITER):
    pJU = AUVTX * V + 2 * lbd * U
    pJV = AUVTX.T * U + 2 * lbd * V
    # TODO: Gauss
    U = U - alpha * pJU
    V = V - alpha * pJV

    UVT = U * V.T
    AUVT = A.multiply(UVT)
    AUVTX = AUVT - trainMatrixLM

    rmseL = rmse
    rmse = norm(mask.multiply(UVT) + trainMean * mask - testMatrix) / np.sqrt(n)
    JL = J
    J = 0.5 * norm(AUVTX)**2 + lbd * np.linalg.norm(U)**2 + lbd * np.linalg.norm(V)**2
    jList.append(J)
    rmseList.append(rmse)

    t4 = time()
    print('Step %d: %.2fms' % (i + 1 ,(t4 - t3) * 1000), file=resFile)
    print('rmse = %.3f, J = %.1f' % (rmse, J), file=resFile)
    t3 = t4

    if JL - J < eps:
        break

print('Average step time = %.2fms' % ((t3 - t0) * 1000 / (i + 1)), file=resFile)

with open('output/exp3_lm_jlist.pkl', 'wb') as f:
    pkl.dump(jList, f)
with open('output/exp3_lm_rmselist.pkl', 'wb') as f:
    pkl.dump(rmseList, f)

fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(rmseList, 'r-')
plt.ylabel("RMSE")
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line2, = ax2.plot(jList, 'b-')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.ylabel("J")
plt.legend([line1, line2], ["RMSE", "J"])
fig.savefig('output/exp3_lm_k%dl%d.png' % (k, -np.log10(lbd)), dpi=300)

resFile.close()
