from scipy.sparse import csr_matrix
from scipy.sparse import diags
import pickle as pkl
from time import time
from utils import csr_cosine_similarity
from utils import HwGlobal as hg
import numpy as np
from scipy.sparse.linalg import norm
resFile = open('output/exp2_LM.log', 'w')
t1 = time()

with open('output/testMatrix.pkl', 'rb') as f:
    testMatrix = pkl.load(f)
with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f)
mask = (trainMatrix != 0).astype(int)
trainRowMeans = list()
for i in range(hg.N):
    trainRowMeans.append(trainMatrix.getrow(i).sum() / trainMatrix.getrow(i).nnz)
trainMean = diags(trainRowMeans, shape=(hg.N, hg.N))
trainMatrixLM = trainMatrix - trainMean * mask
simiMat = csr_cosine_similarity(trainMatrixLM)

t2 = time()
print('Load data: %.2fms' % ((t2 - t1) * 1000), file=resFile)

upper = simiMat * trainMatrixLM

t3 = time()
print('Calc upper: %.2fms' % ((t3 - t2) * 1000), file=resFile)

lower = np.abs(simiMat) * mask

t4 = time()
print('Calc lower: %.2fms' % ((t4 - t3) * 1000), file=resFile)

predMat = upper.multiply(lower.power(-1))

t5 = time()
print('Predict: %.2fms' % ((t5 - t4) * 1000), file=resFile)

n = testMatrix.count_nonzero()
testMask = (testMatrix != 0).astype(int)
maskPredMat = predMat.multiply(testMask) + trainMean * testMask
rmse = norm(maskPredMat - testMatrix) / np.sqrt(n)

t6 = time()
print('Predict test: %.2fms' % ((t6 - t5) * 1000), file=resFile)

print('RMSE = %.2f' % rmse, file=resFile)
resFile.close()
