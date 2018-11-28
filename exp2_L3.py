from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import pickle as pkl
from time import time
from utils import csr_cosine_similarity
from utils import HwGlobal as hg
import numpy as np
from scipy.sparse.linalg import norm
resFile = open('output/exp2_L3.log', 'w')

t1 = time()

with open('output/testMatrix.pkl', 'rb') as f:
    testMatrix = pkl.load(f)
with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f)
with open('output/trainMatrixL3.pkl', 'rb') as f:
    trainMatrixL3 = pkl.load(f)
with open('output/simiMatL3.pkl', 'rb') as f:
    simiMat = pkl.load(f)

t2 = time()
print('Load data: %.2fms' % ((t2 - t1) * 1000), file=resFile)

upper = simiMat * trainMatrixL3

t3 = time()
print('Calc upper: %.2fms' % ((t3 - t2) * 1000), file=resFile)

mask = (trainMatrix != 0).astype(int)
lower = np.abs(simiMat) * mask

t4 = time()
print('Calc lower: %.2fms' % ((t4 - t3) * 1000), file=resFile)

predMat = upper.multiply(lower.power(-1))

n = testMatrix.count_nonzero()
testMask = (testMatrix != 0).astype(int)
maskPredMat = predMat.multiply(testMask) + testMask.multiply(3)
rmse = norm(maskPredMat - testMatrix) / np.sqrt(n)

t5 = time()
print('Predict & test: %.2fms' % ((t5 - t4) * 1000), file=resFile)
print('Total: %.2fms' % ((t5 - t2) * 1000), file=resFile)

print('RMSE = %.3f' % rmse, file=resFile)
resFile.close()
