from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import pickle as pkl
from time import time
from utils import csr_cosine_similarity
from utils import HwGlobal as hg
import numpy as np
from scipy.sparse.linalg import norm
resFile = open('output/exp2.log', 'w')

t1 = time()

with open('output/testMatrix.pkl', 'rb') as f:
    testMatrix = pkl.load(f)
with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = np.matrix(pkl.load(f).todense())
with open('output/simiMat.pkl', 'rb') as f:
    simiMat = np.matrix(pkl.load(f).todense())

t2 = time()
print('Load data: %.2fms' % ((t2 - t1) * 1000), file=resFile)

mask = (trainMatrix > 0)
predMat = simiMat * trainMatrix / (simiMat * mask)

t3 = time()
print('Pred: %.2fms' % ((t3 - t2) * 1000), file=resFile)

n = testMatrix.count_nonzero()
testMask = (testMatrix != 0).astype(int)
maskPredMat = testMask.multiply(predMat)
rmse = norm(maskPredMat - testMatrix) / np.sqrt(n)

t5 = time()
print('Test: %.2fms' % ((t5 - t3) * 1000), file=resFile)
print('Total: %.2fms' % ((t5 - t2) * 1000), file=resFile)

print('RMSE = %.3f' % rmse, file=resFile)
resFile.close()
