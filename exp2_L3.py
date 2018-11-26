from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import pickle as pkl
from time import time
from utils import csr_cosine_similarity
from utils import HwGlobal as hg
import numpy as np
from scipy.sparse.linalg import norm

t1 = time()

with open('output/testMatrix.pkl', 'rb') as f:
    testMatrix = pkl.load(f)
with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f)
with open('output/simiMat.pkl', 'rb') as f:
    simiMat = pkl.load(f)

t2 = time()
print('Load data: %.2fms' % ((t2 - t1) * 1000))

upper = simiMat * trainMatrix

t3 = time()
print('Calc upper: %.2fms' % ((t3 - t2) * 1000))

mask = (trainMatrix != 0).astype(int)
lower = simiMat * mask

t4 = time()
print('Calc lower: %.2fms' % ((t4 - t3) * 1000))

predMat = upper.multiply(lower.power(-1))

t5 = time()
print('Predict: %.2fms' % ((t5 - t4) * 1000))

n = testMatrix.count_nonzero()
testMask = (testMatrix != 0).astype(int)
maskPredMat = predMat.multiply(testMask)
rmse = norm(maskPredMat - testMatrix) / np.sqrt(n)

t6 = time()
print('Predict test: %.2fms' % ((t6 - t5) * 1000))

print('RMSE = %.2f' % rmse)
with open('output/predMat.pkl', 'wb') as f:
    pkl.dump(predMat, f)

t7 = time()
print('Save result: %.2fms' % ((t7 - t6) * 1000))
