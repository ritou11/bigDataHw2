from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import pickle as pkl
from time import time
from utils import csr_cosine_similarity
from utils import HwGlobal as hg
import numpy as np
resFile = open('output/exp2_org.log', 'w')

t1 = time()

with open('output/testMatrix.pkl', 'rb') as f:
    testMatrix = pkl.load(f).tocoo()
with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f).tocsc()
with open('output/simiMat.pkl', 'rb') as f:
    simiMat = pkl.load(f)

def predict(i, j):
    # tt1 = time()
    traincol = trainMatrix.getcol(j)
    upper = simiMat.getrow(i).dot(traincol)
    # tt2 = time()
    # 优化效果不明显，猜测原因为Python本身的缓存作用
    lower = simiMat.getrow(i)[0, traincol.nonzero()[0]].sum()
    # tt3 = time()
    # print('up = %.2fms' % ((tt2 - tt1) * 1000))
    # print('lw = %.2fms' % ((tt3 - tt2) * 1000))
    return upper[0,0] / lower

t2 = time()
print('Load data: %.2fms' % ((t2 - t1) * 1000), file=resFile)

rmse = 0
predict_data = list()
n = testMatrix.count_nonzero()
alt = n // 100
cnt = 0
for (i,j,v) in zip(testMatrix.row, testMatrix.col, testMatrix.data):
    p = predict(i, j)
    cnt += 1
    rmse += (p - v) ** 2
    predict_data.append(p)
    if cnt % alt == 0:
        print('Process %.2f%%' % (cnt / n * 100), file=resFile)
n = testMatrix.count_nonzero()
rmse = np.sqrt(rmse / n)

t3 = time()
print('Predict test: %.2fms' % ((t3 - t2) * 1000), file=resFile)
print('RMSE = %.3f' % rmse, file=resFile)

resFile.close()
