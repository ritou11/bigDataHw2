from dataMatrix import dataMatrix
from utils import HwGlobal as hg
from scipy.sparse import csr_matrix
import pickle as pkl
from time import time

t1 = time()
dtm = dataMatrix('./Project2-data/users.txt')
testMatrix = dtm.getMatrixFromTxt('./Project2-data/netflix_test.txt')
trainMatrix = dtm.getMatrixFromTxt('./Project2-data/netflix_train.txt')
t2 = time()
with open('output/testMatrix.pkl', 'wb') as f:
    pkl.dump(testMatrix, f)
with open('output/trainMatrix.pkl', 'wb') as f:
    pkl.dump(trainMatrix, f)
t3 = time()

with open('output/exp1.log', 'w') as resFile:
    print('Get mats: %.2fms' % ((t2 - t1)*1000), file=resFile)
    print('Put mats: %.2fms' % ((t3 - t2)*1000), file=resFile)
    print('trainMatrix nnz = %d, %.2f%%' % (trainMatrix.nnz, trainMatrix.nnz / hg.N**2 * 100))
    print('testMatrix nnz = %d, %.2f%%' % (testMatrix.nnz, testMatrix.nnz / hg.N**2 * 100))
