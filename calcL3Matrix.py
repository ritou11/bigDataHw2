from dataMatrix import dataMatrix
from scipy.sparse import csr_matrix
import pickle as pkl
# from time import time

dtm = dataMatrix('./Project2-data/users.txt')
trainMatrix = dtm.getMatrixFromTxt('./Project2-data/netflix_train.txt', level=3)
testMatrix = dtm.getMatrixFromTxt('./Project2-data/netflix_test.txt', level=3)
with open('output/testMatrix.pkl', 'rb') as f:
    testMatrixOrg = pkl.load(f)
testMaskMat = (testMatrixOrg != 0).astype(int)

with open('output/trainMatrixL3.pkl', 'wb') as f:
    pkl.dump(trainMatrix, f)
with open('output/testMatrixL3.pkl', 'wb') as f:
    pkl.dump(testMatrix, f)
with open('output/testMaskMatL3.pkl', 'wb') as f:
    pkl.dump(testMaskMat, f)
