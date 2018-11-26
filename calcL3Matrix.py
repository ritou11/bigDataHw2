from dataMatrix import dataMatrix
from scipy.sparse import csr_matrix
import pickle as pkl
from utils import csr_cosine_similarity
# from time import time

dtm = dataMatrix('./Project2-data/users.txt')
trainMatrix = dtm.getMatrixFromTxt('./Project2-data/netflix_train.txt', level=3)
simiMat = csr_cosine_similarity(trainMatrix)

with open('output/trainMatrixL3.pkl', 'wb') as f:
    pkl.dump(trainMatrix, f)
with open('output/simiMatL3.pkl', 'wb') as f:
    pkl.dump(simiMat, f)
