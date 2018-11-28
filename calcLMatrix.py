from dataMatrix import dataMatrix
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import pickle as pkl
from utils import csr_cosine_similarity
from utils import HwGlobal as hg
# from time import time

dtm = dataMatrix('./Project2-data/users.txt')
trainMatrixL3 = dtm.getMatrixFromTxt('./Project2-data/netflix_train.txt', level=3)
simiMat = csr_cosine_similarity(trainMatrixL3)

with open('output/trainMatrixL3.pkl', 'wb') as f:
    pkl.dump(trainMatrixL3, f)
with open('output/simiMatL3.pkl', 'wb') as f:
    pkl.dump(simiMat, f)

with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f)
mask = (trainMatrix != 0).astype(int)
trainRowMeans = list()
for i in range(hg.N):
    trainRowMeans.append(trainMatrix.getrow(i).sum() / trainMatrix.getrow(i).nnz)
trainMean = diags(trainRowMeans, shape=(hg.N, hg.N))
trainMatrixLM = trainMatrix - trainMean * mask
simiMat = csr_cosine_similarity(trainMatrixLM)

with open('output/trainMatrixLM.pkl', 'wb') as f:
    pkl.dump(trainMatrixLM, f)
with open('output/trainMean.pkl', 'wb') as f:
    pkl.dump(trainMean, f)
with open('output/simiMatL3.pkl', 'wb') as f:
    pkl.dump(simiMat, f)
