from dataMatrix import dataMatrix
from utils import HwGlobal as hg
from utils import csr_cosine_similarity
import pickle as pkl
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix
import numpy as np

def simiPred(trainMatrix, testMatrix, A=None, B=None, C=None):
    if A is None:
        A = (trainMatrix != 0).astype(int)
    if B is None:
        B = (testMatrix != 0).astype(int)
    if C is None:
        C = csr_matrix(([0],([0],[0])),shape=(hg.N,hg.N))
    simiMat = csr_cosine_similarity(trainMatrix)
    upper = simiMat * trainMatrix
    lower = simiMat * A
    predMat = upper.multiply(lower.power(-1))
    n = testMatrix.count_nonzero()
    maskPredMat = predMat.multiply(B)
    rmse = norm(maskPredMat + C - testMatrix) / np.sqrt(n)
    return rmse, predMat

def uvPred(trainMatrix, testMatrix, A=None, B=None, C=None,\
            k=50, lbd=0.01, alpha=1e-4, eps=1e3, \
            uinit=1e-2, vinit=1e-2, MAX_ITER=400):
    if type(uinit) is float:
        uinit = np.matrix(np.random.rand(N,k)) * uinit
    if type(vinit) is float:
        vinit = np.matrix(np.random.rand(N,k)) * vinit
    if A is None:
        A = (trainMatrix != 0).astype(int)
    if B is None:
        B = (testMatrix != 0).astype(int)
    if C is None:
        C = csr_matrix(([0],([0],[0])),shape=(hg.N,hg.N))
    n = testMatrix.count_nonzero()
    U = uinit
    V = vinit
    rmseList = list()
    jList = list()
    UVT = U * V.T
    AUVT = A.multiply(UVT)
    AUVTX = AUVT - trainMatrix
    rmse = norm(B.multiply(UVT) - testMatrix) / np.sqrt(n)
    J = 0.5 * norm(AUVTX)**2 + lbd * np.linalg.norm(U)**2 + lbd * np.linalg.norm(V)**2
    jList.append(J)
    rmseList.append(rmse)
    for i in range(MAX_ITER):
        pJU = AUVTX * V + 2 * lbd * U
        pJV = AUVTX.T * U + 2 * lbd * V
        U = U - alpha * pJU
        V = V - alpha * pJV
        UVT = U * V.T
        AUVT = A.multiply(UVT)
        AUVTX = AUVT - trainMatrix
        rmse = norm(B.multiply(UVT) + C - testMatrix) / np.sqrt(n)
        JL = J
        J = 0.5 * norm(AUVTX)**2 + lbd * np.linalg.norm(U)**2 + lbd * np.linalg.norm(V)**2
        jList.append(J)
        rmseList.append(rmse)
        if JL - J < eps:
            break
    return rmse, rmseList, jList
