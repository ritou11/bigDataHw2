import pickle as pkl
from time import time
from utils import csr_cosine_similarity
from utils import HwGlobal as hg

with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f)
t1 = time()
simiMat = csr_cosine_similarity(trainMatrix)
t2 = time()
with open('output/simiMat.pkl', 'wb') as f:
    pkl.dump(simiMat, f)
t3 = time()
print('Get similarity: %.2fms' % ((t2 - t1)*1000))
print('Put similarity: %.2fms' % ((t3 - t2)*1000))
