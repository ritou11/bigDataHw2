import pickle as pkl
from time import time
import numpy as np

with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f)
t1 = time()
with open('output/trainMatrixT.pkl', 'wb') as f:
    pkl.dump(trainMatrix.T, f)
t2 = time()
print('Get/Put mats: %.2fms' % ((t2 - t1)*1000))
