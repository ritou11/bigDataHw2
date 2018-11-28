from hwalg import uvPred
from utils import HwGlobal as hg
import pickle as pkl
import pandas as pd
import numpy as np

with open('output/testMatrix.pkl', 'rb') as f:
    testMatrix = pkl.load(f)
with open('output/trainMatrix.pkl', 'rb') as f:
    trainMatrix = pkl.load(f)
with open('output/trainMatrixL3.pkl', 'rb') as f:
    trainMatrixL3 = pkl.load(f)
with open('output/trainMatrixLM.pkl', 'rb') as f:
    trainMatrixLM = pkl.load(f)
with open('output/trainMean.pkl', 'rb') as f:
    trainMean = pkl.load(f)

A = (trainMatrix != 0).astype(int)
B = (testMatrix != 0).astype(int)
lmC = trainMean * B
l3C = B.multiply(3)
kList = [10, 30, 50, 70, 90]
lbdList = [1e-3, 1e-2, 1e-1, 1e1, 1e2]

res = pd.DataFrame(index=lbdList, columns=kList, dtype=float)
resList = dict()
resL3 = pd.DataFrame(index=lbdList, columns=kList, dtype=float)
resL3List = dict()
resLM = pd.DataFrame(index=lbdList, columns=kList, dtype=float)
resLMList = dict()
for k in kList:
    uinit = np.matrix(np.random.rand(hg.N,k)) * 1e-2
    vinit = np.matrix(np.random.rand(hg.N,k)) * 1e-2
    for lbd in lbdList:
        print('k = %s, lbd = %s, starting...' % (k, lbd))
        print('org...', end='')
        rmse, rmseList, jList = uvPred(trainMatrix, testMatrix, A, B, None, \
                    k, lbd, 1e-4, 1e3, \
                    uinit, vinit, 400)
        res.loc[lbd, k] = rmse
        resList[(lbd, k)] = (rmseList, jList)

        print('l3...', end='')
        rmse, rmseList, jList = uvPred(trainMatrixL3, testMatrix, A, B, l3C, \
                    k, lbd, 5e-4, 1e3, \
                    uinit, vinit, 400)
        resL3.loc[lbd, k] = rmse
        resL3List[(lbd, k)] = (rmseList, jList)

        print('lm...')
        rmse, rmseList, jList = uvPred(trainMatrixLM, testMatrix, A, B, lmC, \
                    k, lbd, 5e-4, 1e3, \
                    uinit, vinit, 400)
        resLM.loc[lbd, k] = rmse
        resLMList[(lbd, k)] = (rmseList, jList)
        break
    break

with open('output/paramRes.pkl', 'wb') as f:
    pkl.dump(res, f)
with open('output/paramResList.pkl', 'wb') as f:
    pkl.dump(resList, f)
with open('output/paramResL3.pkl', 'wb') as f:
    pkl.dump(resL3, f)
with open('output/paramResL3List.pkl', 'wb') as f:
    pkl.dump(resL3List, f)
with open('output/paramResLM.pkl', 'wb') as f:
    pkl.dump(resLM, f)
with open('output/paramResLMList.pkl', 'wb') as f:
    pkl.dump(resLMList, f)
