from utils import HwGlobal as hg
from scipy.sparse import csr_matrix

class dataMatrix():
    def __init__(self, filename = ''):
        self.user_id = dict()
        if filename:
            self.readUserid(filename)

    def readUserid(self, filename):
        with open(filename, 'r') as f:
            for i, l in enumerate(f):
                self.user_id[int(l)] = i

    # user index = row_number of user_id (from 0)
    # movie index = movie_id - 1 (from 0)
    def getMatrixFromTxt(self, filename, level=0):
        uid = list()
        mid = list()
        sc = list()
        with open(filename, 'r') as f:
            for i, l in enumerate(f):
                dt = l.split()
                score = int(dt[2]) - level
                if score != 0:
                    uid.append(self.user_id[int(dt[0])])
                    mid.append(int(dt[1]) - 1)
                    sc.append(score)
        return csr_matrix((sc, (uid, mid)), shape=(hg.N, hg.N))
