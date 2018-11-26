import numpy as np

class HwGlobal():
    N = 10000

def csr_cosine_similarity(input_csr_matrix):
    similarity = input_csr_matrix * input_csr_matrix.T
    square_mag = similarity.diagonal()
    zeros = np.where(square_mag == 0)
    square_mag[zeros] = 1
    similarity[zeros, zeros] = 1
    inv_square_mag = 1 / square_mag
    inv_mag = np.sqrt(inv_square_mag)
    return similarity.multiply(inv_mag).T.multiply(inv_mag).tocsr()
