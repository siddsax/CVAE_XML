import numpy as np
def precision_k(score_mat,true_mat,K):

    num_inst = score_mat.shape[1]
    num_lbl = score_mat.shape[0]

    P = np.zeros(K,1)
    rank_mat = np.argsort(score_mat)

    for k in range(K):
        mat = rank_mat
        mat[rank_mat[:, :-k]] = 0
        mat[np.argwhere(mat>0)] = 1
        mat = np.multiply(mat, true_mat)
        num = np.sum(mat,0)

        P[k] = np.mean(num/k)

    return P