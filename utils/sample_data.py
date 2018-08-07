import numpy as np
from scipy import sparse
import time
x  = np.load('x_tr.npy')
ty = np.load('y_tr.npy')
ty_cp = np.copy(ty) 
ty = np.copy(ty_cp)
labels_num = np.sum(ty,axis=0)
removed_indices = []
N = ty.shape[0]
nf = .2*N
n_now = N
labels_num = np.sum(ty,axis=0)
not_zero = np.argwhere(labels_num!=0)
def index_rows_by_exclusion_nptake(arr, i):
    """
    Return copy of arr excluding single row of position i using
    numpy.take function
    """
    return arr.take(range(i)+range(i+1,arr.shape[0]), axis=0)

# indices = np.random.randint(0, high=N, size=N)
while (ty.shape[0] > nf):

    candidate = labels_num - ty[0]
    not_zero = np.argwhere(labels_num!=0)
    if(len(np.argwhere(candidate[not_zero]==0))==0):
        print("="*10 + " "*5 + str(ty.shape[0]) + " "*5 + "="*10)
	labels_num = candidate
    	ty = ty[1:]
	x = x[1:]
	print('-'*10)
    else:
	ty = np.concatenate((ty[1:], ty[0].reshape((1, ty[0].shape[0]))), axis=0)
	x = np.concatenate((x[1:], x[0].reshape((1, x[0].shape[0]))), axis=0)
	print("Num labels offended: " +str(len(np.argwhere(candidate[not_zero]==0))))
print(x.shape)
print(ty.shape)
np.save('x_20', x)
np.save('y_20', ty)
