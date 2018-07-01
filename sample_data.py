import numpy as np
from scipy import sparse

x  = np.load('x_train.npy')
ty = sparse.load_npz('y_train.npz').todense()
ty_cp = np.copy(ty) 
ty = np.copy(ty_cp)
labels_num = np.sum(ty,axis=0)
removed_indices = []
count = 0
N = ty.shape[0]
nf = .1*N
n_now = N
labels_num = np.sum(ty,axis=0)

# indices = np.random.randint(0, high=N, size=N)
while (ty.shape[0] > nf):
    count = np.random.randint(0, high=ty.shape[0])             
    candidate = labels_num - ty[count]
    print(count)
    if(len(np.argwhere(candidate==0))==0):
        print("="*10 + " "*5 + str(ty.shape[0]) + " "*5 + "="*10)
        labels_num = candidate
        ty = np.delete(ty, count, axis=0)
        x = np.delete(x, count, axis=0)
        removed_indices.append(count)

print(x.shape)
print(ty.shape)
np.save('x_20', x)
np.save('y_20', ty)
