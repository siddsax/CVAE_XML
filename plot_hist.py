import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn import preprocessing

p = np.load('regen_data.npy')
g = np.load('../datasets/Eurlex/eurlex_docs/x_te.npy')
#g = np.load('../datasets/Eurlex/eurlex_docs/x_20.npy')
#p = np.load('regen_data_trn.npy')
x_for_pp = np.load('../datasets/Eurlex/eurlex_docs/x_tr.npy')

pp = preprocessing.MinMaxScaler()
scaler = pp.fit(x_for_pp)
g = scaler.transform(g)



ax1 = np.nonzero(l1)
ax2 = np.nonzero(l2)
ax = np.intersect1d(ax1, ax2)
ind = np.arange(ax.shape[-1])
fig, x = plt.subplots()
w = .1

l1 = p[0, ax]
l2 = g[0, ax]

p1 = x.bar(ind, l1, w, color='r', bottom=0)
p1 = x.bar(ind+w, l2, w, color='y', bottom=0)



x.set_xticks(ind + width / 2)
x.autoscale_view()
plt.show()

