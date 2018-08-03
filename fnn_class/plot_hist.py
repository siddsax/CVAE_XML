import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=int, default=0, help='0 train, 1 test')
parser.add_argument('--index', type=int, default=0, help='layer over labels')
params = parser.parse_args()

if(params.index):
	p = np.load('regen_data.npy')
	g = np.load('../datasets/Eurlex/eurlex_docs/x_te.npy')
else:
	g = np.load('../datasets/Eurlex/eurlex_docs/x_20.npy')
	p = np.load('regen_data_trn.npy')

x_for_pp = np.load('../datasets/Eurlex/eurlex_docs/x_tr.npy')

pp = preprocessing.MinMaxScaler()
scaler = pp.fit(x_for_pp)
g = scaler.transform(g)



ax1 = np.nonzero(p[params.index])
ax2 = np.nonzero(g[params.index])
ax = np.intersect1d(ax1, ax2)

ind = np.arange(ax.shape[-1])
fig, x = plt.subplots()
w = .1

l1 = p[params.index, ax]
l2 = g[params.index, ax]

l1 = l1[-50:]
l2 = l2[-50:]
ind = ind[-50:]

p1 = x.bar(ind, l1, w, color='r', bottom=0)
p2 = x.bar(ind+w, l2, w, color='y', bottom=0)



x.set_xticks(ind)
x.autoscale_view()
plt.show()

