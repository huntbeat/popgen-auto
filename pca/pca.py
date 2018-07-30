import numpy as np
# turns off plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# turns off plotting
plt.ioff()

from sklearn.decomposition import PCA
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

import pdb; pdb.set_trace()

pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

import pdb; pdb.set_trace()

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')

plt.savefig('pca_iris_example.png')
plt.show()
