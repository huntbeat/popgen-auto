import numpy as np
# turns off plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# turns off plotting
plt.ioff()

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

FILENAME = 'statsZI/example/stats/stats_1.txt'
stats_name = []
stats = []
total_sim = 0

with open(FILENAME, "r+") as stat_inputfile:

    stats_names = next(stat_inputfile).split(" ")

    for line in stat_inputfile:
        stats.append(list(map(float,line.split(" "))))
        total_sim += 1

types = 4

X = np.array(stats)
y = np.array(list(range(types)))
y = np.repeat(y, int(total_sim/types))

#"""
#feature scaling, TESTING
#"""
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

pca = PCA(n_components=2, whiten=False)
X_r = pca.fit_transform(X)

plt.figure()

colors = ['turquoise', 'blue', 'red', 'green']
labels = ['0', '10', '100', '1000']
lw = 2

for index, color, label in zip(list(range(types)), colors, labels):
    plt.scatter(X_r[y == index, 0], X_r[y == index, 1], color=color, alpha=.8, lw=lw,
                    label=label)

plt.legend()

plt.savefig(FILENAME.split("/")[-1].replace('.txt','.png'))
plt.show()

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
