from data_loader import DataLoader
from retrieval_systems import Hist, HOG, DL
from sklearn.manifold import TSNE
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
n = 15000
data_class = DataLoader()
data = data_class.X_train
labels = data_class.Y_train[:n]
hist, _ = np.histogram(labels)
print(hist)
colors = ['blue', 'red', 'yellow', 'magenta', 'green', 'orange', 'gray',
          'cyan', 'purple', 'black']

# Random features
features = np.random.randn(n, 512)
tsne = TSNE()
embedded = tsne.fit_transform(features)
figA = plt.figure()
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s=3,
            cmap=matplotlib.colors.ListedColormap(colors))
figA.savefig('../plots/tsne_random', dpi=1000)

# Color histogram features
hist = Hist(data)
features = hist.database[:n, :]
tsne = TSNE()
embedded = tsne.fit_transform(features)
figB = plt.figure()
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s=3,
            cmap=matplotlib.colors.ListedColormap(colors))
figB.savefig('../plots/tsne_hist', dpi=1000)

# Histogram of oriented gradients features
hog = HOG(data)
features = hog.database[:n, :]
tsne = TSNE()
embedded = tsne.fit_transform(features)
figC = plt.figure()
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s=3,
            cmap=matplotlib.colors.ListedColormap(colors))
figC.savefig('../plots/tsne_hog', dpi=1000)

# Deep learning features
dl = DL(data)
features = dl.database[:n, :]
tsne = TSNE()
embedded = tsne.fit_transform(features)
figD = plt.figure()
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s=3,
            cmap=matplotlib.colors.ListedColormap(colors))
figD.savefig('../plots/tsne_dl', dpi=1000)

end = time.time()
print(f"\nTime elapsed: {round((end-start)/60)} min.")
