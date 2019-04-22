from data_loader import DataLoader
from retrieval_systems import Hist, HOG, DL
from sklearn.manifold import TSNE
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
n = 20000
data_class = DataLoader(n=n)
data = data_class.X
labels = data_class.Y.reshape(-1)
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
plt.xlabel('First embedded dimension')
plt.ylabel('Second embedded dimension')
plt.title('t-SNE visualization of Random features')
figA.savefig('../plots/tsne_random', dpi=1000)

# Color histogram features
hist = Hist(data)
features = hist.database[:n, :]
tsne = TSNE()
embedded = tsne.fit_transform(features)
figB = plt.figure()
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s=3,
            cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('First embedded dimension')
plt.ylabel('Second embedded dimension')
plt.title('t-SNE visualization of Histogram features')
figB.savefig('../plots/tsne_hist', dpi=1000)

# Histogram of oriented gradients features
hog = HOG(data)
features = hog.database[:n, :]
tsne = TSNE()
embedded = tsne.fit_transform(features)
figC = plt.figure()
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s=3,
            cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('First embedded dimension')
plt.ylabel('Second embedded dimension')
plt.title('t-SNE visualization of HOG features')
figC.savefig('../plots/tsne_hog', dpi=1000)

# Deep learning features
dl = DL(data)
features = dl.database[:n, :]
tsne = TSNE()
embedded = tsne.fit_transform(features)
figD = plt.figure()
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, s=3,
            cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('First embedded dimension')
plt.ylabel('Second embedded dimension')
plt.title('t-SNE visualization of Deep Learning features')
figD.savefig('../plots/tsne_dl', dpi=1000)

end = time.time()
print(f"\nTime elapsed: {round((end-start)/60)} min.")
