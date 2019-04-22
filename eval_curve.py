from data_loader import DataLoader
from retrieval_systems import Hist, HOG, DL
import numpy as np
import matplotlib.pyplot as plt

# Initialize data
dataset_size = 20000
data_class = DataLoader(n=dataset_size)
x = data_class.X
labels = data_class.Y.reshape(-1)

# Initialize databases
hist = Hist(x)
hog = HOG(x)
dl = DL(x)

# Compute precision-recall scores (average over the first 1000 samples of the
# training set for all retrieval systems
n = 2000
points = 15
ranking = dict()
precision = {'hist': np.zeros(points), 'hog': np.zeros(points),
             'dl': np.zeros(points)}
recall = {'hist': np.zeros(points), 'hog': np.zeros(points),
          'dl': np.zeros(points)}
for i in range(n):
    ranking['hist'] = hist.rank(hist.database[i, :].reshape(1, -1))[1:]
    ranking['hog'] = hog.rank(hog.database[i, :].reshape(1, -1))[1:]
    ranking['dl'] = dl.rank(dl.database[i, :].reshape(1, -1))[1:]
    linspace = np.linspace(1, 2000, points, dtype=np.uint32)
    for j, threshold in enumerate(linspace):
        predictions = (np.ones(threshold),
                       np.zeros(dataset_size - threshold - 1))
        predictions = np.concatenate(predictions) == 1
        for method in ['hist', 'hog', 'dl']:
            truth = labels[ranking[method]] == labels[i]
            true_positives = np.sum(np.bitwise_and(truth, predictions))
            precision[method][j] += true_positives / threshold
            recall[method][j] += true_positives / 2000
for method in ['hist', 'hog', 'dl']:
    precision[method] = precision[method] / n
    recall[method] = recall[method] / n

# Plot the curves
fig = plt.figure()
for method in ['hist', 'hog', 'dl']:
    plt.plot(recall[method], precision[method])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f"{points}-point precision-recall curve")
plt.legend(('Histogram', 'HOG', 'VGG16'))
fig.savefig('../plots/pr_curve', dpi=1000)
