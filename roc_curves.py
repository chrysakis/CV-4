from data_loader import DataLoader
from retrieval_systems import Hist, HOG, DL
import numpy as np
from sklearn.metrics import confusion_matrix
import time
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
n = 4000
points = 15
ranking = dict()
true_positives = {'hist': np.zeros(points), 'hog': np.zeros(points),
                  'dl': np.zeros(points)}
false_negatives = {'hist': np.zeros(points), 'hog': np.zeros(points),
                   'dl': np.zeros(points)}
for i in range(n):
    start = time.time()
    ranking['hist'] = hist.rank(hist.database[i, :].reshape(1, -1))[1:]
    ranking['hog'] = hog.rank(hog.database[i, :].reshape(1, -1))[1:]
    ranking['dl'] = dl.rank(dl.database[i, :].reshape(1, -1))[1:]
    linspace = np.linspace(1, dataset_size - 2, points, dtype=np.uint32)
    for j, threshold in enumerate(linspace):
        predictions = (np.ones(threshold),
                       np.zeros(dataset_size - threshold - 1))
        predictions = np.concatenate(predictions) == 1
        for method in ['hist', 'hog', 'dl']:
            truth = labels[ranking[method]] == labels[i]
            cm = confusion_matrix(truth, predictions, labels=(True, False))
            true_positives[method][j] += cm[0, 0] / 2000
            false_negatives[method][j] += cm[1, 0] / 18000
        # print(sum(truth), sum(predictions), cm[0, 0], cm[0, 1])
    end = time.time()
    print(f"Time per sample: {end - start:.3f} s")
for method in ['hist', 'hog', 'dl']:
    true_positives[method] = true_positives[method] / n
    false_negatives[method] = false_negatives[method] / n

# Plot the curves
fig = plt.figure()
for method in ['hist', 'hog', 'dl']:
    plt.plot(false_negatives[method], true_positives[method])
plt.xlabel('False Negatives')
plt.ylabel('True Positives')
plt.title(f"{points}-point FN-TP curve")
plt.legend(('Histogram', 'HOG', 'VGG16'))
plt.show()
