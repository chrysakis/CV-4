from data_loader import DataLoader
from retrieval_systems import Hist, HOG, DL
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Initialize data
n = 100
data_class = DataLoader()
x = data_class.X
labels = data_class.Y_train

# Initialize databases
hist = Hist(x)
hog = HOG(x)
dl = DL(x)

# Compute precision-recall scores (average over the first 1000 samples of the
# training set for all retrieval systems
ranking = dict()
precision = {'hist': np.zeros(100), 'hog': np.zeros(100), 'dl': np.zeros(100)}
recall = {'hist': np.zeros(100), 'hog': np.zeros(100), 'dl': np.zeros(100)}
for i in range(n):
    print(i)
    ranking['hist'] = hist.rank(hist.database[i, :].reshape(1, -1))
    ranking['hog'] = hog.rank(hog.database[i, :].reshape(1, -1))
    ranking['dl'] = dl.rank(dl.database[i, :].reshape(1, -1))
    for threshold in range(100):
        for method in ['hist', 'hog', 'dl']:
            retrieved = labels[ranking[method]]
            predictions = np.concatenate((np.ones(threshold),
                                          np.zeros(50000 - n - threshold)))
            precision[method][j] += precision_score(retrieved, predictions,
                                                    average='micro')
            recall[method][j] += recall_score(retrieved, predictions,
                                              average='micro')
for method in ['hist', 'hog', 'dl']:
    precision[method] = precision[method] / n
    recall[method] = recall[method] / n

# Plot the curves
