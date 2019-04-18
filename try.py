import numpy as np
from sklearn.metrics import precision_score, recall_score

labels = np.array([0, 2, 1, 2, 1, 0, 2, 1, 0])
predictions = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])

precision = precision_score(labels, predictions, average='micro')
recall = recall_score(labels, predictions, average='micro')

print(f"Precision:  {precision:.3f}\n"
      f"Recall:     {recall:.3f}\n")
