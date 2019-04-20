from data_loader import DataLoader
from retrieval_systems import DL
import numpy as np
import matplotlib.pyplot as plt

# Initialize data
dataset_size = 20000
data_class = DataLoader(n=dataset_size)
x = data_class.X
labels = data_class.Y.reshape(-1)
np.random.seed(33)
test_data = data_class.Xo
test_indexes = np.random.choice(range(test_data.shape[0]), 10)

# Initialize database and retrieve matches
method = DL(x)
for index in test_indexes:
    print(index)
    query = test_data[index:index+1]
    retrieved = method.rank(method.compress_test(query))
    top_4_indexes = retrieved[:4]
    bottom_4_indexes = retrieved[-4:]
    fig = plt.figure()
    plt.subplot(3, 4, 1)
    plt.imshow(query[0])
    plt.title('Query image')
    for i in range(4):
        plt.subplot(3, 4, 5+i)
        plt.imshow(x[top_4_indexes[i]])
        if i == 0:
            plt.ylabel('Best 4 matches')
        plt.subplot(3, 4, 9+i)
        plt.imshow(x[bottom_4_indexes[i]])
        if i == 0:
            plt.ylabel('Worst 4 matches')
    plt.show()
