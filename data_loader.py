from keras.datasets import cifar10
import numpy as np


class Data:
    def __init__(self, n=100):
        (self.X, self.Y), (self.Xo, self.Yo) = cifar10.load_data()
        np.random.seed(200)
        (self.X_train, self.X_val) = self.validation_split(self.X, self.Y,
                                                           max_count=n)

    @classmethod
    def validation_split(cls, X, Y, max_count=100):
        train_indexes = set(range(len(X)))
        val_indexes = set()
        count = [0] * 10
        while True:
            index = np.random.choice(list(train_indexes))
            if count[Y[index, 0]] < max_count:
                val_indexes.add(index)
                train_indexes.remove(index)
                count[Y[index, 0]] += 1
            if count == [max_count] * 10:
                break
        X_train = np.zeros((len(train_indexes), 32, 32, 3), dtype=np.uint8)
        X_val = np.zeros((len(val_indexes), 32, 32, 3), dtype=np.uint8)
        for i, j in enumerate(train_indexes):
            X_train[i, :, :, :] = X[j, :, :, :]
        for i, j in enumerate(val_indexes):
            X_val[i, :, :, :] = X[j, :, :, :]
        return X_train, X_val
