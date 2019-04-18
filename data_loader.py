from keras.datasets import cifar10
import numpy as np


class DataLoader:
    def __init__(self, n=100):
        (self.X, self.Y), (self.Xo, self.Yo) = cifar10.load_data()
        np.random.seed(200)
        (self.X_train, self.X_val, self.Y_train, self.Y_val) = \
            self.validation_split(self.X, self.Y, n=n)

    @staticmethod
    def validation_split(x, y, n=100):
        return x[:-n], x[-n:], y[:-n], y[-n:]
        train_indexes = set(range(len(x)))
        val_indexes = set()
        count = [0] * 10
        while True:
            index = np.random.choice(list(train_indexes))
            if count[y[index, 0]] < n:
                val_indexes.add(index)
                train_indexes.remove(index)
                count[y[index, 0]] += 1
            if count == [n] * 10:
                break
        x_train = np.zeros((len(train_indexes), 32, 32, 3), dtype=np.uint8)
        x_val = np.zeros((len(val_indexes), 32, 32, 3), dtype=np.uint8)
        y_train = np.zeros(len(train_indexes), dtype=np.uint8)
        y_val = np.zeros(len(val_indexes), dtype=np.uint8)
        for i, j in enumerate(train_indexes):
            x_train[i, :, :, :] = x[j, :, :, :]
            y_train[i] = y[j]
        for i, j in enumerate(val_indexes):
            x_val[i, :, :, :] = x[j, :, :, :]
            y_val[i] = y[j]
        return x_train, x_val, y_train, y_val
