from keras.datasets import cifar10


class DataLoader:
    def __init__(self, n):
        (self.X, self.Y), (self.Xo, self.Yo) = cifar10.load_data()
        self.X, self.Y = self.X[:n], self.Y[:n]

    @staticmethod
    def validation_split(x, y, n=100):
        pass
