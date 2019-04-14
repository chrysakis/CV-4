import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import hog
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.decomposition import PCA
import pickle


class Hist:
    def __init__(self, data):
        self.database = self.compress(data)

    @classmethod
    def compress(cls, data):
        bins = 40
        compressed = []
        for i, image in enumerate(data):
            red = np.histogram(image[:, :, 0], bins=bins)
            blue = np.histogram(image[:, :, 1], bins=bins)
            green = np.histogram(image[:, :, 2], bins=bins)
            features = np.concatenate((red, blue, green))
            compressed.append(features)
        return compressed

    @classmethod
    def rank(cls, query):
        query = self.compress(query)
        distances = cdist(query, self.database, metric='euclidean')
        return np.argsort(distances)


class HOG:
    def __init__(self, data):
        self.database = self.compress(data)

    @classmethod
    def compress(cls, data):
        compressed = []
        for i, image in enumerate(data):
            features = hog(image, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(4, 4), block_norm='L2-Hys',
                           multichannel=True)
            compressed.append(features)
        return compressed

    @classmethod
    def rank(cls, query):
        query = self.compress(query)
        distances = cdist(query, self.database, metric='euclidean')
        return np.argsort(distances)


class DL:
    def __init__(self, data):
        self.database = self.compress(data)
        self.model = VGG16(weights='imagenet', include_top=False)

    @classmethod
    def compress(cls, data):
        try:
            with open('../data/DL_features.pkl', mode='rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            data = preprocess_input(data)
            data = self.model.predict(data)
            pca = PCA(n_components=128, whiten=True)
            data = pca.fit_transform(data)
            with open('../data/DL_features.pkl', mode='wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            return data

    @classmethod
    def rank(cls, query):
        query = self.compress(query)
        distances = cdist(query, self.database, metric='euclidean')
        return np.argsort(distances)
