import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import hog
from keras.applications.vgg16 import VGG16, preprocess_input
import pickle


class Hist:
    def __init__(self, data):
        self.database = self.compress(data)

    @staticmethod
    def compress(data):
        try:
            with open('../data/Hist_features.pkl', mode='rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            bins = 170
            compressed = []
            for i, image in enumerate(data):
                red, _ = np.histogram(image[:, :, 0], bins=bins)
                green, _ = np.histogram(image[:, :, 1], bins=bins)
                blue, _ = np.histogram(image[:, :, 2], bins=bins)
                features = np.concatenate((red, blue, green))
                compressed.append(features)
            compressed = np.array(compressed)
            with open('../data/Hist_features.pkl', mode='wb') as file:
                pickle.dump(compressed, file, pickle.HIGHEST_PROTOCOL)
            return compressed

    def rank(self, query):
        query = self.compress(query)
        distances = cdist(query, self.database, metric='euclidean')
        return np.argsort(distances)


class HOG:
    def __init__(self, data):
        self.database = self.compress(data)

    @staticmethod
    def compress(data):
        try:
            with open('../data/HOG_features.pkl', mode='rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            compressed = []
            for i, image in enumerate(data):
                features = hog(image, orientations=8, pixels_per_cell=(6, 6),
                               cells_per_block=(4, 4), block_norm='L2-Hys',
                               multichannel=True)
                compressed.append(features)
            compressed = np.array(compressed)
            with open('../data/HOG_features.pkl', mode='wb') as file:
                pickle.dump(compressed, file, pickle.HIGHEST_PROTOCOL)
            return compressed

    def rank(self, query):
        query = self.compress(query)
        distances = cdist(query, self.database, metric='euclidean')
        return np.argsort(distances)


class DL:
    def __init__(self, data):
        self.model = VGG16(weights='imagenet', include_top=False)
        self.database = self.compress(data)

    def compress(self, data):
        try:
            with open('../data/DL_features.pkl', mode='rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            data = preprocess_input(data)
            features = self.model.predict(data)
            features = features.reshape(features.shape[0], features.shape[3])
            with open('../data/DL_features.pkl', mode='wb') as file:
                pickle.dump(features, file, pickle.HIGHEST_PROTOCOL)
            return features

    def rank(self, query):
        query = self.compress(query)
        distances = cdist(query, self.database, metric='euclidean')
        return np.argsort(distances)
