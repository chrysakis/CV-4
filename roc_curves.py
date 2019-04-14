from data_loader import Data
from retrieval_systems import Hist, HOG, DL

data = Data()
print('Data loaded')
hist = Hist(data.X)
print('Hist initialized')
hog = HOG(data.X)
print('HOG initialized')
