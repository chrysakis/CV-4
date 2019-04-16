import time
from data_loader import DataLoader
from retrieval_systems import Hist, HOG, DL

start = time.time()
data_class = DataLoader()
data = data_class.X_train
end = time.time()
print(f"Time elapsed: {end-start}s")

start = time.time()
systemA = Hist(data)
print(systemA.database.shape)
end = time.time()
print(f"Time elapsed: {end-start}s")

start = time.time()
systemB = HOG(data)
print(systemB.database.shape)
end = time.time()
print(f"Time elapsed: {end-start}s")

start = time.time()
systemC = DL(data)
print(systemC.database.shape)
end = time.time()
print(f"Time elapsed: {end-start}s")
