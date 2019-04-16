from keras.datasets import cifar10
import matplotlib.pyplot as plt
from skimage.feature import hog

(x, y), (_, _) = cifar10.load_data()

image = x[0]
fig = plt.figure()
plt.imshow(image)
features = hog(image, orientations=8, pixels_per_cell=(6, 6),
               cells_per_block=(4, 4), block_norm='L2-Hys', multichannel=True)
print(features.shape)

