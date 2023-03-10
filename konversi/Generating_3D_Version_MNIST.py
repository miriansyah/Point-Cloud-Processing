import os
# turning off annoying tensorflow verbose logging messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.datasets import mnist
import os, sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import random
import math
import cv2

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import h5py

# new_size=16
# zero_min = np.zeros((28, 28))
# one_max = np.ones((28, 28))


# def gaussian_noise(X, sigma):
#     ''' adds a gaussian noise limited to 0 and 1 inclusive'''
#     X_nonzero_indexes = np.nonzero(X)
#     noise = np.random.normal(0, sigma, X.shape)
#     copy = X.copy()
#     copy[X_nonzero_indexes] = np.minimum(np.maximum(X[X_nonzero_indexes] + 
#         noise[X_nonzero_indexes], zero_min[X_nonzero_indexes]), one_max[X_nonzero_indexes])
#     return copy
# ########################################################################################################
# color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
#                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu']

# def convert_to_rgb(gray_image, color_map):
#     '''Convert gray image to RGB using the given color map'''
#     s_m = pyplot.cm.ScalarMappable(cmap = color_map)
#     img_shape = gray_image.shape
#     flattened = gray_image.flatten()
#     colors = s_m.to_rgba(flattened)
#     result = np.zeros(flattened.shape + (3,))

#     for i in range(len(flattened)):
#         if flattened[i] > 0:
#             result[i] = colors[i][:-1]
#     return result.reshape(img_shape + (3,))
# ###########################################################################################################
# def create_colored_grid(image, depth, color_map):
#     grid = np.zeros(image.shape + (depth, 3,))
#     for z in range(depth):
#         instance_points = gaussian_noise(image, 0.2)
#         rgb_points = convert_to_rgb(instance_points, color_map)
#         grid[:, :, z] = rgb_points
#     return grid
# ############################################################################################################
# def transform_instance(instance):
#     resized = cv2.resize(instance, dsize=(new_size, new_size), interpolation=cv2.INTER_CUBIC)
#     grid = create_colored_grid(resized / 255.0, new_size, random.choice(color_maps))
#     return grid

(train_x, train_y), (test_x, test_y) = mnist.load_data()
# #print(np.array(train_x).shape)
# train_x_3d = np.zeros((len(train_x), new_size, new_size, new_size, 3))
# test_x_3d = np.zeros((len(test_x), new_size, new_size, new_size, 3))
        
# for i in range(len(train_x)):
#     train_x_3d[i] = transform_instance(train_x[i])

# for i in range(len(test_x)):
#     test_x_3d[i] = transform_instance(test_x[i])

# print(train_x_3d.shape)
# print(test_x_3d.shape)

# output_file = h5py.File('3d-mnist-colour.h5', 'w')

# output_file.create_dataset('train_x', data=train_x_3d)
# output_file.create_dataset('train_y', data=train_y)
# output_file.create_dataset('test_x', data=test_x_3d)
# output_file.create_dataset('test_y', data=test_y)

# output_file.close()

train_x = np.array(train_x)
instance = train_x[2]
print(train_x.shape)
#print(train_y)
pyplot.imshow(instance, cmap=pyplot.get_cmap('gray'))
pyplot.show()

grid = np.zeros((28, 28, 28))

score = instance / 255.0


zero_min = np.zeros((28, 28))
one_max = np.ones((28, 28))

def gaussian_noise(X, sigma):
    ''' adds a gaussian noise limited to 0 and 1 inclusive'''
    X_nonzero_indexes = np.nonzero(X)
    noise = np.random.normal(0, sigma, X.shape)
    copy = X.copy()
    copy[X_nonzero_indexes] = np.minimum(np.maximum(X[X_nonzero_indexes] + 
        noise[X_nonzero_indexes], zero_min[X_nonzero_indexes]), one_max[X_nonzero_indexes])
    return copy

# generating a 3D instance from a 2D image
for z in range(28):
    instance_points = gaussian_noise(score, 0.2)
    grid[:, :, z] = score

print(grid.shape)
print(grid)