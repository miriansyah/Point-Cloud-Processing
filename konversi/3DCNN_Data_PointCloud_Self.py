import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from matplotlib import style
from matplotlib import animation
from sklearn.preprocessing import MinMaxScaler
import glob
import h5py

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from keras.utils import to_categorical
import tensorflow as tf

from tensorflow import keras


import os, sys
np.set_printoptions(threshold=sys.maxsize)

from VoxelGrid import VoxelGrid

def normalize_pc_range(pcd_np):
    scaler = MinMaxScaler()
    scaler.fit(pcd_np)
    pcd_np = scaler.transform(pcd_np)
    return pcd_np

def count_plot(array):
    cm = plt.cm.get_cmap('gist_rainbow')
    n, bins, patches = plt.hist(array, bins=64)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    plt.show()


#LOAD PCD
#pcd = o3d.io.read_point_cloud("D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/crop_of_tangan_samping.pcd")
pcd = o3d.io.read_point_cloud("D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/convert_kitti_to_pcd/berjalan0_crop.pcd")
R = pcd.get_rotation_matrix_from_axis_angle([0,0,1.571])
pcd = pcd.rotate(R, center=(0,0,0))
R = pcd.get_rotation_matrix_from_axis_angle([-1.571,0,0])
pcd = pcd.rotate(R, center=(0,0,0))
pcd_np= np.asarray(pcd.points)
pcd_np= normalize_pc_range(pcd_np)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_title ("Normalize")
ax.scatter(pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2],color='g', alpha=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#print(pcd_np.shape)

#VOXELIZATION
voxel_grid = VoxelGrid(pcd_np, x_y_z = [16, 16, 16])

print(voxel_grid.shape)
#Plotting VOXEL
plt.title("VOXEL from PCD data")
plt.xlabel("VOXEL")
plt.ylabel("POINTS INSIDE THE VOXEL")

#print(voxel_grid.structure)
count_plot(voxel_grid.structure[:,3])
# vector = np.zeros(4096)
# data_voxel=np.bincount(voxel_grid.structure[:,3])
# print(data_voxel)
# vector[:len(data_voxel)] = data_voxel
# vector = np.where(vector > 0.0, 1, 0)
# print(vector)

voxel_2d = np.array(voxel_grid.vector[:,:,:]) 
voxel_2d = voxel_2d.reshape(-1)
voxel_2d = np.where(voxel_2d > 0.0, 1, 0)
voxel_2d = voxel_2d.astype('float64')

voxel_final= np.array(voxel_2d.reshape(16,16,16))

#print(voxel_final[:,:,15])
# for i in range(voxel_final.shape[0]):
#         print(voxel_final[:,:, i].shape)

train_loader = tf.data.Dataset.from_tensor_slices((voxel_final))

# for ele in train_loader:
#     print(ele.numpy())

data = train_loader.take(16)
data_array = np.array(list(data))
#print(data_array.shape)
#print(data_array[0])


fig = plt.figure(figsize=(20,20))
for j in range(16):
    plt.subplot(2,8,j+1).set_title(f'Channel {j+1}')
    plt.imshow(data_array[j], cmap="binary")
plt.show()


ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxel_final[:,:,:], facecolors='g', edgecolor='k')

plt.show()


