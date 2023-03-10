import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import os
import glob
import pykitti
import mayavi.mlab
import open3d as o3d

path = 'D:/Disertasi/Oprek/DataSet_KITTI/2011_09_28/2011_09_28/2011_09_28_drive_0053_sync'

file_velo = sorted(glob.glob(os.path.join(path, 'velodyne_points','data', '*.bin')))

def load_data_velo(file):
    data_velo = np.fromfile(file, dtype=np.float32)
    return data_velo.reshape((-1, 4))

lidar = load_data_velo(file_velo[0])
lidar = np.array([lidar[:,0],lidar[:,1],lidar[:,2]]).T
#print(lidar.shape)

with open('D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/convert_kitti_to_pcd/berjalan0.txt','wb') as f:
    np.savetxt(f,lidar)

pcd = o3d.io.read_point_cloud("D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/convert_kitti_to_pcd/berjalan0.txt", format='xyz')
o3d.io.write_point_cloud("D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/convert_kitti_to_pcd/berjalan0.pcd", pcd)