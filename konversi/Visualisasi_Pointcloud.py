import mayavi.mlab
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

cubic_size = 1
voxel_resolution =16

def normalize_pc(points):
	centroid = np.mean(points, axis=0)
	points = points - centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance

	return points

def normalize_pc_range(pcd_np):
    scaler = MinMaxScaler()
    scaler.fit(pcd_np)
    pcd_np = scaler.transform(pcd_np)
    return pcd_np

def normalize_voxel(voxel_point):
    scaler = MinMaxScaler((1,16))
    scaler.fit(voxel_point)
    voxel_point = scaler.transform(voxel_point)
    return voxel_point

def pointcloud_to_voxel(pcd,cubic_size=0.5,voxel_resolution=10):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd_norm)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd_,
                voxel_size=cubic_size / (voxel_resolution),
                min_bound=(-cubic_size , -cubic_size , -cubic_size ),
                max_bound=(cubic_size , cubic_size , cubic_size ))
    voxels = voxel_grid.get_voxels()
    indices = np.stack(list(vx.grid_index for vx in voxels))
    colors = np.stack(list(vx.color for vx in voxels))
    return indices,colors

###################### MAIN PROGRAM #################################################################################
pcd = o3d.io.read_point_cloud("D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/convert_kitti_to_pcd/berjalan0.pcd")
pcd_np= np.asarray(pcd.points)
pcd_n=pcd_np

pcd_norm = normalize_pc_range(pcd_np)
indices, colors = pointcloud_to_voxel(pcd_norm,1,16)
indices = normalize_voxel (indices)

####################### VISUALITATION POINTCLOUD ###################################################################

#o3d.visualization.draw_geometries([pcd])

####################### VISUALITATION VOXELIZATION ##################################################################

o3d.visualization.draw_geometries([pcd])

####################### VISUALIZATION MATPLOTLIB ####################################################################
fig = plt.figure()
fig.suptitle ("Comparison between Point Cloud not normalize and normalize")
ax1 = fig.add_subplot(131, projection="3d")
ax2 = fig.add_subplot(132, projection="3d")
ax3 = fig.add_subplot(133, projection="3d")



ax1.set_title ("Not normalize")
ax1.scatter(pcd_n[:, 0], pcd_n[:, 1], pcd_n[:, 2],color='r', alpha=1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2.set_title ("Normalize")
ax2.scatter(pcd_norm[:, 0], pcd_norm[:, 1], pcd_norm[:, 2],color='g', alpha=1)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

ax3.set_title ("Normalize Voxel")
ax3.scatter(indices[:, 0], indices[:, 1], indices[:, 2],color='b', alpha=1)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')


plt.show()