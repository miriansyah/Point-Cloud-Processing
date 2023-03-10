import numpy as np
import mayavi.mlab as mlab
fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))

def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan

#for i in range(10):
lidar_bin_path = "D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_set_KITTI/back/velodyne_points/data/0000000000.bin"
pc = load_velo_scan(lidar_bin_path)
pc = np.array(pc)
print(pc.shape)
mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        pc[:, 2],
        mode='point',
        color=(1,0,1),
        #colormap="gnuplot",
        colormap="copper",
        scale_factor=10,
        figure=fig)
mlab.show()