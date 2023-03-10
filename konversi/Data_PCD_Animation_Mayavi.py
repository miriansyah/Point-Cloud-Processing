from __future__ import absolute_import, division, print_function
from mayavi import mlab
import numpy as np
import math
import open3d as o3d
import glob
import struct
import os



def lidar_callback(path):
   pcd = o3d.io.read_point_cloud(path)
   pcd_np= np.asarray(pcd.points)
   #copy points into buffer
   return pcd_np
    

dataset_path = "D:/Disertasi/Oprek/OPREK/ModelNetOrientasi_pcd/front/"
bin_path = dataset_path + "train"


fig = mlab.figure(bgcolor=(1, 1, 1), size=(1024, 800))
vis = mlab.points3d(0, 0, 0, 
                    mode='cube',
                    #colormap='spectral', 
                    color=(1, 0, 0),
                    scale_factor=0.01,
                    line_width=10, 
                    figure=fig)


data = {}

for i,f in enumerate(glob.glob(bin_path+"/*")):
    data [i]= lidar_callback(f)
    #print(data)
    

@mlab.animate(delay=100)
def anim():
    while True:
        for i, j in enumerate(data) :
            #print(data[i][:,2])
            vis.mlab_source.reset(x=data[i][:,0], y=data[i][:,1], z=data[i][:,2])
            yield

anim()
mlab.show()