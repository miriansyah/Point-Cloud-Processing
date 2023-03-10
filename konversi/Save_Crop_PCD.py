# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 19:10:46 2022

@author: baith
"""
#!/usr/bin/python3

import numpy as np
import copy
import open3d as o3d
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob

dir_source = "pose_4/velodyne_pcd"
dir_save = "pose_4/velodyne_pcd_crop"

DATA_DIR = "D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_set_KITTI/" +dir_source
SAVE_PATH = "D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_set_KITTI/"+dir_save
print(DATA_DIR)

folders = sorted(glob.glob(os.path.join(DATA_DIR, "*")))

for i, folder in enumerate(folders):
      frame = Path(folder).stem
      print(frame)
      # # gather all files
      # frame_files = glob.glob(os.path.join(folder))
      # print(frame_files)

      # for f in frame_files:
      #       fname = Path(f).stem
      pcd = o3d.io.read_point_cloud(folder)
      vis = o3d.visualization.VisualizerWithEditing()
      vis.create_window()
      vis.add_geometry(pcd)
      vis.run()
      vis.destroy_window()
      cropped_geometry= vis.get_cropped_geometry()
      save_path = SAVE_PATH +"/"+ frame +".pcd"
      o3d.io.write_point_cloud(save_path, cropped_geometry)


   