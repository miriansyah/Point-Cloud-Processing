import os
from keras.models import load_model
import open3d as o3d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from VoxelGrid import VoxelGrid
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import h5py

dir_dataset = "D:/Disertasi/Oprek/OPREK/ModelNetOrientasi_pcd"
dir_klasifikasi = "left"

def normalize_pc_range(pcd_np):
    scaler = MinMaxScaler()
    scaler.fit(pcd_np)
    pcd_np = scaler.transform(pcd_np)
    return pcd_np

def pcd_to_voxel(path):
    voxel_2d=[]
    #LOAD PCD
    pcd = o3d.io.read_point_cloud(path)
    R = pcd.get_rotation_matrix_from_axis_angle([0,0,1.571])
    pcd = pcd.rotate(R, center=(0,0,0))
    R = pcd.get_rotation_matrix_from_axis_angle([-1.571,0,0])
    pcd = pcd.rotate(R, center=(0,0,0))
    pcd_np= np.asarray(pcd.points)
    pcd_np=np.array(normalize_pc_range(pcd_np))
    #VOXELIZATION
    voxel_grid = VoxelGrid(pcd_np, x_y_z = [16, 16, 16])
    voxel_2d =np.array(voxel_grid.vector[:,:,:]) 
    voxel_2d=voxel_2d.reshape(-1)
    voxel_2d = np.where(voxel_2d > 0.0, 1, 0)
    voxel_final = voxel_2d.astype('float64')
    return voxel_final

def Klasifikasi(dir_dataset, dir_klasifikasi, LabelKelas,ModelCNN =[]):
    test_points = []

    dir_kelas = dir_dataset + '/'+ dir_klasifikasi + '/test'
    #print(dir_kelas)
    files = os.listdir(dir_kelas)
    #print(files)
    for fdata in files:
        nama_file = os.path.join(dir_kelas,fdata)
        test_points.append(pcd_to_voxel(nama_file))
    
    test_point=np.array(test_points)
    test_points = test_point.reshape(test_point.shape[0], 16, 16, 16)
    #print(test_points.shape)

    hs= ModelCNN.predict(test_points)
    #print(hs)

    LKlasifikasi =[]
    LKelasPoint =[]
    n = test_point.shape[0]
    for i in range(n):
        v=hs[i ,:]
        if v.max()> 0.5:
            idx = np.max(np.where(v==v.max()))
            LKelasPoint.append(LabelKelas[idx])
        else :
            idx =-1
            LKelasPoint.append("-")
        # ------ akhir if
        LKlasifikasi.append(idx)
    # ---- akhir for
    LKlasifikasi = np.array(LKlasifikasi)
    return hs, LKelasPoint

LabelKelas =("back","front","left","right")
#LabelKelas =("bathtub","bed","chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet")
# LabelKelas =("airplane","bathtub","bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone", "cup", 
# "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp", 
# "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio", "range_hood", 
# "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox")

ModelCNN = load_model("model_2d_orientasi.h5")

with h5py.File("data_voxel_orientasi.h5", "r") as hf:    

    # Split the data into training/test features/targets
    X_test = hf["X_test"][:] 
    X_test = np.array(X_test)

    targets_test = hf["y_test"][:]
    print(targets_test.shape)

hs, LKelasPoint = Klasifikasi (dir_dataset,dir_klasifikasi,LabelKelas, ModelCNN)
# array = confusion_matrix(targets_test, hs)
# cm = pd.DataFrame(array, index = range(10), columns = range(10))
# plt.figure(figsize=(20,20))
# sns.heatmap(cm, annot=True)
# plt.show()

print(LKelasPoint)