import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import open3d as o3d
import trimesh

DATA_DIR = os.path.join("D:/Disertasi/Oprek/OPREK/", "ModelNet40")
SAVE_PATH = os.path.join("D:/Disertasi/Oprek/OPREK/", "ModelNet40_pcd/")
print(DATA_DIR)

folders = glob.glob(os.path.join(DATA_DIR, "*"))
print (folders)
train_points = []
train_labels = []
class_map={}

for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("\\")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))
        #print(train_files)
        for f in train_files:
                #print(f)
                nama_train = os.path.splitext(f.split("\\")[-1])[0]
                xyz=trimesh.load(f).sample(2048)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                save_path =os.path.join(SAVE_PATH,class_map[i] + "/train/"+ nama_train+".pcd")
                o3d.io.write_point_cloud(save_path, pcd)

        for f in test_files:
                #print(f)
                nama_test = os.path.splitext(f.split("\\")[-1])[0]
                xyz_test=trimesh.load(f).sample(2048)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz_test)
                save_path =os.path.join(SAVE_PATH,class_map[i] + "/test/"+ nama_test+".pcd")
                o3d.io.write_point_cloud(save_path, pcd)
        

print(class_map)
#print(train_files)


