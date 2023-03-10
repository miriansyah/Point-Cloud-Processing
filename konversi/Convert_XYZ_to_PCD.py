from ouster import client
from ouster import pcap
import mayavi.mlab
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


hostname = 'os-122215001365.local'
lidar_port = 7502
imu_port = 7503

meta_path = "D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/data_lidar_ouster/tangan_samping/tangan_samping.json"
pcap_path = "D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/data_lidar_ouster/tangan_samping/tangan_samping.pcap"

with open(meta_path, 'r') as f:
    info = client.SensorInfo(f.read())

source = pcap.Pcap(pcap_path, info)
metadata = source.metadata
scans = iter(client.Scans(source))

scan = next(scans,1)

xyzlut = client.XYZLut(metadata)
xyz = xyzlut(scan.field(client.ChanField.RANGE))

x,y,z = [c.flatten() for c in np.dsplit(xyz, 3)]
data_velo = np.array([x,y,z]).T
print(data_velo.shape)

with open('D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/txt_and_pcd/tangan_samping.txt','wb') as f:
    np.savetxt(f,data_velo)

pcd = o3d.io.read_point_cloud("D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/txt_and_pcd/tangan_samping.txt", format='xyz')
o3d.io.write_point_cloud("D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/data_lidar_indoor/convert_to_pcd/txt_and_pcd/tangan_samping.pcd", pcd)