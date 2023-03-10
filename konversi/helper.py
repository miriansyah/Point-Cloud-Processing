import numpy as np

import os
import argparse
from contextlib import closing
from typing import Tuple, List

import numpy as np

from ouster import client, pcap

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def pcap_to_pcd(source: client.PacketSource,
                metadata: client.SensorInfo,
                num: int = 0,
                pcd_dir: str = ".",
                pcd_base: str = "pcd_out",
                pcd_ext: str = "pcd") -> None:
    "Write scans from a pcap to pcd files (one per lidar scan)."

    if (metadata.format.udp_profile_lidar ==
            client.UDPProfileLidar.PROFILE_LIDAR_RNG19_RFL8_SIG16_NIR16_DUAL):
        print("Note: You've selected to convert a dual returns pcap. Second "
              "returns are ignored in this conversion by this example "
              "for clarity reasons.  You can modify the code as needed by "
              "accessing it through github or the SDK documentation.")

    from itertools import islice
    try:
        import open3d as o3d  # type: ignore
    except ModuleNotFoundError:
        print(
            "This example requires open3d, which may not be available on all "
            "platforms. Try running `pip3 install open3d` first.")
        exit(1)

    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)

    # precompute xyzlut to save computation in a loop
    xyzlut = client.XYZLut(metadata)

    # create an iterator of LidarScans from pcap and bound it if num is specified
    scans = iter(client.Scans(source))
    if num:
        scans = islice(scans, num)

    for idx, scan in enumerate(scans):

        xyz = xyzlut(scan.field(client.ChanField.RANGE))

        pcd = o3d.geometry.PointCloud()  # type: ignore

        pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1,
                                                            3))  # type: ignore

        pcd_path = os.path.join(pcd_dir, f'{pcd_base}_{idx:06d}.{pcd_ext}')
        print(f'write frame #{idx} to file: {pcd_path}')

        o3d.io.write_point_cloud(pcd_path, pcd)  # type: ignore