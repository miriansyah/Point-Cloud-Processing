import math
import numpy as np

import h5py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import os, sys
np.set_printoptions(threshold=sys.maxsize)

OMEGA = 27
VOXEL_MIN_VALUE = -53

def transform_voxel_data(voxels, n_voxs):
    weighted_voxels = np.zeros((n_voxs[0], n_voxs[1], n_voxs[2]))

    for i in range(n_voxs[0]):
        for j in range(n_voxs[1]):
            for k in range(n_voxs[2]):
                # Count zeros and ones in neighbors
                zeros_count = 0
                ones_count  = 0

                for m in range(i - 1, i + 2):
                    for n in range(j - 1, j + 2):
                        for p in range(k - 1, k + 2):
                            weight = OMEGA if m == i and n == j and p == k else 1
                            v = 0

                            if m >= 0 and n >= 0 and p >= 0 and m < n_voxs[0] and n < n_voxs[1] and p < n_voxs[2]:
                               v = voxels[m][n][p]

                            if v:
                                ones_count += weight
                            else:
                                zeros_count += weight

                # VOXEL_MIN_VALUE is used for normalizing to values with minimum value 0
                # The value in binvox cannot less than 0
                value = ones_count - zeros_count
                weighted_voxels[i][j][k] = value - VOXEL_MIN_VALUE

    return weighted_voxels








