import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import animation
import seaborn as sns

import h5py as h5
import os, sys

from VoxelGrid import VoxelGrid

style.use("ggplot")
sns.set_style("white")

#matplotlib inline
plt.rcParams['image.cmap'] = 'gray'


with h5.File('train_point_clouds.h5', 'r') as f:
  # Reading digit at zeroth index    
  a = f["0"]   
  # Storing group contents of digit a
  digit=(a["img"][:], a["points"][:], a.attrs["label"])

plt.imshow(digit[0])

digits = []

with h5.File("train_point_clouds.h5", 'r') as f:
    for i in range(15):
        d = f[str(i)]
        digits.append((d["img"][:],d["points"][:],d.attrs["label"]))


# Plot some examples from original 2D-MNIST
fig, axs = plt.subplots(3,5, figsize=(12, 12), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2)

for ax, d in zip(axs.ravel(), digits):
  ax.imshow(d[0][:])
  ax.set_title("Digit: " + str(d[2]))

plt.show()

print(digit[0].shape, digit[1].shape)

voxel_grid = VoxelGrid(digit[1], x_y_z = [16, 16, 16])

def count_plot(array):
    cm = plt.cm.get_cmap('gist_rainbow')
    n, bins, patches = plt.hist(array, bins=64)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    plt.show()

# Get the count of points within each voxel.
plt.title("DIGIT: " + str(digits[0][-1]))
plt.xlabel("VOXEL")
plt.ylabel("POINTS INSIDE THE VOXEL")

count_plot(voxel_grid.structure[:,-1]) 
print(voxel_grid.structure.shape)
print(digit[1].shape)

cloud_vis = np.concatenate((digit[1], voxel_grid.structure), axis=1)

print(cloud_vis)
np.savetxt('Cloud Visualization - ' + str(digit[2]) + '.txt', cloud_vis)