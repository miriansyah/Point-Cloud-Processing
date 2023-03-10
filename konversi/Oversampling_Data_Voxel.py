from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
import numpy as np
import h5py

# define dataset

with h5py.File("data_voxel_40.h5", "r") as hf:    

    # Split the data into training/test features/targets
    X_train = hf["X_train"][:]
    X_train = np.array(X_train)

    targets_train = hf["y_train"][:]
  
    X_test = hf["X_test"][:] 
    X_test = np.array(X_test)

    targets_test = hf["y_test"][:]
    test_y = targets_test

oversample = SMOTE()
X, y = oversample.fit_resample(X_train, targets_train)

# summarize the new class distribution
counter = Counter(y)
print(counter)
print(np.array(X))
