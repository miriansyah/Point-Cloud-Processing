from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from Data_ModelNET10_PCD_to_Voxel_HDF5 import read_hdf5

X_train, y_train, X_test, y_test = read_hdf5("data_voxel.h5")

reg = LogisticRegression()
reg.fit(X_train,y_train)
print("LR-Accuracy: ", reg.score(X_test,y_test))

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("DT-Accuracy: ", dt.score(X_test,y_test))

svm = LinearSVC()
svm.fit(X_train,y_train)
print("SVM-Accuracy: ", svm.score(X_test,y_test))

knn = KNN()
knn.fit(X_train,y_train)
print("KNN-Accuracy: ", knn.score(X_test,y_test))

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)
print("RF-Accuracy: ", rf.score(X_test,y_test))