import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as ly
from keras.models import Model
from matplotlib import pyplot as plt

tf.random.set_seed(1234)
DATA_DIR = os.path.join("D:/Disertasi/Oprek/OPREK/", "ModelNet10")

#parsing dataset
def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            print(f)
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

def conv_bn(x, filters):
    x = ly.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = ly.BatchNormalization(momentum=0.0)(x)
    return ly.Activation("relu")(x)


def dense_bn(x, filters):
    x = ly.Dense(filters)(x)
    x = ly.BatchNormalization(momentum=0.0)(x)
    return ly.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = ly.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = ly.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,)(x)
    feat_T = ly.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return ly.Dot(axes=(2, 1))([inputs, feat_T])


NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

#INPUT DATASET

#mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))
#mesh.show()

#points = mesh.sample(2048)

# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)

#SETTING DATASET
# train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

# train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
# test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

# inputs = keras.Input(shape=(NUM_POINTS, 3))

# x = tnet(inputs, 3)
# x = conv_bn(x, 32)
# x = conv_bn(x, 32)
# x = tnet(x, 32)
# x = conv_bn(x, 32)
# x = conv_bn(x, 64)
# x = conv_bn(x, 512)
# x = ly.GlobalMaxPooling1D()(x)
# x = dense_bn(x, 256)
# x = ly.Dropout(0.3)(x)
# x = dense_bn(x, 128)
# x = ly.Dropout(0.3)(x)

# outputs = ly.Dense(NUM_CLASSES, activation="softmax")(x)

# model = Model(inputs=inputs, outputs=outputs, name="pointnet")
# model.summary()

# model.compile(
#     loss="sparse_categorical_crossentropy",
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     metrics=["sparse_categorical_accuracy"],
# )

# model.fit(train_dataset, epochs=1, validation_data=test_dataset)
# #model.save('D:/Disertasi/Oprek/OPREK/Point_Cloud_Dis/konversi/weight_model.h5')

# data = test_dataset.take(1)

# points, labels = list(data)[0]
# points = points[:8, ...]
# labels = labels[:8, ...]

# # run test data through model
# preds = model.predict(points)
# preds = tf.math.argmax(preds, -1)

# points = points.numpy()

# # plot points with predicted class and label
# fig = plt.figure(figsize=(15, 10))
# for i in range(8):
#     ax = fig.add_subplot(2, 4, i + 1, projection="3d")
#     ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
#     ax.set_title(
#         "pred: {:}, label: {:}".format(
#             CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
#         )
#     )
#     ax.set_axis_off()
# plt.show()