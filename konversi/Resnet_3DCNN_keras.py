import tensorflow as tf
from tensorflow import file_io
#import tensorflow.compat.v2 as tf
import keras
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import h5py
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt

import numpy as np
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical

layers = None

def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           classifier_activation='softmax',
           base_channel=None,
           **kwargs,
):
      
    global layers
    if 'layers' in kwargs:
        layers = kwargs.pop('layers')
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError('Unknown argument(s): %s' % (kwargs,))
    if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
        raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1

    base_channel = 64 if base_channel == None else base_channel

    x = layers.ZeroPadding3D(padding=3, name='conv1_pad')(img_input)
    x = layers.Conv3D(base_channel, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if not preact:
        x = layers.BatchNormalization( axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding3D(padding=1, name='pool1_pad')(x)
    x = layers.MaxPooling3D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation,name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling3D(name='max_pool')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

  # Create model.
    model = training.Model(inputs, x, name=model_name)

    if weights is not None:
        model.load_weights(weights)

    return model

def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    
    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv3D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv3D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv3D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv3D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1
    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = layers.Conv3D(4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling3D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv3D(filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding3D(padding=1, name=name + '_2_pad')(x)
    x = layers.Conv3D(filters,kernel_size,strides=stride,use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv3D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def block3(x,
           filters,
           kernel_size=3,
           stride=1,
           groups=32,
           conv_shortcut=True,
           name=None,
           base_channel=None):
  
    bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1
    
    base_channel = 64 if base_channel == None else base_channel

    if conv_shortcut:
        shortcut = layers.Conv3D(
                    (base_channel // groups) * filters,
                    1,
                    strides=stride,
                    use_bias=False,
                    name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
                    axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv3D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding3D(padding=1, name=name + '_2_pad')(x)
    x = layers.DepthwiseConv3D(
          kernel_size,
          strides=stride,
          depth_multiplier=c,
          use_bias=False,
          name=name + '_2_conv')(x)
    x_shape = backend.shape(x)[:-1]
    x = backend.reshape(x, backend.concatenate([x_shape, (groups, c, c)]))
    x = layers.Lambda(
          lambda x: sum(x[:, :, :, :, i] for i in range(c)),
          name=name + '_2_reduce')(x)
    x = backend.reshape(x, backend.concatenate([x_shape, (filters,)]))
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv3D(
      (base_channel // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
  
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None, base_channel=None):
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(
            x,
            filters,
            groups=groups,
            conv_shortcut=False,
            name=name + '_block' + str(i),
            base_channel=base_channel)
    return x


def ResNet50(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             base_channel=None,
            **kwargs,
):
    
    if base_channel == None:
        base_channel = 64 
    else:
        base_channel

    def stack_fn(x):
        x = stack1(x, base_channel*1, 3, stride1=1, name='conv2')
        x = stack1(x, base_channel*2, 4, name='conv3')
        x = stack1(x, base_channel*4, 6, name='conv4')
        
        return stack1(x, base_channel*8, 3, name='conv5')
    
    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        base_channel=base_channel,
        **kwargs,
    )

batch_size = 10
no_epochs = 50
learning_rate = 0.001
no_classes = 10
validation_split = 0.2
verbosity = 1

with h5py.File("/kaggle/input/data-set-modelnet10/data_voxel_modelnet10_16.h5", "r") as hf:    

    # Split the data into training/test features/targets
    X_train = hf["X_train"][:]
    X_train = np.array(X_train)

    targets_train = hf["y_train"][:]
  
    X_test = hf["X_test"][:] 
    X_test = np.array(X_test)

    targets_test = hf["y_test"][:]
    test_y = targets_test
    # Determine sample shape
    sample_shape = (16, 16, 16,1)

    #Over sampling
    oversample = SMOTE()
    X_train, targets_train = oversample.fit_resample(X_train, targets_train)
    X_train = np.array(X_train)

    X_train = X_train.reshape(X_train.shape[0], 16, 16, 16,1)
    X_test  = X_test.reshape(X_test.shape[0], 16, 16, 16,1)

    # Convert target vectors to categorical targets
    targets_train = to_categorical(targets_train).astype(np.int32)
    targets_test = to_categorical(targets_test).astype(np.int32)
    
    model = ResNet50(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=sample_shape,
             pooling=None,
             classes=no_classes,
             base_channel=16)
    
    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    # Fit data to model
    history = model.fit(X_train, targets_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_split=validation_split)

    #model.save('model_3d.h5')

    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)
    array = confusion_matrix(test_y, pred)
    cm = pd.DataFrame(array, index = range(10), columns = range(10))
    plt.figure(figsize=(20,20))
    sns.heatmap(cm, annot=True)
    plt.show()

    # # Generate generalization metrics
    score = model.evaluate(X_test, targets_test, verbose=0)
    print('Test accuracy: %.2f%% Test loss: %.3f' % (score[1]*100, score[0]))

    # # Plot history: Categorical Loss
    plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
    plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
    plt.title('Model performance for 3D Voxel Keras Conv3D (Loss)')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(['train','test'],loc="upper left")
    plt.show()

    # # Plot history: Categorical Accuracy
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title('Model performance for 3D Voxel Keras Conv3D (Accuracy)')
    plt.ylabel('Accuracy value')
    plt.xlabel('No. epoch')
    plt.legend(['train','test'],loc="upper left")
    plt.show()
