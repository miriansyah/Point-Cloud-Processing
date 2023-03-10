import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from keras.utils import to_categorical
from keras.initializers import Constant
import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import h5py
import seaborn as sns
sns.set_style('white')

from imblearn.over_sampling import SMOTE

# -- Preparatory code --
# Model configuration
batch_size = 10
no_epochs = 50
learning_rate = 0.001
no_classes = 4
validation_split = 0.2
verbosity = 1


# -- Process code --
# Load the HDF5 data file
with h5py.File("data_voxel_orientasi.h5", "r") as hf:    

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

    #print(X_train.shape)
    #print(X_test.shape)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_title ("Normalize")
    # ax.scatter(X_plan[:, 0], X_plan[:, 1], X_plan[:, 2],color='g', alpha=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    # Convert target vectors to categorical targets
    targets_train = to_categorical(targets_train).astype(np.int32)
    targets_test = to_categorical(targets_test).astype(np.int32)
    

    # #Create the model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.5))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(no_classes, activation='softmax'))

    #create the model 2
    # model = Sequential()
    # model.add(Conv3D(32,(3,3,3),activation='relu',input_shape=(16,16,16,1),bias_initializer=Constant(0.01)))
    # model.add(Conv3D(32,(3,3,3),activation='relu',bias_initializer=Constant(0.01)))
    # model.add(MaxPooling3D((2,2,2)))
    # model.add(Conv3D(64,(3,3,3),activation='relu'))
    # model.add(Conv3D(64,(2,2,2),activation='relu'))
    # model.add(MaxPooling3D((2,2,2)))
    # model.add(Dropout(0.6))
    # model.add(Flatten())
    # model.add(Dense(256,'relu'))
    # model.add(Dropout(0.7))
    # model.add(Dense(128,'relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10,'softmax'))
    # model.summary()
    # print(model.summary())
    # # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
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
    cm = pd.DataFrame(array, index = range(4), columns = range(4))
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

   