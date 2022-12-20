#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, Dropout
from tensorflow.keras.optimizers import Adam

#loading MNIST dataset
(X_train,Y_train),(X_test,Y_test) = tf.keras.datasets.mnist.load_data()
print(f'Image Size: {X_train[1].shape}')
print(f'No of Training Images: {X_train.shape[0]}')
print(f'No of Testing Images: {X_test.shape[0]}')

#plotting example images
"""n = 3
fig, ax = plt.subplots(n,n,figsize=(10,10))
for i in range(n):
    for j in range(n):
        img_ind = np.random.randint(0,len(X_train))
        ax[i,j].imshow(X_train[img_ind],cmap='Greys')
        ax[i,j].axis('off')
        ax[i,j].set_title(f'Index: {img_ind} Y: {Y_train[img_ind]}')
plt.show()"""

#Normalizing and one hot encoding
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255.0
X_test = X_test/255.0
Y_train = tf.keras.utils.to_categorical(Y_train)
Y_test = tf.keras.utils.to_categorical(Y_test)
X_train = X_train.reshape((X_train.shape[0],28,28,1))
X_test = X_test.reshape((X_test.shape[0],28,28,1))

#building model 
def build_model():
    model = Sequential()

    #First Conv Layer 
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding = 'same', activation='relu', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #Second Conv Layer 
    model.add(Conv2D(filters = 128, kernel_size=(3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #Third Conv Layer 
    model.add(Conv2D(filters = 128, kernel_size=(3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #Fourth Conv Layer 
    model.add(Conv2D(filters = 128, kernel_size=(3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(units = 256, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(units = 10, activation='softmax'))

    model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model()
model.summary()  

#fitting model
model.fit(X_train,Y_train,epochs=5,validation_data=(X_test,Y_test))

#getting results
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")