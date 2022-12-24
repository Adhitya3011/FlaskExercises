import os
import cv2
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, Dropout
from keras.optimizers import Adam

#loading model
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

#function to load the model given path of weights file
def load_trained_model(weights_path):
   model = build_model()
   model.load_weights(weights_path)
   return model

#creating flask app
app = Flask(__name__, template_folder='templates')

def init():
    global model
    model = load_trained_model('D:\EigenMaps.AI\Assignments\Flask Scikit Learn\DL MODEL MNIST\model.h5')

@app.route('/')
def upload_file():
   return render_template('index.html')

@app.route('/uploader', methods = ['POST'])
def upload_image_file():
    if request.method == 'POST':
        img = Image.open(request.files['file'].stream).convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        model = load_trained_model('D:\EigenMaps.AI\Assignments\Flask Scikit Learn\DL MODEL MNIST\model.h5')
        y_pred = np.argmax(model.predict(im2arr),axis=1)
        return 'Predicted Number: ' + str(y_pred[0])

if __name__ == '__main__':
    print("Loading Model...")
    init()
    app.run(debug=True)