import os
import cv2
import keras
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, Dropout
from tensorflow.keras.optimizers import Adam

app = Flask(__name__, static_folder='/templates')
run_with_ngrok(app)

app.config['UPLOADS'] = 'uploads'

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

def load_trained_model(weights_path):
   model = build_model()
   model.load_weights(weights_path)


cnn = load_trained_model('model.h5')

def process(file):
    image = cv2.imread(file)
    image = cv2.resize(image, (32, 32))
    image = np.resize(image, (1, 32, 32, 3))
    image = image/255.0
    image = 1-image
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        file = request.files['file']
        filepath = f'uploads/{file.filename}'
        file.save(filepath)
        image = process(filepath)
        print('process done')
        prediction = cnn.predict_classes(image)
    
        return render_template('index.html', number=prediction[0])

if __name__ == '__main__':
    app.run()