#model_from_json library
def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration string and returns a model instance.
    Usage:
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Dense(5, input_shape=(3,)),
    ...     tf.keras.layers.Softmax()])
    >>> config = model.to_json()
    >>> loaded_model = tf.keras.models.model_from_json(config)
    Args:
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
    Returns:
        A Keras model instance (uncompiled).
    """
    from keras.layers import (
        deserialize_from_json,
    )

    return deserialize_from_json(json_string, custom_objects=custom_objects)

import numpy as np
import keras.models
from scipy.misc import imread, imresize,imshow
import tensorflow as tf


def init(): 
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model,graph