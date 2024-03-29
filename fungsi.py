import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout,LeakyReLU
from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D

def make_model():
    mobilenet = MobileNet(input_shape=(224 , 224, 3),
                          include_top=False,
                          weights='imagenet')

    model = Sequential()
    model.add(mobilenet)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(4, activation="softmax", name="classification"))
    
    return model
