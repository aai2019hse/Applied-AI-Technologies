import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D,Lambda, Flatten, Dense
from keras import backend as K

from scipy.misc import imread
import numpy as np


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape, name="input_1")
    right_input = Input(input_shape, name="input_2")

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(12, (10,10), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(24, (7,7), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(24, (4,4), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (4,4), activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid', name="main_output")(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # return the model
    return siamese_net


model = get_siamese_model((128, 128, 1))


model.load_weights(os.path.join("../01_weights", "weights.650.h5"))



image_reference = imread("../katzen_validierung/02_katzen/sansa.thecat/10_resized.jpg", mode='I')
image_test = imread("../katzen_validierung/02_katzen/sansa.thecat/03_resized.jpg", mode='I')

pairs = [image_reference.reshape(1, 128, 128, 1), image_test.reshape(1, 128, 128, 1)]
print(model.predict(pairs))
