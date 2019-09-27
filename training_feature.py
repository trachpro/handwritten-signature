""" This example shows how to extract features for a new signature, 
    using the CNN trained on the GPDS dataset [1]. It also compares the
    results with the ones obtained by the authors, to ensure consistency.

    Note that loading and compiling the model takes time. It is preferable
    to load and process multiple signatures in the same python session.

    [1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features
    for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks"

"""
from scipy.misc import imread, imshow
from load import get_dataset, read_new_dataset
from preprocess.normalize import preprocess_signature
# import signet
import keras_model
# from cnn_model import CNNModel
import numpy as np
import six
import cv2
from keras.optimizers import Adadelta
import keras


# canvas_size = (952, 1360)  # Maximum signature size
# original = imread('/home/tupm/projects/sigver_wiwd/data/a2.png', flatten=1)
# processed = preprocess_signature(original, canvas_size)
# imshow(processed)
# Load the model
X, y = read_new_dataset('/home/tupm/projects/TextRecognitionDataGenerator/out')
model_weight_path = 'models/signet.pkl'
# model = CNNModel(signet, model_weight_path)
model = keras_model.SegNet((952, 1360, 1), 20)

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=Adadelta(lr=0.001),
              metrics=['accuracy'])

model.fit(x=X, y=y, batch_size=16, epochs=50)

# feature_vector = model.get_feature_vector(processed)

