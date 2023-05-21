import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import random
import matplotlib.pyplot as plt
import pickle


valid_dir = 'path to valid directory'
train_dir = 'path to train directory'
test_dir = 'path to test directory'

train_datagen = ImageDataGenerator(horizontal_flip = True, 
                                   vertical_flip = True, 
                                   rescale =1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (100, 100),
    color_mode = "rgb",
    batch_size = 200,
    class_mode = "input",
    shuffle = True,
    seed = 42)


valid_generator = train_datagen.flow_from_directory(
    directory = valid_dir,
    target_size = (100, 100),
    color_mode = "rgb",
    batch_size = 200,
    class_mode = "input",
    shuffle = False,
    seed = 42)

test_generator = train_datagen.flow_from_directory(
    directory = test_dir,
    target_size = (100, 100),
    color_mode = "rgb",
    batch_size = 200,
    class_mode = "input",
    shuffle = False,
    seed = 42)

input_shape = (100, 100, 3)

input_img = Input(shape=input_shape)
x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(input_img)
x = Conv2D(64, 3, strides=2, activation='relu', padding='valid')(x)
x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
x = Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)
x = Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)
x = Conv2D(64, 3, strides=2, activation='relu', padding='valid', name='embedding')(x)

x = Conv2DTranspose(128, 3, strides=2, padding='valid', activation='relu', name='deconv6')(x)
x = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu', name='deconv5')(x)
x = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu', name='deconv4')(x)
x = Conv2DTranspose(64, 3, strides=2, padding='valid', activation='relu', name='deconv3')(x)
x = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu', name='deconv2')(x)
reconstructed = Conv2DTranspose(input_shape[2], 3, strides=2, padding='same', name='deconv1')(x)

model = Model(inputs = input_img, outputs = reconstructed)
model.summary()

opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.9)
model.compile(optimizer=opt, loss='mse')
encoder = Model(inputs = model.input, outputs = model.get_layer('embedding').output)
encoder.summary()

history = model.fit(x = test_generator, steps_per_epoch = 10000, epochs = 100, verbose = 1)

test_generator.reset()
data = encoder.predict(test_generator,verbose = 1)

data = data.reshape((data.shape[0],64))

out_dir = 'path to the output directory'
np.save(os.path.join(out_dir, 'nuclei_embedding_whole_64_part_02.npy'), data)

model.save_weights(os.path.join(out_path, 'model_weights_64.h5'))