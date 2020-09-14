# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 23:25:10 2020

@author: deepak
"""

#### Convolution nural network

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout


classifier = Sequential()

### cnn network

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64 , 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.02))


classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.02))


classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.02))

classifier.add(Flatten())


classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### fitting the cnn


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('brain_tumor_dataset/training set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('brain_tumor_dataset/test set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 10,
                         validation_steps = 50)

from keras.preprocessing import image
import numpy as np

test_image = image.load_img('brain_tumor_dataset/Single_predication/No11.JPG', target_size = (64, 64 ))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

training_set.class_indices
if result[0][0] == 1:
    prediction = 'Brain_tumor exist'
else:
    prediction = 'No Brain_tumor'


