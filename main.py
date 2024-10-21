"""
Problem Statement - The normal chest X-ray (left panel) depicts clear lungs without any areas of abnormal opacification (opaque/cloudy)
in the image. Bacterial pneumonia (middle) typically exhibits a focal lobar consolidation, in this case in the right upper lobe 
(white arrows), whereas viral pneumonia (right) manifests with a more diffuse ‘‘interstitial’’ pattern in both lungs.
"""
# Importing Libraries
import numpy
import os
import random
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

import cv2 as cv
from keras import backend as bknd
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPool2D, Flatten, Dense, Dropout
warnings.filterwarnings("ignore")

# Base directory of dataset
base_dir = r'E:\Kaggle\Notebooks\Chest_X_Ray_Pneumonia\Dataset\chest_xray'

# Train, Test , Val dataset
train_dir = base_dir + r'\train'
test_dir = base_dir + r'\test'
val_dir = base_dir + r'\val'

# Categories location
normal_case_train = train_dir + r'\NORMAL'
pneumonia_case_train = train_dir + r'\PNEUMONIA'


# Read random training image
print("Random normal images : ")


for _ in range(3):
    random_img = random.choice(os.listdir(normal_case_train))
    print(random_img)
    img_path = os.path.join(normal_case_train, random_img)
    img_sample = cv.imread(img_path)
    height, width, channels = img_sample.shape
    print("Image size : ",(width, height, channels))
    plt.figure()
    plt.imshow(img_sample)

# Count Plot for both classes [NORMAL & PNEUMONIA] 
# Creating DataFrame 
train_df = []

# Assigning 0 to normal case
for img in os.listdir(normal_case_train):
    
    train_df.append((img, 0))
    
for img in os.listdir(pneumonia_case_train):
    
    train_df.append((img, 1))
    
train_df = pd.DataFrame(train_df, columns=['Image', 'Label'], index = None)

# Random shuffling complete dataset
train_df = train_df.sample(frac=1).reset_index(drop=True)

cnt_plt = sns.countplot(train_df.Label)

cnt_plt.set_title("Count of Positive (0) and Negative Cases (1)")

print(train_df.shape)


print(train_df.head())

# Data Augumentation
img_width, img_height = 250, 250
batch_size=20
epochs = 50
nb_val_samples = 1000
# Image shape
if bknd.image_data_format == "channels_first":

    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range = 0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range = 0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Apply on data
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size,class_mode='binary')

val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size,class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size,class_mode='binary')


# Model Building
"""
Steps:
1. Initializing the ConvNet
2. Define by adding layers
3. Compiling the model
4. Fit/Train model
"""
# Initialize model
model = Sequential()

# Defining model
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dropout(1))
model.add(Activation('sigmoid'))

model.summary()

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(train_generator, steps_per_epochs=train_df.shape[0]//batch_size, epochs=epochs,
validation_data=val_generator, validation_steps=nb_val_samples//batch_size)

test_accuracy = model.evaluate_generator(test_generator)

print("The accuracy on test set : ", test_accuracy[1] * 100)

