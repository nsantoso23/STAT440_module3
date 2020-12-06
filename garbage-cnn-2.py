#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 23:53:28 2020

@author: nathaniasantoso
"""

#load libraries
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

train = pd.read_csv('/Users/nathaniasantoso/Downloads/labels.csv')


# fix random seed for reproducibility
seed = 321
np.random.seed(seed)
# We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('/Users/nathaniasantoso/Downloads/garbage/tr/'+train['name'][i], target_size=(28,28,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
x = np.array(train_image)

#one-hot encode target variable
y=train['label'].values
y = to_categorical(y)

#create validation set from training data
x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=42, test_size=0.2)
#test = pd.read_csv('/Users/nathaniasantoso/Downloads/garbage_test.csv')
#x_test = test['name']
#y_test = 
print('Training data shape: ', x_train.shape, y_train.shape)
print ('Validation data shape: ', x_val.shape, y_val.shape)

#finding unique numbers from train dataset in labels column
class_train = np.unique(y_train)
nclass_train = len(class_train)
print('Total no. of outputs : ', nclass_train)
print('Output classes : ', class_train)

#display image of training data
#plt.figure(figsize=[5,5])

#plt.subplot(121)
#plt.imshow(x_train[0,:,:], cmap = 'gray')
#plt.title("Ground Truth : {}".format(y_train[0]))

# Display the first image in testing data
#plt.subplot(122)
#plt.imshow(x_val[0,:,:], cmap='gray')
#plt.title("Ground Truth : {}".format(y_val[0]))

#x_train = x_train.reshape(-1,28,28,1)
#x_val = x_val.reshape(-1,28,28,1)
#x_train.shape, x_val.shape

#contributes massively to determining the 
#learning parameters and affects the prediction accuracy
batch_size = 64
epochs = 40
num_classes = 7

#build model structure with layers
garbage_model = Sequential()
garbage_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,3),padding='same'))
garbage_model.add(LeakyReLU(alpha=0.1))
garbage_model.add(MaxPooling2D((2, 2),padding='same'))
garbage_model.add(Dropout(0.2))
garbage_model.add(BatchNormalization())

garbage_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
garbage_model.add(LeakyReLU(alpha=0.1))
garbage_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
garbage_model.add(Dropout(0.2))
garbage_model.add(BatchNormalization())

garbage_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
garbage_model.add(LeakyReLU(alpha=0.1))                  
garbage_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
garbage_model.add(Dropout(0.2))
garbage_model.add(BatchNormalization())

garbage_model.add(Flatten())
#garbage_model.add(Dense(64, activation='linear'))
#garbage_model.add(LeakyReLU(alpha=0.1))         
#garbage_model.add(Dropout(0.3)) 
garbage_model.add(Dense(7, activation='linear'))
garbage_model.add(LeakyReLU(alpha=0.1))         
garbage_model.add(Dropout(0.2))         
garbage_model.add(Dense(num_classes, activation='softmax'))

garbage_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

garbage_model.summary()

#train the model
garbage_train = garbage_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

#making prediction
#load test set
test = pd.read_csv('/Users/nathaniasantoso/Downloads/garbage_test.csv')
#read and store test image
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('/Users/nathaniasantoso/Downloads/garbage/te/'+test['name'][i], target_size=(28,28,3), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)

# making predictions
prediction = garbage_model.predict_classes(test)

from pandas import *

idx = Int64Index(range(1264,2528))
df = DataFrame(index = idx, data = prediction)

    
   
#creating submission file
#df_test= pd.DataFrame(img1, columns=['name'])
df.to_csv('/Users/nathaniasantoso/Downloads/kaggle_submission.csv', index=True)