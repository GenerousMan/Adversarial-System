##coding=utf-8###
'''
Created on 2018年9月10日
@author: lixurong
'''
import os
import numpy as np
from PIL import Image
from PIL import ImageFilter,ImageChops,ImageOps
import tensorflow as tf
from Resnet50.resnet_model import resnet50
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

    
if __name__=="__main__":
    image_path="./results/ResNet50_k=0__l2_images.npy"
    all_image =np.load(image_path)
    
    adv = all_image[1]
    adv_label = np.ones([adv.shape[0],1])
    clean = all_image[0]
    clean_label = np.zeros([clean.shape[0],1])
    
    X_all_train = np.vstack([adv,clean])
    y_all_train = np.vstack([np.ones([adv.shape[0], 1]),
                         np.zeros([clean.shape[0], 1])])

    print (X_all_train.shape)
    print (y_all_train.shape)
    input_shape = (20000,224,224,3)
    model1 = Sequential([
        Convolution2D(32, 3, 3, input_shape=input_shape),
        Activation('relu'),
        Convolution2D(32, 3, 3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(1),
        Activation('sigmoid')])

    model1.compile(loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

    print('\nTraining model1')
#     os.makedirs('model', exist_ok=True)
    model1.fit(X_all_train, y_all_train, nb_epoch=5,
               validation_split=0.1)

    print('\nSaving model1')
    os.mkdir("model")
    model1.save('model/table_1_mnist_model1.h5')
    
    print('\nTesting against adv test data')
    score = model1.evaluate(adv, adv_label)
    print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))


    print('\nTesting against clean  data')
    score = model1.evaluate(clean, clean_label)
    print('\nloss: {0:.4f} acc: {1:.4f}'.format(score[0], score[1]))

    
