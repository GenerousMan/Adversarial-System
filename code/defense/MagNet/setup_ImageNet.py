## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

## Modified for MagNet's use.

import numpy as np
import os
import gzip
import urllib.request

from keras.models import load_model

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class ImageNet:
    def __init__(self):
        #if not os.path.exists("data"):
        #    os.mkdir("data")
  
            #for name in files:
            #    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)
        data_all=np.load("/home/nesa320/Ji_3160102420/attacker_new/work/Original-50000.npy")
        data_all=data_all/255.
        label_all=np.load("/home/nesa320/Ji_3160102420/attacker_new/work/Ori_labels_50000.npy")
        train_data = data_all[0:45000]#extract_data("data/train-images-idx3-ubyte.gz", 60000)+0.5
        train_labels = label_all[0:45000]#extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = data_all[45000:50000]#extract_data("data/t10k-images-idx3-ubyte.gz", 10000)+0.5
        self.test_labels = label_all[45000:50000]#extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 2000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

    @staticmethod
    def print():
        return "ImageNet"


class ImageNetModel:
    def __init__(self, restore, session=None):
        self.num_channels = 3
        self.image_size = 224
        self.num_labels = 1000
        self.model = load_model(restore)

    def predict(self, data):
        return self.model(data)
