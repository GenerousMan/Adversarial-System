## train_defense.py
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

#from setup_mnist import MNIST
from setup_ImageNet import ImageNet
from defensive_models import DenoisingAutoEncoder as DAE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

poolings = ["average", "max"]

shape = [224, 224, 3]
combination_I = [3]
combination_II = [3,"average",3]
#max,average,max
#average,max,average
activation = "sigmoid"
reg_strength = 1e-9
epochs = 200

#data = MNIST()
data=ImageNet()

AE_I = DAE(shape, combination_I, v_noise=0.1, activation=activation,
           reg_strength=reg_strength)
AE_I.train(data, "ImageNet_III", num_epochs=epochs)

AE_II = DAE(shape, combination_II, v_noise=0.1, activation=activation,
            reg_strength=reg_strength)
AE_II.train(data, "ImageNet_IV", num_epochs=epochs)

