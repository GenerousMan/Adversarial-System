## coding=utf-8
import numpy as np
import skimage
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
from PIL import Image
import cv2 as cv
import numpy.matlib
import copy
import math
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


'''
xurong's code
'''

def rotate_image_30_tf(images):
    return tf.contrib.image.rotate(images,30)

def rotate_image_60_tf(images):
    return tf.contrib.image.rotate(images,60)

def rotate_image_90_tf(images):
    return tf.contrib.image.rotate(images,90)

def transfer_LR_tf(images):
    return tf.image.flip_left_right(images)

def transfer_TB_tf(images):
    return tf.image.flip_up_down(images)

def add_gaussian_Noise_tf(images,std):
    ## images[-0.5,0.5]
    return tf.clip_by_value(tf.add(images,tf.random_normal(shape=tf.shape(images),mean=0.0,stddev=std,dtype=tf.float32)),-0.5,0.5) #[-1,1]

def add_brightness_tf(images,delta):
    ##[-0.5,0.5]
    return tf.clip_by_value(tf.image.adjust_brightness(images,delta),-0.5,0.5)

def add_contrast(images,factor):
    return tf.clip_by_value(tf.image.adjust_contrast(images,factor),-0.5,0.5)

def add_saturation(images,factor):
    new = tf.clip_by_value(tf.image.adjust_saturation(images,factor),-0.5,0.5)
    return new

def add_hue(images,delta):
    new = tf.clip_by_value(tf.image.adjust_hue(images, delta=delta), -0.5, 0.5)
    return new

class Model():
    def __init__(self,image_size,num_channels,num_labels,recognizer):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.recognizer = recognizer

    def predict(self,images):
        images =(images+0.5)*255.0
        images = images - tf.constant([123.68, 116.779, 103.939])
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logit,end= self.recognizer(images,1000,is_training=False,reuse=True)
            logit = tf.reshape(logit,[-1,1000])
            return logit

if __name__=="__main__":
    images = np.load("./results/CW-untarget-slim-k30-1000.npy")
    labels = np.load("./results/slim-10000-labels.npy")[:100]
    # labels[labels > 998] = 960
    # labels = np.clip((labels + 10), 0, 999)

    # images = images/255.0-0.5
    # images = images[:5000]
    # labels =labels[:5000]
    start = time.time()
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        model_file = "./models/resnet_v1_50.ckpt"
        model = resnet_v1.resnet_v1_50
        image_class =1000
        x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        lab = tf.placeholder(tf.int32)
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end = model(x, image_class, is_training=False)

        model = Model(224, 3, image_class, model)
        net = model.predict(x)
        labs = tf.argmax(net,1)
        top_k = 1
        top_k_op = tf.nn.in_top_k(net, lab, top_k)

        x_30 = rotate_image_30_tf(x)
        x_60 = rotate_image_60_tf(x)
        x_90 = rotate_image_90_tf(x)
        x_TB = transfer_TB_tf(x)
        x_LR = transfer_LR_tf(x)
        x_guass = add_gaussian_Noise_tf(x, 0.05)
        x_bright = add_brightness_tf(x, 0.2)
        x_contrast = add_contrast(x, 2)
        x_sature = add_saturation(x, 2)
        x_hue = add_hue(x, 0.5)

        total_size = images.shape[0]
        batch_size = 100
        epochs_num = total_size // batch_size

        feats_30 =[]
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(tf.global_variables())
            checkpoint_path = model_file
            saver.restore(sess, checkpoint_path)
            print("load success!")

            for i in range(epochs_num):
                imgs_batch = images[i * batch_size:(i + 1) * batch_size]
                labs_batch = labels[i * batch_size:(i + 1) * batch_size]
                ori_lab = sess.run(labs,feed_dict={x:imgs_batch})

                # temp30 = sess.run(x_30,feed_dict={x:imgs_batch})
                temp60 = sess.run(x_60, feed_dict={x: imgs_batch})
                # temp90 = sess.run(x_90, feed_dict={x: imgs_batch})
                # templr = sess.run(x_LR, feed_dict={x: imgs_batch})
                # temptb = sess.run(x_TB, feed_dict={x: imgs_batch})
                # tempgauss = sess.run(x_guass, feed_dict={x: imgs_batch})
                # tempbright = sess.run(x_bright, feed_dict={x: imgs_batch})
                # tempcontrast = sess.run(x_contrast, feed_dict={x: imgs_batch})
                # tempsature = sess.run(x_sature, feed_dict={x: imgs_batch})
                # temphue = sess.run(x_hue, feed_dict={x: imgs_batch})

                acc = sess.run(top_k_op,feed_dict={x:imgs_batch,lab:ori_lab})
                print (np.sum(acc))
                # feat_30 = sess.run(top_k_op,feed_dict={x:temp30,lab:labs_batch})
                feat_60 = sess.run(top_k_op, feed_dict={x: temp60, lab: ori_lab})
                # feat_90 = sess.run(top_k_op, feed_dict={x: temp90, lab: labs_batch})
                # feat_lr = sess.run(top_k_op, feed_dict={x: templr, lab: labs_batch})
                # feat_tb = sess.run(top_k_op, feed_dict={x: temptb, lab: labs_batch})
                # feat_gauss = sess.run(top_k_op, feed_dict={x: tempgauss, lab: labs_batch})
                # feat_bright = sess.run(top_k_op, feed_dict={x: tempbright, lab: labs_batch})
                # feat_contrast = sess.run(top_k_op, feed_dict={x: tempcontrast, lab: labs_batch})
                # feat_sature = sess.run(top_k_op, feed_dict={x: tempsature, lab: labs_batch})
                # feat_hue = sess.run(top_k_op, feed_dict={x: temphue, lab: labs_batch})
                feats_30.append(feat_60)

            feats_30 = np.array(feats_30).astype("int")
            print (np.sum(feats_30))