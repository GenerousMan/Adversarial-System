###coding=utf-8
import tensorflow as tf
import numpy as np
import sys
import time
import os

from attacks.l2_attack_target import CarliniL2
from Resnet50.resnet_model import resnet50

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Model():
    def __init__(self, image_size, num_channels, num_labels, recognizer):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.recognizer = recognizer

    def predict(self, data, flag=False):
        return self.recognizer(data, flag)


if __name__ == '__main__':

    model_file = "./Resnet50/asset/model/resnet"
    model_name = "ResNet50"
    model = resnet50
    images = np.load("./results/images-res-10000.npy")
    labels = np.load("./results/labels-res-10000.npy")
    images = images[:1000]
    labels = labels[:1000]

    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = model(x)
    batch_size = 200
    total_size = images.shape[0]
    imgs_original = images
    labs_original = labels
    imgs_class = 1001
    labels = np.clip((labels + 10), 0, 999)

    b = np.zeros((total_size, imgs_class))
    b[np.arange(total_size), labels] = 1.0
    labs_original = b

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        imgs = tf.placeholder(tf.float32, (None, images.shape[1], images.shape[2], images.shape[3]))
        labs = tf.placeholder(tf.float32, (None, imgs_class))
        y = model(imgs)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labs, 1))
        accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        modelCW = Model(224, 3, imgs_class, model)
        attack = CarliniL2(sess, modelCW, targeted=True, batch_size=batch_size, max_iterations=150, confidence=0)

        vars_dict = {}
        for v in tf.trainable_variables():
            if v.name.startswith(model_name):
                vars_dict[v.op.name] = v
        saver = tf.train.Saver(vars_dict)
        saver.restore(sess, model_file)

        epochs_num = total_size // batch_size
        Acc = np.zeros((1), dtype=np.float32)
        images_temp= np.zeros((total_size,224,224,3),dtype=np.float32)
        for i in range(epochs_num):
            imgs_batch = imgs_original[i * batch_size:(i + 1) * batch_size]
            labs_batch = labs_original[i * batch_size:(i + 1) * batch_size]
            acc, proba = sess.run([accurary, y], feed_dict={imgs: imgs_batch,
                                                            labs: labs_batch})
            print("acc:", acc)

            timestart = time.time()
            adv_imgs = attack.attack(imgs_batch, labs_batch)
            timeend = time.time()
            print('use time %f to generate %d sample' % (timeend - timestart, batch_size))

            acc, proba = sess.run([accurary, y], feed_dict={imgs: adv_imgs,
                                                            labs: labs_batch})
            print('%d-th adv image acc is %f' % ((i + 1) * batch_size, acc))
            Acc += acc / epochs_num
            for j in range(batch_size):
                images_temp[i*batch_size+j]=adv_imgs[j]

        print("end,", Acc)
        np.save("./results/adv-1000-images-detector-target-k0-max150", images_temp)