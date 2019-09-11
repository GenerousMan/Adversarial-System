'''
@author: lixurong
'''
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def Accuracy_slim(imagenpy,labelpath):
    start =time.time()
    with tf.Graph().as_default():
        images = np.load(imagenpy)
        labels= np.load(labelpath).astype(np.int)

        model_file = "../../models/resnet_v1_50.ckpt"

        total_size = images.shape[0]
        print(total_size)
        #images = (images+0.5)*255.0
        images = images - [123.68, 116.779, 103.939]
        # labels = np.clip((labels + 10), 0, 999)
        model = resnet_v1.resnet_v1_50
        height = 224
        width = 224
        image_class = 1000
        batch_size = 100
        b = np.zeros((total_size, image_class))
        b[np.arange(total_size), labels] = 1.0
        labels = b

        X = tf.placeholder(tf.float32, [None, height, width, 3])
        lab = tf.placeholder(tf.float32, (None, image_class))
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end = model(X, 1000, is_training=False)

        net = tf.reshape(net, [-1, 1000])
        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(lab, 1))
        accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(tf.global_variables())
            checkpoint_path = model_file
            saver.restore(sess, checkpoint_path)
            epochs_num = total_size // batch_size
            cur_accuracy =0
            count =0
            for i in range(epochs_num):
                imgs_batch = images[i * batch_size:(i + 1) * batch_size]
                labs_batch = labels[i * batch_size:(i + 1) * batch_size]
                acc = sess.run(accurary, feed_dict={X: imgs_batch,
                                                                  lab: labs_batch})
                count += len(labs_batch)
                cur_accuracy += acc*100
                
                print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy/(i+1)))

            print ("total time: ",time.time()-start)

def Accuracy_slim2(imagenpy,labelpath):
    start =time.time()
    with tf.Graph().as_default():
        images = np.load(imagenpy)
        print(images.shape)
        labels= np.load(labelpath).astype(np.int)

        model_file = "../../models/resnet_v1_50.ckpt"

        total_size = images.shape[0]
        print(total_size)
        #images = (images+0.5)*255.0
        images = images - [123.68, 116.779, 103.939]
        # labels = np.clip((labels + 10), 0, 999)
        model = resnet_v1.resnet_v1_50
        height = 224
        width = 224
        image_class = 1000
        batch_size = 100
        b = np.zeros((total_size, image_class))
        b[np.arange(total_size), labels] = 1.0
        labels = b

        X = tf.placeholder(tf.float32, [None, height, width, 3])
        lab = tf.placeholder(tf.float32, (None, image_class))
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end = model(X, 1000, is_training=False)

        prob = end["predictions"]
        prob = tf.reshape(prob, [-1, 1000])
        probs =tf.maximum(prob,1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver(tf.global_variables())
            checkpoint_path = model_file
            saver.restore(sess, checkpoint_path)
            epochs_num = total_size // batch_size
            cur_accuracy =0
            count =0
            for i in range(epochs_num):
                imgs_batch = images[i * batch_size:(i + 1) * batch_size]
                labs_batch = labels[i * batch_size:(i + 1) * batch_size]
                acc = sess.run(prob, feed_dict={X: imgs_batch})
                print(acc)
                count += len(labs_batch)
                cur_accuracy += acc*100
                print(count,total_size,cur_accuracy)
                print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy/(i+1)))

            print ("total time: ",time.time()-start)

if __name__=="__main__":
    imgpath="../../datas/image/1000-vggres-same-image.npy"
    labelpath = "../../datas/label/1000-vggres-same-label.npy"
    Accuracy_slim(imgpath,labelpath)
