import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim
from l2_attack import  CarliniL2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class Model():
    def __init__(self,image_size,num_channels,num_labels,recognizer):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.recognizer = recognizer

    def predict(self,images):
        images =(images+0.5)*255.0
        images = images - tf.constant([123.68, 116.779, 103.939])
        #with slim.arg_scope(vgg.vgg_arg_scope()):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
          tf.get_variable_scope().reuse_variables()
          logit, end = self.recognizer(images, self.num_labels, is_training=False)
          logit = tf.reshape(logit, [-1, self.num_labels])
          return logit


if __name__=="__main__":

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  images = np.load("./results/1000-vggres-same-image.npy")[:1000]
  print(images.shape)
  labels = np.load("./results/1000-vggres-same-label.npy")[:1000]

  labels=labels.astype(np.int64)
  model_file = "./models/vgg_16.ckpt"
  total_size = images.shape[0]
  batch_size = 50
  image_class = 1000
  height, width = 224, 224

  images = images/255-0.5

  b = np.zeros((total_size, image_class))
  b[np.arange(total_size), labels] = 1.0
  labels = b
  print (labels.shape)

  X = tf.placeholder(tf.float32, [None, height, width, 3])
  lab = tf.placeholder(tf.float32, (None, image_class))

  model = vgg.vgg_16
  with slim.arg_scope(vgg.vgg_arg_scope()):
      net, end_points = model(X, image_class,is_training=False)

  model = Model(224, 3, 1000, model)
  net = model.predict(X)
  correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(lab, 1))
  accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  ks=[10]
  for k in ks:
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = model_file
        saver.restore(sess, checkpoint_path)
        print("load success!")
  
        CW = CarliniL2(sess, model, batch_size=batch_size, confidence=k, targeted=False, max_iterations=1000)
  
        start_time = time.time()
        epochs_num = total_size // batch_size
        adv_all = np.zeros(shape=(1000,224,224,3))
        count  = 0
        for i in range(epochs_num):
          imgs_batch = images[i * batch_size:(i + 1) * batch_size]
          labs_batch = labels[i * batch_size:(i + 1) * batch_size]
          acc,logit= sess.run([accurary,net], feed_dict={X: imgs_batch,
                                              lab:labs_batch})
          print (acc)
          count += len(labs_batch)
          cur_accuracy = acc*100
          print('sec {:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy))
  
          adv = CW.attack(imgs_batch,labs_batch)
          adv= np.reshape(adv,(batch_size,224,224,3))
          acc2,logit= sess.run([accurary,net], feed_dict={X: adv,
                                              lab:labs_batch})
  
          cur_accuracy2 = acc2*100
          print('sec {:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy2))
          adv_all[i*batch_size:(i+1)*batch_size]=adv
        np.save("./results/CW-untarget-slim-vgg16-k10",adv_all)
        duration = time.time() - start_time
        print('run time:', duration)