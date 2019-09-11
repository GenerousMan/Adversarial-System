import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
from attacks.l2_attack import  CarliniL2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

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
  start = time.time()
  with tf.Graph().as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      images = np.load("./results/slim-10000-images.npy")
      labels = np.load("./results/slim-10000-labels.npy")
      model_file = "./models/resnet_v1_50.ckpt"
      total_size = images.shape[0]
      batch_size = 100
      image_class = 1000
      height, width = 224, 224

      labels[labels > 998] = 960
      labels = np.clip((labels+10),0,999)
      images =images/255.0-0.5

      b = np.zeros((total_size, image_class))
      b[np.arange(total_size), labels] = 1.0
      labels = b
      print (labels.shape)

      X = tf.placeholder(tf.float32, [None, height, width, 3])
      lab = tf.placeholder(tf.float32, (None, image_class))

      model = resnet_v1.resnet_v1_50
      with slim.arg_scope(resnet_v1.resnet_arg_scope()):
          net,end = model(X,1000,is_training=False)

      model = Model(224, 3, 1000, model)
      net = model.predict(X)

      correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(lab, 1))
      accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      with tf.Session(config=config) as sess:

          init = tf.global_variables_initializer()
          sess.run(init)
          saver = tf.train.Saver(tf.global_variables())
          checkpoint_path = model_file
          saver.restore(sess, checkpoint_path)
          print("load success!")

          CW = CarliniL2(sess, model, batch_size=batch_size, confidence=10, targeted=True, max_iterations=1000)

          epochs_num = total_size // batch_size
          count  = 0
          count2 = 0
          cur_accuracy=0
          cur_accuracy2=0
          temp = np.zeros(( total_size, 224, 224, 3), dtype=np.float32)
          for i in range(epochs_num):
                imgs_batch = images[i * batch_size:(i + 1) * batch_size]
                labs_batch = labels[i * batch_size:(i + 1) * batch_size]
                acc,logit= sess.run([accurary,net], feed_dict={X: imgs_batch,
                                                    lab:labs_batch})
                print (acc)
                count += len(labs_batch)
                cur_accuracy += acc*100
                print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy/(i+1)))

                adv = CW.attack(imgs_batch,labs_batch)
                adv= np.reshape(adv,(batch_size,224,224,3))
                acc2,logit= sess.run([accurary,net], feed_dict={X: adv,
                                                    lab:labs_batch})

                cur_accuracy2 += acc2*100
                print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy2/(i+1)))

                for j in range(0, batch_size):
                    temp[i * batch_size + j] = adv[j]
                print (np.max(temp),np.min(temp))

          np.save("./results/CW-target-slim-k10",temp)
          print('run time:',time.time() - start)