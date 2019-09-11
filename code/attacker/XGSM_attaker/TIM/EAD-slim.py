import tensorflow as tf

from TMI_fgsm import Tmi_fgsm
import numpy as np
import time
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
import os
import copy
from PIL import Image
import cv2 as cv
from en_attack import EADEN
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Model():
    def __init__(self,image_size,num_channels,num_labels,recognizer):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.recognizer = recognizer

    def predict(self,images,flag=True):
        images = (images) * 255.0
        images = images - tf.constant([123.68, 116.779, 103.939])
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                scope.reuse_variables()
                logit, end = self.recognizer(images, 1000, is_training=False)
                probabilities = tf.nn.softmax(logit)
                logit = tf.reshape(logit, [-1, 1000])
                if flag:
                    return logit
                else:
                    prob = tf.reshape(probabilities, [-1, 1000])
                    return prob

if __name__=="__main__":

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  
  images = np.load("/home/nesa320/Ji_3160102420/1000-vggres-same-image.npy")

  #images[:,:,:,0:3]=images[:,:,:,0:3]-[123.68, 116.779, 103.939]
  images=images/255.0
  labels = np.load("/home/nesa320/Ji_3160102420/1000-vggres-same-label.npy").astype(int)
  attack_labels=np.load("/home/nesa320/Ji_3160102420/new_dataset/RES-50-LL-LABEL.npy").astype(int)
  model_file = "/home/nesa320/Ji_3160102420/model/resnet_v1_50.ckpt"
  batch_size = 10
  image_class =1000
  total_size = images.shape[0]

  b = np.zeros((total_size, image_class))
  b[np.arange(total_size), labels] = 1.0
  labels = b

  b = np.zeros((total_size, image_class))
  b[np.arange(total_size), attack_labels] = 1.0
  attack_labels = b
  
  height, width = 224, 224
  X = tf.placeholder(tf.float32, [None, height, width, 3])
  lab = tf.placeholder(tf.int32)
  model = resnet_v1.resnet_v1_50

  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = model(X, image_class,is_training=False)

  model = Model(224, 3, 1000, model)
  net =model.predict(X)
  correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(lab,1))
  accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  ep_s=[0.5]
  ks=[6,8,12,20]
  #ep_s=[8.0,16.0]
  adv_images=np.zeros(total_size*224*224*3)
  adv_images.shape=(total_size,224,224,3)
  for ep in ep_s:
    with tf.Session(config=config) as sess:
        k=0

        
        #imgs_adv=Tmi_fgsm(model,X,None,eps=ep)
        #imgs_adv = fgsm(model, X, lab, epochs=1, eps=ep, clip_min=0., clip_max=1.)
        #imgs_adv = igsm(model, X, lab, epochs=k, eps=1./255., clip_min=0., clip_max=1.,min_proba=2.0)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = model_file
        saver.restore(sess, checkpoint_path)
        print("model load success!")
  
        start_time = time.time()
        epochs_num = total_size // batch_size
        count = 0
        cur_accuracy=0
        cur_accuracy2=0
        attack = EADEN(sess, model, batch_size=batch_size, max_iterations=100, confidence=0,targeted=True)

        
        #adv_images=np.zeros(total_size)
        
        for i in range(epochs_num):
          imgs_batch = images[i * batch_size:(i + 1) * batch_size]
          labs_batch = labels[i * batch_size:(i + 1) * batch_size]
          attack_labs=attack_labels[i * batch_size:(i + 1) * batch_size]
          adv = attack.attack(imgs_batch, attack_labs)
          print(adv.shape)
          print(adv[0])
          #adv = sess.run([imgs_adv], feed_dict={X: imgs_batch,
          #                                      lab:attack_labs})
          for k in range(10):
            Pic_each=(adv[k]+0.)*255.0
            cv.imwrite("./test/EAD_0-1-"+str(k)+"_"+".png",Pic_each)
  
          logit,acc = sess.run([net,accurary], feed_dict={X: (imgs_batch+0.),
                                                          lab:labs_batch})
          count += len(labs_batch)
          cur_accuracy += acc * 100 / epochs_num
          print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy))
          adv= np.reshape(adv,(batch_size,224,224,3))
          adv_images[i * batch_size:(i + 1) * batch_size]=adv
          logit,acc = sess.run([net,accurary], feed_dict={X: (adv+0.),
                                                          lab:attack_labs})
          cur_accuracy2 += acc * 100 / epochs_num
          print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy2))
        
        np.save("/home/nesa320/Ji_3160102420/other_model_result/EAD-Res50-Slim.npy",adv_images)
        f=open("/home/nesa320/Ji_3160102420/modified_result/accuracy_EAD.txt","a+")
        f.write("0-1, res50, EAD , acc, "+str(cur_accuracy2)+"\n" )
        f.close()
        duration = time.time() - start_time
        print('run time:', duration)
        for i in range(10):
          Pic_each=(adv_images[i]+0.)*255.0
          img = Image.fromarray(Pic_each.astype('uint8'))
          img.save("/home/nesa320/Ji_3160102420/modified_result/test/EAD_0-1-"+str(k)+"_"+str(i)+".png")

  