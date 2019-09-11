import tensorflow as tf
from fgsmwrap import fgsm
from igsm import igsm
import numpy as np
import time
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
import os
import copy
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Model():
    def __init__(self,image_size,num_channels,num_labels,recognizer):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.recognizer = recognizer

    def predict(self,images):
        images =(images)*255.0
        images = images - tf.constant([123.68, 116.779, 103.939])
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logit,end= self.recognizer(images,1000,is_training=False,reuse=True)
            logit = tf.reshape(logit,[-1,1000])
            return logit


if __name__=="__main__":

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  
  images = np.load("../../../datas/image/1000-vggres-same-image.npy")

  #images[:,:,:,0:3]=images[:,:,:,0:3]-[123.68, 116.779, 103.939]
  images=images/255.0
  labels = np.load("../../../datas/label/1000-vggres-same-label.npy").astype(int)
  #model_file = "./model/resnet_v1_50.ckpt"
  batch_size = 10
  image_class =1000
  total_size = images.shape[0]

  b = np.zeros((total_size, image_class))
  b[np.arange(total_size), labels] = 1.0
  labels = b

  height, width = 224, 224
  X = tf.placeholder(tf.float32, [None, height, width, 3])
  lab = tf.placeholder(tf.int32)
  
  #resnet_v2_152,0-1,no sub
  model_file = "../../../models/resnet_v1_50.ckpt"
  model = resnet_v1.resnet_v1_50
  #images=images/255.
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net,end = model(X,1000,is_training=False)
      
  '''     
  model = resnet_v1.resnet_v1_50
  model_file = "./model/resnet_v1_50.ckpt"
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = model.predict(X)
  
  model_file = "./model/vgg_16.ckpt"
  model = vgg.vgg_16

  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = model.predict(X)
  '''
  model = Model(224, 3, 1000, model)
  net =model.predict(X)
  correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(lab,1))#sub 1 ,because the rank has 1001 classes.
  accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  ep_s=[2./255.,4./255.,8./255.,16./255.]
  #ep_s=[8.0,16.0]
  adv_images=np.zeros(total_size*224*224*3)
  adv_images.shape=(total_size,224,224,3)
  for ep in ep_s:
    with tf.Session(config=config) as sess:
        tf.get_variable_scope().reuse_variables()
        #imgs_adv = fgsm(model, X, lab, epochs=1, eps=ep, clip_min=0., clip_max=1.)
        imgs_adv = fgsm(model, X, lab,  eps=ep, clip_min=0.,clip_max=1.)
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
        #adv_images=np.zeros(total_size)
        for i in range(epochs_num):
          imgs_batch = images[i * batch_size:(i + 1) * batch_size]
          labs_batch = labels[i * batch_size:(i + 1) * batch_size]
          adv = sess.run([imgs_adv], feed_dict={X: imgs_batch,
                                                lab:labs_batch})
  
          logit,acc = sess.run([net,accurary], feed_dict={X: (imgs_batch+0.),
                                                          lab:labs_batch})

          print(np.argmax(logit,1))
          print(np.argmax(labs_batch,1))
          count += len(labs_batch)
          cur_accuracy += acc * 100 / epochs_num
          print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy))
          adv= np.reshape(adv,(batch_size,224,224,3))
          adv_images[i * batch_size:(i + 1) * batch_size]=adv
          logit,acc = sess.run([net,accurary], feed_dict={X: (adv+0.),
                                                          lab:labs_batch})
          cur_accuracy2 += acc * 100 / epochs_num
          print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy2))
        np.save("../../../datas/image/FGSM-Res-50-Slim-"+str(int(ep*255.))+".npy",adv_images)
        f=open("accuracy_FGSM.txt","a+")
        f.write("-0.5~0.5, "+str(ep*255.)+", acc, "+str(cur_accuracy2)+"\n" )
        f.close()
        duration = time.time() - start_time
        print('run time:', duration)
        for i in range(10):
          Pic_each=(adv_images[i]+0.)*255.0
          img = Image.fromarray(Pic_each.astype('uint8'))
          img.save("./test/FGSM_-0.5-0.5_"+str(ep)+"_"+str(i)+".png")
          Pic_each=(images[i])*255.0
          img = Image.fromarray(Pic_each.astype('uint8'))
          img.save("test/Ori_"+str(ep)+"_"+str(i)+".png")
  