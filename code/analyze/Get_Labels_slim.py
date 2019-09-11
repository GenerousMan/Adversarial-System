'''
@author: lixurong
'''
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def Accuracy_slim(images,labels):
    start =time.time()
    with tf.Graph().as_default():

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
        now_label = tf.argmax(net, 1)
        correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(lab, 1))
        accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        real_label=np.zeros(total_size)

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
                [temp_label,acc] = sess.run([now_label,accurary], feed_dict={X: imgs_batch,
                                                                  lab: labs_batch})
                count += len(labs_batch)
                cur_accuracy += acc*100
                real_label[i*batch_size:(i+1)*batch_size]=temp_label

                
                print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy/(i+1)))

            print ("total time: ",time.time()-start)
    return real_label

if __name__=="__main__":
    # CW - target - slim - k0.npy
    orgpath = "../../datas/image/"
    labelpath = "../../datas/label/"
    orglabelpath = "../../datas/label/1000-vggres-same-label.npy"

    file_list = os.listdir(orgpath)
    for file in file_list:
        print(file)
        imgpath=orgpath+file
        if(file.split("-")[0]=="IGSM" or file.split("-")[0]=="FGSM" or file.split("-")[0]=="EAD"):
            images=np.load(imgpath)*255.
        elif(file.split("-")[0]=="CW"):
            images=(np.load(imgpath)+0.5)*255.
        else:
            images=np.load(imgpath)
        labels=np.load(orglabelpath).astype(np.int)
        real_labels=Accuracy_slim(images,labels)
        real_labels.shape=[-1]
        labels.shape=[-1]
        ml_labels=np.zeros(real_labels.shape[0])
        for j in range(labels.shape[0]):
            ml_labels[j]=(real_labels[j]==labels[j])
        print(ml_labels)
        np.save("../../datas/label/ML_Labels_"+file,ml_labels.astype(np.int))
      # print(labels.shape)
      #np.save("./results/labels-slim-CW-new-k"+str(eps),labels)