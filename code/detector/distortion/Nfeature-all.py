### coding=utf-8
import numpy as np
import tensorflow as tf
from Resnet50.resnet_model import resnet50
import os
import time
from Nfeature_all_final import extract_feature
from sklearn.externals import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''
juntao's code
'''
class Model():
    def __init__(self,image_size,num_channels,num_labels,recognizer):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.recognizer = recognizer

    def predict(self,data,flag=False):
        return self.recognizer(data,flag)
if __name__=='__main__':
    with tf.Graph().as_default():
        start = time.time()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        detector = joblib.load("./ML_models/K0_RF.m")

        model_file ="./Resnet50/asset/model/resnet"
        model_name="ResNet50"
        model = resnet50
        images = np.load("./results/adv-100-images-detector-target-k0.npy")
        labels = np.load("./results/labels-res-10000.npy")
        images=images[:100]
        labels=labels[:100]
        x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        y = model(x)
        lab = tf.placeholder(tf.int32)

        batch_size=100
        total_size = images.shape[0]
        imgs_original = images
        labs_original = labels
        imgs_class =1000

        labels = np.clip((labels+10),0,999)

        b = np.zeros((total_size, imgs_class))
        b[np.arange(total_size), labels] = 1.0
        labs_original = b

        print (labs_original.shape)


        with tf.Session(config=config) as sess:
            imgs = tf.placeholder(tf.float32, (100, images.shape[1], images.shape[2], images.shape[3]))
            labs = tf.placeholder(tf.float32, (None, imgs_class))
            _,y = model(imgs,True)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labs,1))
            accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            vars_dict = {}
            for v in tf.trainable_variables():
                if v.name.startswith(model_name):
                    vars_dict[v.op.name] = v
            saver = tf.train.Saver(vars_dict)
            saver.restore(sess, model_file)

            epochs_num = total_size//batch_size
            Acc = np.zeros((1), dtype=np.float32)
            model = Model(224,3,1000,model)

            pre_vars = tf.global_variables()
            feat = extract_feature(model,imgs)
            pos_vars = tf.global_variables()
            uninit = list(set(pos_vars) - set(pre_vars))
            sess.run(tf.variables_initializer(uninit))

            for i in range(epochs_num):
                imgs_batch = imgs_original[i * batch_size:(i + 1) * batch_size]
                labs_batch = labs_original[i * batch_size:(i + 1) * batch_size]
                acc, logit = sess.run([accurary, y], feed_dict={imgs: imgs_batch,
                                                                labs: labs_batch})
                print('orginal image acc is %f' % (acc))
                res4 = sess.run(feat,feed_dict={imgs:imgs_batch})
                res4 = res4[:,1:]
                print (res4.shape,res4)
                prob = detector.predict_proba(res4)
                prob = np.reshape(prob[:,1],[batch_size,1])
                print (prob)
                res = detector.predict(res4)
                print(res,np.sum(res))
                newlogit = np.reshape(np.max(logit, 1), [batch_size, 1])
                temp = 2 * prob* newlogit
                logitall = np.hstack((logit, temp))
                print (logitall.shape)

                print (np.argmax(logit,1))
                print (np.argmax(logitall,1))


        end = time.time()
        print ("total time:",end-start)