'''
@author: lixurong
'''
import numpy as np
from Resnet50.resnet_model import resnet50
from Googlenet.googlenet_model import googlenet
from VGG16.vgg16_model import vgg16
from Nin.nin_model import nin
import tensorflow as tf
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


'''
the imagesnpy is  (100,224,224,3)
'''
def Accurate(imagesnpy):
    with tf.Graph().as_default():

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        model_file ="./Resnet50/asset/model/resnet"
        model_name="ResNet50"

        batch_size=100
        images = np.load(imagesnpy)
        labels = np.load("./results/labels-res-10000.npy")

        print (np.shape(images))
        total_size = images.shape[0]
        imgs_original = images[0:total_size]
        labs_original = labels[0:total_size]

        top_k = 1
        x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        y = resnet50(x)
        lab = tf.placeholder(tf.int32)

        top_k_op = tf.nn.in_top_k(y, lab, top_k)
        reload_vars = {}
        uninit_vars = []
        for v in tf.global_variables():
            if v.name.startswith(model_name):
                reload_vars[v.op.name] = v
            else:
                uninit_vars.extend(v)

        saver = tf.train.Saver(reload_vars)

        with tf.Session(config=config) as sess:
            saver.restore(sess, model_file)
            sess.run(tf.variables_initializer(var_list=uninit_vars))

            epochs_num = total_size//batch_size
            correct=0
            count=0
            Acc=[]
            adv_conficence=[]
            corr_confidence=[]
            for i in range(epochs_num):
                imgs_batch = imgs_original[i * batch_size:(i + 1) * batch_size]
                labs_batch = labs_original[i * batch_size:(i + 1) * batch_size]

                match = sess.run(top_k_op,feed_dict={x: imgs_batch,
                                                        lab: labs_batch})
                a=sess.run(y,feed_dict={x:imgs_batch})
                b= np.max(a,1)##get prob of every amples

                prob_correct = np.mean(match*b)
                prob_adv = np.mean((1-match)*b)

                adv_conficence.append(prob_adv)
                corr_confidence.append(prob_correct)


                correct += np.sum(match)
                count += len(labs_batch)
                cur_accuracy = float(correct) * 100 / count
                Acc.append(cur_accuracy)
                print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy))

        print ("total accuracy:",np.mean(Acc))
        print ("aver confidence on adversarail examples:",np.mean(adv_conficence))
        print ("aver confidence on correct examples:",np.mean(corr_confidence))
        print (" compute accrute end")
   
'''
the imagesnpy is (2,10000,224,224,3)
''' 
def Accurate2(imagesnpy):##(2,10000,224,224,3)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    model_file ="./Resnet50/asset/model/resnet"
    model_name="ResNet50"
    
    batch_size=20
    images = np.load(imagesnpy)
    labels = np.load("./results/labels-res-100.npy")
        
    print (np.shape(images))
    total_size = images.shape[1]
    imgs_original = images[1][0:total_size]
    labs_original = labels[0:total_size]
           
    top_k = 1
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet50(x)
    lab = tf.placeholder(tf.int32)
        
    top_k_op = tf.nn.in_top_k(y, lab, top_k)
    reload_vars = {}
    uninit_vars = []
    for v in tf.global_variables():
        if v.name.startswith(model_name):
            reload_vars[v.op.name] = v
        else:
            uninit_vars.extend(v)
            
    saver = tf.train.Saver(reload_vars)
                
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_file)
        sess.run(tf.variables_initializer(var_list=uninit_vars))
        
        epochs_num = total_size//batch_size
        correct=0
        count=0
        Acc=[]
        adv_conficence=[]
        corr_confidence=[]
        for i in range(epochs_num):
            imgs_batch = imgs_original[i * batch_size:(i + 1) * batch_size]
            labs_batch = labs_original[i * batch_size:(i + 1) * batch_size]
            
            match = sess.run(top_k_op,feed_dict={x: imgs_batch,
                                                    lab: labs_batch})
            a=sess.run(y,feed_dict={x:imgs_batch})
            b= np.max(a,1)##get prob of every amples
            
            prob_correct = np.mean(match*b)
            prob_adv = np.mean((1-match)*b)
            
            adv_conficence.append(prob_adv)
            corr_confidence.append(prob_correct)
            
            
            correct += np.sum(match)
            count += len(labs_batch)
            cur_accuracy = float(correct) * 100 / count
            Acc.append(cur_accuracy)
            print('{:>6}/{:<6} {:>6.2f}%'.format(count, total_size, cur_accuracy))    
            
    print ("total accuracy:",np.mean(Acc)) 
    print ("aver confidence on adversarail examples:",np.mean(adv_conficence))   
    print ("aver confidence on correct examples:",np.mean(corr_confidence))   
    print (" compute accrute end")

'''
crop a image to 80 small images,
and vote through the the final results among 
'''
def Accurate3(imagesnpy,labelnpy):
##(10000,80,224,224,3)   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    model_file ="./Resnet50/asset/model/resnet"
    model_name="ResNet50"
    modelr = resnet50
    
    batch_size=80
    image_class=1000
    images = np.load(imagesnpy)
    labels_original = np.load(labelnpy)
        
    print (np.shape(images))
    print(np.shape(labels_original))
    total_size = images.shape[0]
    num=images.shape[1]
    
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet50(x)
    
    reload_vars = {}
    uninit_vars = []
    for v in tf.global_variables():
        if v.name.startswith(model_name):
                reload_vars[v.op.name] = v
        else:
                uninit_vars.extend(v)
                
    saver = tf.train.Saver(reload_vars)
                    
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_file)
        sess.run(tf.variables_initializer(var_list=uninit_vars))
        labels=[]

        for index in range(total_size):
            print("index:",index)
            imgs_batch = images[index]    
            a=sess.run(y,feed_dict={x:imgs_batch})
            b= np.argmax(a,1)##get prob of every samples
            counts = np.bincount(b)
            labels.append(np.argmax(counts))
            print (labels[index],labels_original[index])
            
        labels=np.array(labels)
        acc= labels_original-labels
        correct=0
        for i in range(total_size):
            if acc[i]==0:
                correct=correct+1
        print (correct*1.0/total_size)
        print ("end")
###
def model_predict(imagesnpy):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    model_file ="./Resnet50/asset/model/resnet"
    model_name="ResNet50"
    
    batch_size=100
    images = np.load(imagesnpy)
        
    print (np.shape(images))
    total_size = images.shape[1]
    imgs_original = images[1][0:total_size]
           
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = resnet50(x)
        
    reload_vars = {}
    uninit_vars = []
    for v in tf.global_variables():
        if v.name.startswith(model_name):
            reload_vars[v.op.name] = v
        else:
            uninit_vars.extend(v)
            
    saver = tf.train.Saver(reload_vars)
                
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_file)
        sess.run(tf.variables_initializer(var_list=uninit_vars))
        
        epochs_num = total_size//batch_size
        labels=[]
        for i in range(epochs_num):
            print ((i+1)*batch_size)
            imgs_batch = imgs_original[i * batch_size:(i + 1) * batch_size]
            a=sess.run(y,feed_dict={x:imgs_batch})
            pro = np.round(-np.sort(-a,1),decimals=3)
            top_index=np.argsort(-a,1)
            for j in range(top_index.shape[0]):
                labels.append(top_index[j][0])
        labels=np.array(labels)
        np.save("./results/old_CWk0_labels",labels)
        print (labels.shape)

    
if __name__=="__main__":
    start = time.time()
    path="./results/images-res-10000.npy"
    Accurate(path)
    print (time.time()-start)
