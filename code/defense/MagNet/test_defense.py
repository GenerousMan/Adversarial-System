## test_defense.py -- test defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

#from setup_mnist import MNIST
from utils import prepare_data
from worker import AEDetector, SimpleReformer, IdReformer, AttackData, Classifier, Operator, Evaluator
import utils
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

detector_I = AEDetector("./defensive_models/ImageNet_V", p=2)
detector_II = AEDetector("./defensive_models/ImageNet_III", p=1)
reformer = SimpleReformer("./defensive_models/ImageNet_V")

id_reformer = IdReformer()
classifier_path = "/home/nesa320/Ji_3160102420/MagNet-master/resnet_v1_50.ckpt"

detector_dict = dict()
detector_dict["I"] = detector_I
detector_dict["II"] = detector_II

untrusted_image_dir="/home/nesa320/Ji_3160102420/other_model_result/"
untrusted_label_dir="/home/nesa320/Ji_3160102420/modified_result/labels/"

UntrustedData_Path=os.listdir(untrusted_image_dir)
Untrustedlabel_Path=os.listdir(untrusted_label_dir)

classes=[3,1,1,1,1,1,1,2,2,1,1,1,1,1,1]
bounds=[0.08,0.1]
for bound in bounds:
  for j in range(len(UntrustedData_Path)):
    if(UntrustedData_Path[j].split("-")[1]!="Res50"):
      continue
    un_data=np.load(untrusted_image_dir+UntrustedData_Path[j])
    #nomal_data=np.load("/home/nesa320/Ji_3160102420/fgsm_new/results/slim-10000-images.npy")
    nomal_label=np.load("/home/nesa320/Ji_3160102420/1000-vggres-same-label.npy").astype(np.int)
    
    image_test=un_data
    mark=detector_I.mark(image_test)
    mark.shape=(un_data.shape[0])
    
    print("adv mark : ",mark.sum()/un_data.shape[0])
    f=open("test_detector_reformer_0122.txt","a+")
    f.write(UntrustedData_Path[j].split(".")[0]+", detecor mark: "+str(mark.sum()/un_data.shape[0])+"\n")
    f.close()
    np.save("mark/"+UntrustedData_Path[j].split(".")[0],mark)
    
    record_mark=np.zeros(un_data.shape[0])
    for i in range(un_data.shape[0]):
      if(mark[i]<bound):
        record_mark[i]=1
    
    Pic_new=reformer.heal(image_test)
    images = (Pic_new) * 255.
    labels = nomal_label
    labels2=np.load(untrusted_label_dir+"labels_"+UntrustedData_Path[j].split(".")[0]+".npy").astype(np.int)
    labels.shape=(images.shape[0])
    labels2.shape=(images.shape[0])
    labels2=labels2.astype(np.int)
    
    print("ori 0:",labels[0])
    print("adv 0:",labels2[0])
    #mark=detector_I.mark(nomal_data)
    
    #print("ori image mark : ",mark.sum()/10000)
    
    
    #np.save("./test/original.npy",mark)
    '''
    print(mark[0][0][0][0],mark[0][0][0][1])
    Pic_new=mark
    for i in range(10):
            Pic_each=Pic_new[i][...,::-1]*255.0
            img = Image.fromarray(Pic_each.astype('uint8'))
            img.save("test/AEC_"+str(i)+".png")
            #Pic_each=pic_temp[i][...,::-1]
            #img = Image.fromarray(Pic_each.astype('uint8'))
            #img.save("test/Ori_"+str(strength)+"_"+str(i)+".png")
    '''
    #operator = Operator(MNIST(), classifier, detector_dict, reformer)
    
    #idx = utils.load_obj("example_idx")
    #_, _, Y = prepare_data(MNIST(), idx)
    #f = "example_carlini_0.0"
    #testAttack = AttackData(f, Y, "Carlini L2 0.0")
    
    #evaluator = Evaluator(operator, testAttack)
    #evaluator.plot_various_confidences("defense_performance",
    #                                   drop_rate={"I": 0.001, "II": 0.001})
  
    with tf.Graph().as_default():
  
            #labels=labels.astype(np.int)
            #print(labels[0])
            model_file = "./resnet_v1_50.ckpt"
    
            total_size = un_data.shape[0]
            print(total_size)
            #images=(images)*255.0
            images = images - [123.68, 116.779, 103.939]
            #labels = np.clip((labels + 10), 0, 999)
            model = resnet_v1.resnet_v1_50
            height = 224
            width = 224
            image_class = 1000
            batch_size = 100
            b = np.zeros((total_size, image_class))
            b[np.arange(total_size), labels] = 1.0
            labels = copy.deepcopy(b)
            b = np.zeros((total_size, image_class))
            b[np.arange(total_size), labels2] = 1.0
            labels2 = copy.deepcopy(b)
            X = tf.placeholder(tf.float32, [None, height, width, 3])
            lab = tf.placeholder(tf.float32, (None, image_class))
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end = model(X, 1000, is_training=False)
    
            net = tf.reshape(net, [-1, 1000])
            labels_tf=tf.argmax(net, 1)
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
                count0=0
                count1=0
                acc_all=0
                acc_all2=0
                sum_mark=0
                for i in range(epochs_num):
                    imgs_batch = images[i * batch_size:(i + 1) * batch_size]
                    labs_batch = labels[i * batch_size:(i + 1) * batch_size]
                    print(labs_batch.shape)
                    labs_batch2 = labels2[i * batch_size:(i + 1) * batch_size]
                    mark_batch=record_mark[i * batch_size:(i + 1) * batch_size]
                    acc_now = sess.run(accurary,feed_dict={X: imgs_batch,lab:labs_batch})
                    #print(acc_now)
                    acc_now2 = sess.run(accurary,feed_dict={X: imgs_batch,lab:labs_batch2})
                    print(acc_now,acc_now2)
                    labels_now=sess.run(labels_tf,feed_dict={X: imgs_batch})
                    labels_now.shape=(100)
                    #print(i,",",acc_now)
                    for k in range(batch_size):
                      print((labs_batch[k].tolist().index(1),labs_batch2[k].tolist().index(1),labels_now[k]))
                      #print(labs_batch[k],labs_batch2[k],mark_batch[k],labels_now[k])
                      if((labs_batch[k].tolist().index(1)==labs_batch2[k].tolist().index(1)) and np.any(mark_batch[k]==1) and (labels_now[k]==labs_batch[k].tolist().index(1))):
                      #not a adv pic, and not be deleted, and it's also clean after reformer
                        count0=count0+1
                      elif((labs_batch[k].tolist().index(1)!=labs_batch2[k].tolist().index(1)) and np.any(mark_batch[k]==1) and (labels_now[k]==labs_batch[k].tolist().index(1))):
                        count1=count1+1
                      #it's an adv pic ,and not be deleted, then reforme it to a clean pic.
                    acc_all=acc_all+acc_now
                    acc_all2=acc_all2+acc_now2
                    print(count0,count1,mark_batch.sum())
                    sum_mark=sum_mark+mark_batch.sum()
                
                count2=sum_mark-count1-count0
                f=open("test_detector_and_reformer_0122.txt","a+")
                
                #sum_mark :total number which was not deleted
                #count0 : it has been clean before reformed
                #count1 : it's reformed as a clean pic.
                #count2 : it's reformed failed
                
                f.write(str(bound)+","+UntrustedData_Path[j].split(".")[0]+","+str(sum_mark)+","+str(count0)+","+str(count1)+","+str(count2)+"\n")
                f.close()
                print(acc_all/epochs_num)
                print(acc_all2/epochs_num)
                f=open("test_reformer_0122.txt","a+")
                f.write(str(j)+UntrustedData_Path[j].split(".")[0]+", Ori->Clean: "+str(acc_all/epochs_num)+"\n")
                f.write(str(j)+UntrustedData_Path[j].split(".")[0]+", Ori->Different: "+str((1-acc_all2/epochs_num))+"\n")
                if(UntrustedData_Path[j].split(".")[0]!="clean"):
                  f.write(str(j)+UntrustedData_Path[j].split(".")[0]+", Ori->Other: "+str((1-acc_all2/epochs_num)-acc_all/epochs_num)+"\n")
                f.close()
                
                