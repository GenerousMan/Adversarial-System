## coding=utf-8
import numpy as np
import skimage
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim
from PIL import Image
import cv2 as cv
import numpy.matlib
import copy
import math
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def fisheye_out(imgs,ga):
    gamma = ga
    row, col, channel = imgs[0].shape
    img_temp = copy.deepcopy(imgs)
    R = (min(row, col) / 2)
    center_x = (col - 1) / 2.0
    center_y = (row - 1) / 2.0
    xx = np.arange(col)
    yy = np.arange(row)
    x_mask = np.matlib.repmat(xx, row, 1)
    y_mask = np.matlib.repmat(yy, col, 1)
    y_mask = np.transpose(y_mask)
    xx_dif = x_mask - center_x
    yy_dif = center_y - y_mask
    r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
    theta = np.arctan(yy_dif / xx_dif)
    mask_1 = xx_dif < 0
    theta = theta * (1 - mask_1) + (theta + math.pi) * mask_1
    r_new = R * np.power(r / R, gamma)
    x_new = r_new * np.cos(theta) + center_x
    y_new = center_y - r_new * np.sin(theta)
    int_x = np.floor(x_new)
    int_x = int_x.astype(int)
    int_y = np.floor(y_new)
    int_y = int_y.astype(int)
    for ii in range(row):
        for jj in range(col):
            new_xx = int_x[ii, jj]
            new_yy = int_y[ii, jj]
            if x_new[ii, jj] < 0 or x_new[ii, jj] > col - 1:
                continue
            if y_new[ii, jj] < 0 or y_new[ii, jj] > row - 1:
                continue
            img_temp[:, ii, jj, :] = imgs[:, new_yy, new_xx, :]

    return img_temp.astype(np.float32)


def fisheye_in(imgs,ga):
    img_temp = copy.deepcopy(imgs)
    gamma = ga
    img = imgs[0]
    row, col, channel = img.shape
    R = (min(row, col) / 2)
    center_x = (col - 1) / 2.0
    center_y = (row - 1) / 2.0
    xx = np.arange(col)
    yy = np.arange(row)
    x_mask = np.matlib.repmat(xx, row, 1)
    y_mask = np.matlib.repmat(yy, col, 1)
    y_mask = np.transpose(y_mask)
    xx_dif = x_mask - center_x
    yy_dif = center_y - y_mask
    r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
    theta = np.arctan(yy_dif / xx_dif)
    mask_1 = xx_dif < 0
    theta = theta * (1 - mask_1) + (theta + math.pi) * mask_1
    r_new = R * np.power(r / R, gamma)
    x_new = r_new * np.cos(theta) + center_x
    y_new = center_y - r_new * np.sin(theta)
    int_x = np.floor(x_new)
    int_x = int_x.astype(int)
    int_y = np.floor(y_new)
    int_y = int_y.astype(int)
    for ii in range(row):
        for jj in range(col):
            new_xx = int_x[ii, jj]
            new_yy = int_y[ii, jj]
            if x_new[ii, jj] < 0 or x_new[ii, jj] > col - 1:
                continue
            if y_new[ii, jj] < 0 or y_new[ii, jj] > row - 1:
                continue
            img_temp[:, ii, jj, :] = imgs[:, new_yy, new_xx, :]
    return img_temp.astype(np.float32)


def Wave(imgs,a,b):
    img_temp = copy.deepcopy(imgs)
    A = a
    B = b
    img = imgs[0]
    row, col, channel = img.shape
    center_x = (col - 1) / 2.0
    center_y = (row - 1) / 2.0
    xx = np.arange(col)
    yy = np.arange(row)

    x_mask = np.matlib.repmat(xx, row, 1)
    y_mask = np.matlib.repmat(yy, col, 1)
    y_mask = np.transpose(y_mask)

    xx_dif = x_mask - center_x
    yy_dif = center_y - y_mask

    theta = np.arctan2(yy_dif, xx_dif)
    r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
    r1 = r + A * col * 0.01 * np.sin(B * 0.1 * r)

    x_new = r1 * np.cos(theta) + center_x
    y_new = center_y - r1 * np.sin(theta)

    int_x = np.floor(x_new)
    int_x = int_x.astype(int)
    int_y = np.floor(y_new)
    int_y = int_y.astype(int)

    for ii in range(row):
        for jj in range(col):
            new_xx = int_x[ii, jj]
            new_yy = int_y[ii, jj]

            if x_new[ii, jj] < 0 or x_new[ii, jj] > col - 1:
                continue
            if y_new[ii, jj] < 0 or y_new[ii, jj] > row - 1:
                continue
            img_temp[:, ii, jj, :] = imgs[:, new_yy, new_xx, :]

    return img_temp.astype(np.float32)

def TwirlDeal(imgs, Num):
    img_temp= imgs.copy()
    row, col, channel = imgs[0].shape
    xx = np.arange(col)
    yy = np.arange(row)
    x_mask = np.matlib.repmat(xx, row, 1)
    y_mask = np.matlib.repmat(yy, col, 1)
    y_mask = np.transpose(y_mask)

    center_y = (row - 1) / 2.0
    center_x = (col - 1) / 2.0

    R = np.sqrt((x_mask - center_x) ** 2 + (y_mask - center_y) ** 2)
    angle = np.arctan2(y_mask - center_y, x_mask - center_x)
    arr = (np.arange(Num) + 1) / 100.0

    for j in range(col):
        t = np.resize(angle[:,j], (col,1))
        T_angle = t + arr

        R_= np.resize( R[:, j], (col,1))
        new_x = R_ * np.cos(T_angle) + center_x
        new_y = R_ * np.sin(T_angle) + center_y

        int_x = new_x.astype(int)
        int_y = new_y.astype(int)

        int_x[int_x > col - 1] = col - 1
        int_x[int_x < 0] = 0
        int_y[int_y < 0] = 0
        int_y[int_y > row - 1] = row - 1

        img_temp[:,:, j, :] = imgs[:,int_y, int_x, :].sum(axis=2) / Num
    return img_temp
# def TwirlDeal(img, Num):
#     img1= img.copy()
#     row, col, channel = img1.shape
#     xx = np.arange(col)
#     yy = np.arange(row)
#     x_mask = numpy.matlib.repmat(xx, row, 1)
#     y_mask = numpy.matlib.repmat(yy, col, 1)
#     y_mask = np.transpose(y_mask)
#
#     center_y = (row - 1) / 2.0
#     center_x = (col - 1) / 2.0
#
#     R = np.sqrt((x_mask - center_x) ** 2 + (y_mask - center_y) ** 2)
#     angle = np.arctan2(y_mask - center_y, x_mask - center_x)
#     arr = (np.arange(Num) + 1) / 100.0
#
#     for j in range(col):
#         t = np.resize(angle[:,j], (col,1))
#         T_angle = t + arr
#
#         R_= np.resize( R[:, j], (col,1))
#         new_x = R_ * np.cos(T_angle) + center_x
#         new_y = R_ * np.sin(T_angle) + center_y
#
#         int_x = new_x.astype(int)
#         int_y = new_y.astype(int)
#
#         int_x[int_x > col - 1] = col - 1
#         int_x[int_x < 0] = 0
#         int_y[int_y < 0] = 0
#         int_y[int_y > row - 1] = row - 1
#
#         img1[:, j, :] = img[int_y, int_x, :].sum(axis=1) / Num
#     return img1

# def meanBlur(images):
#     images_new = np.zeros(shape=images.shape)
#     print (images.shape)
#     for l in range(images.shape[0]):#images.shape[1]
#         src = images[l]
#         # print ("ddd:",src.shape,type(src))
#         src1 = cv.blur(src, (3, 3))  #
#         src1.resize((224, 224, 3))
#         images_new[l]= src1
#     return images_new.astype(np.float32)

def bilateralBlur(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):  # images.shape[1]
        src = images[l]
        src1 = cv.bilateralFilter(src,5,80,80)
        src1.resize((224, 224, 3))
        images_new[l] = src1
    return images_new.astype(np.float32)

def twirl(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):  # images.shape[1]
        src = images[l]
        src1 = TwirlDeal(src, 4)
        src1.resize((224, 224, 3))
        images_new[l] = src1
    return images_new.astype(np.float32)

# def motionBlurDeal(img, size):
#     # generating the kernel
#     kernel_motion_blur = np.zeros((size, size))
#     kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
#     kernel_motion_blur = kernel_motion_blur / size
#     # applying the kernel to the input image
#     output = cv.filter2D(img, -1, kernel_motion_blur)
#     return output

# def MotionBlur(images):
#     images_new = np.zeros(shape=images.shape)
#     for l in range(images.shape[0]):  # images.shape[1]
#         src = images[l]
#         src1 = motionBlurDeal(src, 4)
#         src1.resize((224, 224, 3))
#         images_new[l] = src1
#     return images_new.astype(np.float32)

def Blur(images, blur, size):
    filter_weight = tf.get_variable('weights', [size, size, 1, 1],
                                    initializer=tf.constant_initializer(blur, dtype=tf.float32))
    conv =  tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:,:,:,0], axis=3), filter_weight, strides=[1, 1, 1, 1], padding='SAME'))#, 1,1
    conv1 = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:,:,:,1], axis=3), filter_weight, strides=[1, 1, 1, 1], padding='SAME') ) # , 1,1
    conv2 = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:,:,:,2], axis=3), filter_weight, strides=[1, 1, 1, 1], padding='SAME'))  # , 1,1
    conv_all = tf.stack([conv, conv1, conv2], axis=3)
    return conv_all

def meanBlur(images, blur, size):
    filter_weight = tf.get_variable('sweights', [size, size, 1, 1],
                                    initializer=tf.constant_initializer(blur, dtype=tf.float32))
    conv =  tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:,:,:,0], axis=3), filter_weight, strides=[1, 1, 1, 1], padding='SAME'))#, 1,1
    conv1 = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:,:,:,1], axis=3), filter_weight, strides=[1, 1, 1, 1], padding='SAME') ) # , 1,1
    conv2 = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:,:,:,2], axis=3), filter_weight, strides=[1, 1, 1, 1], padding='SAME'))  # , 1,1
    conv_all = tf.stack([conv, conv1, conv2], axis=3)
    return conv_all

'''
xurong's code
'''

def rotate_image_tf(images,angle):
    return tf.contrib.image.rotate(images,angle*np.pi/180.0)

def transfer_LR_tf(images):
    return tf.image.flip_left_right(images)

def transfer_TB_tf(images):
    return tf.image.flip_up_down(images)

def add_gaussian_Noise_tf(images,std):
    ## images[-0.5,0.5]
    return tf.clip_by_value(tf.add(images,tf.random_normal(shape=tf.shape(images),mean=0.0,stddev=std,dtype=tf.float32)),-0.5,0.5) #[-1,1]

def add_brightness_tf(images,delta):
    ##[-0.5,0.5]
    return tf.clip_by_value(tf.image.adjust_brightness(images,delta),-0.5,0.5)

def add_contrast(images,factor):
    return tf.clip_by_value(tf.image.adjust_contrast(images,factor),-0.5,0.5)

def add_saturation(images,factor):
    new = tf.clip_by_value(tf.image.adjust_saturation(images,factor),-0.5,0.5)
    return new

def add_hue(images,delta):
    new = tf.clip_by_value(tf.image.adjust_hue(images, delta=delta), -0.5, 0.5)
    return new

class Model():
    def __init__(self,image_size,num_channels,num_labels,recognizer):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.recognizer = recognizer

    def predict(self,images,flag=True):
        images =(images+0.5)*255.0
        images = images - tf.constant([123.68, 116.779, 103.939])
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logit,end= self.recognizer(images,1000,is_training=False,reuse=True)
            logit = tf.reshape(logit,[-1,1000])
            if flag:
                return logit
            else:
                prob = end["predictions"]
                prob = tf.reshape(prob,[-1,1000])
                return prob

if __name__=="__main__":
    # images = np.load("./results/slim-10000-images.npy")
    # images = np.load("../../ren_21721296/attacker/results/CW-untarget-slim-k10-10000.npy")
    

    labels = np.load("../../../datas/label/1000-vggres-same-label.npy")
    # images = images/255.0-0.5
    # images = images[:9900]
    # labels =labels[:9900]
    print ("attack")
    start = time.time()

    file_list=os.listdir("../../../datas/image/")
    for detect_file in file_list:
        images = np.load("../../../datas/image/"+detect_file)
        images =images.astype("float32")
        image_type = detect_file.split("-")[0]
        if(image_type=="CW"):
            images=images
        elif(image_type=="1000"):
            images=images/255.-0.5
        else:
            images=images-0.5
        print(image_type,np.max(images),np.min(images))
        with tf.Graph().as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            model_file = "../../../models/resnet_v1_50.ckpt"
            model = resnet_v1.resnet_v1_50
            image_class =1000
            x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
            lab = tf.placeholder(tf.int32)
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end = model(x, image_class, is_training=False)

            model = Model(224, 3, image_class, model)
            net = model.predict(x,False)
            top_k = 1
            top_k_op = tf.nn.in_top_k(net, lab, top_k)

            x_5 = rotate_image_tf(x,5)
            x_10 = rotate_image_tf(x,10)
            x_15 = rotate_image_tf(x,15)
            x_TB = transfer_TB_tf(x)
            x_LR = transfer_LR_tf(x)
            x_guass = add_gaussian_Noise_tf(x, 0.04)
            x_bright = add_brightness_tf(x, 0.1)
            x_contrast = add_contrast(x, 0.3)
            x_sature = add_saturation(x, 0.1)
            x_hue = add_hue(x, 0.5)

            size = 4
            kernel_motion_blur = np.zeros((size, size),dtype=np.float32)
            kernel_motion_blur[int((size - 1) / 2)+1, :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size
            kernel_mean_blur = np.ones([3, 3]) / 9

            pre_vars = tf.global_variables()
            x_motion = Blur(x, kernel_motion_blur, 4)  # motionBlur
            x_meanblur = meanBlur(x, kernel_mean_blur, 3)  # meanBlur
            pos_vars = tf.global_variables()
            uninit = list(set(pos_vars) - set(pre_vars))

            print (time.time()-start)
            x_fish_in =fisheye_in(images,0.9)
            x_fish_out = fisheye_out(images,1.1)
            x_wave = Wave(images,a=1.5,b=1.0)
            x_twirl = TwirlDeal(images,5)
            x_biater = bilateralBlur(images)
            print(time.time() - start)

            total_size = images.shape[0]
            batch_size = 100
            epochs_num = total_size // batch_size
            features = np.zeros((100,17))
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                varss = list(set(tf.global_variables()) - set(uninit))
                saver = tf.train.Saver(varss)
                checkpoint_path = model_file
                saver.restore(sess, checkpoint_path)
                print("load success!")

                for i in range(epochs_num):
                    print (i)
                    imgs_batch = images[i * batch_size:(i + 1) * batch_size]
                    labs_batch = labels[i * batch_size:(i + 1) * batch_size]
                    temp5 = sess.run(x_5,feed_dict={x:imgs_batch})
                    temp10 = sess.run(x_10, feed_dict={x: imgs_batch})
                    temp15 = sess.run(x_15, feed_dict={x: imgs_batch})
                    templr = sess.run(x_LR, feed_dict={x: imgs_batch})
                    temptb = sess.run(x_TB, feed_dict={x: imgs_batch})
                    tempgauss = sess.run(x_guass, feed_dict={x: imgs_batch})
                    tempbright = sess.run(x_bright, feed_dict={x: imgs_batch})
                    tempcontrast = sess.run(x_contrast, feed_dict={x: imgs_batch})
                    tempsature = sess.run(x_sature, feed_dict={x: imgs_batch})
                    temphue = sess.run(x_hue, feed_dict={x: imgs_batch})
                    tempmeanblur = sess.run(x_meanblur, feed_dict={x: imgs_batch})
                    tempmotion = sess.run(x_motion, feed_dict={x: imgs_batch})

                    prob_ori = sess.run(net,feed_dict={x:imgs_batch})
                    feat_1 = sess.run(net,feed_dict={x:temp5})
                    feat_2 = sess.run(net, feed_dict={x: temp10})
                    feat_3 = sess.run(net, feed_dict={x: temp15})
                    feat_4 = sess.run(net, feed_dict={x: templr})
                    feat_5 = sess.run(net, feed_dict={x: temptb})
                    feat_6 = sess.run(net, feed_dict={x: tempgauss})
                    feat_7 = sess.run(net, feed_dict={x: tempbright})
                    feat_8 = sess.run(net, feed_dict={x: tempcontrast})
                    feat_9 = sess.run(net, feed_dict={x: tempsature})
                    feat_10 = sess.run(net, feed_dict={x: temphue})
                    feat_11 = sess.run(net, feed_dict={x: x_fish_in[i * batch_size:(i + 1) * batch_size]})
                    feat_12 = sess.run(net, feed_dict={x: x_fish_out[i * batch_size:(i + 1) * batch_size]})
                    feat_13 = sess.run(net, feed_dict={x: x_wave[i * batch_size:(i + 1) * batch_size]})
                    feat_14 = sess.run(net, feed_dict={x: x_twirl[i * batch_size:(i + 1) * batch_size] })
                    feat_15 = sess.run(net, feed_dict={x: tempmotion})
                    feat_16 = sess.run(net, feed_dict={x: tempmeanblur})
                    feat_17 = sess.run(net, feed_dict={x: x_biater[i * batch_size:(i + 1) * batch_size]})

                    feat =[feat_1,feat_2,feat_3,feat_4,feat_5,feat_6,feat_7,feat_8,feat_9,feat_10,feat_11,feat_12,feat_13
                        , feat_14,feat_15,feat_16,feat_17]
                    KL = ['KL1','KL2','KL3','KL4','KL5','KL6','KL7','KL8','KL9','KL10','KL11','KL12','KL13','KL14','KL15','KL16','KL17']

                    for ind in range(len(feat)):
                        KL[ind]=np.reshape(np.sum(np.abs(feat[ind]-prob_ori),1),(100,1))

                    # print (KL[0].shape)
                    temp = np.hstack((KL[0],KL[1],KL[2],KL[3],KL[4],KL[5],KL[6],KL[7],KL[8],KL[9],KL[10]
                                      ,KL[11],KL[12],KL[13],KL[14],KL[15],KL[16]))
                    # print(temp.shape)
                    features = np.vstack((features,temp))
                    print(features.shape)

            features = features[100:]
            print(features.shape,np.max(features),np.min(features))
            np.save("../../../features/"+detect_file.split(".")[0]+"_feature",features)
            print (time.time()-start)