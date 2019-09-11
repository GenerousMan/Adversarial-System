## coding=utf-8
import numpy as np
import tensorflow as tf
import cv2 as cv
import copy
import math

def rotate_image_tf_2(images):
    return tf.contrib.image.rotate(images,2*np.pi/180.0)

def rotate_image_tf_3(images):
    return tf.contrib.image.rotate(images,3*np.pi/180.0)

def rotate_image_tf_4(images):
    return tf.contrib.image.rotate(images,4*np.pi/180.0)

def rotate_image_tf_5(images):
    return tf.contrib.image.rotate(images,5*np.pi/180.0)

def rotate_image_tf_7(images):
    return tf.contrib.image.rotate(images,7*np.pi/180.0)

def transfer_LR_tf(images):
    return tf.image.flip_left_right(images)

def transfer_TB_tf(images):
    return tf.image.flip_up_down(images)

def add_gaussian_Noise_tf(images):
    ## images[-0.5,0.5]
    std = 0.04
    return tf.clip_by_value(tf.add(images,tf.random_normal(shape=tf.shape(images),mean=0.0,stddev=std,dtype=tf.float32)),-0.5,0.5) #[-1,1]

def add_brightness_tf(images):
    ##[-0.5,0.5]
    delta=0.1
    return tf.clip_by_value(tf.image.adjust_brightness(images,delta),-0.5,0.5)

def add_contrast(images):
    factor=0.3
    return tf.clip_by_value(tf.image.adjust_contrast(images,factor),-0.5,0.5)

def add_saturation(images):
    factor=0.1
    new = tf.clip_by_value(tf.image.adjust_saturation(images,factor),-0.5,0.5)
    return new

def add_hue(images):
    delta=0.5
    new = tf.clip_by_value(tf.image.adjust_hue(images, delta=delta), -0.5, 0.5)
    return new

def motionBlur(images):
    print(images.shape)
    size = 4
    kernel_motion_blur = np.zeros((size, size), dtype=np.float32)
    kernel_motion_blur[int((size - 1) / 2) + 1, :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    filter_weight = tf.get_variable('motionWeights', [size, size, 1, 1],
                                    initializer=tf.constant_initializer(kernel_motion_blur, dtype=tf.float32))
    conv = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:, :, :, 0], axis=3), filter_weight, strides=[1, 1, 1, 1],
                                   padding='SAME'))  # , 1,1
    conv1 = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:, :, :, 1], axis=3), filter_weight, strides=[1, 1, 1, 1],
                                    padding='SAME'))  # , 1,1
    conv2 = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:, :, :, 2], axis=3), filter_weight, strides=[1, 1, 1, 1],
                                    padding='SAME'))  # , 1,1
    conv_all = tf.stack([conv, conv1, conv2], axis=3)
    # conv_all.set_shape([10, 224, 224, 3])
    return conv_all

def meanBlur(images):
    size = 4
    kernel_mean_blur = np.ones([4, 4]) / 16
    filter_weight = tf.get_variable('meanWeights', [size, size, 1, 1],
                                    initializer=tf.constant_initializer(kernel_mean_blur, dtype=tf.float32))
    conv = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:, :, :, 0], axis=3), filter_weight, strides=[1, 1, 1, 1],
                                   padding='SAME'))  # , 1,1
    conv1 = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:, :, :, 1], axis=3), filter_weight, strides=[1, 1, 1, 1],
                                    padding='SAME'))  # , 1,1
    conv2 = tf.squeeze(tf.nn.conv2d(tf.expand_dims(images[:, :, :, 2], axis=3), filter_weight, strides=[1, 1, 1, 1],
                                    padding='SAME'))  # , 1,1
    conv_all = tf.stack([conv, conv1, conv2], axis=3)
    # conv_all.set_shape([10, 224, 224, 3])
    return conv_all

def fisheye_out(imgs):
    ga = 1.2
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


def fisheye_in(imgs):
    ga = 0.8
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


def Wave(imgs):
    a = 1.5
    b = 1.0
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

def TwirlDeal(imgs):
    Num = 5
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

def bilateralBlur(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):  # images.shape[1]
        src = images[l]
        src1 = cv.bilateralFilter(src,5,80,80)
        src1.resize((224, 224, 3))
        images_new[l] = src1
    return images_new.astype(np.float32)

def extract_feature(model, images, batch_size):
    print("images_shape_extract:", images.shape)
    prob_ori = model.predict(images, False)
    labels_original = tf.argmax(prob_ori, 1)

    # print("prob_ori.shape=", prob_ori.shape)
    # print("labels_original.shape=", labels_original.shape)

    py_list= [12, 13, 14, 15, 18]
    labels_l1 = tf.Variable(np.zeros([batch_size, 1]), dtype=tf.float32)
    labels_match = tf.Variable(np.zeros([batch_size, 1]), dtype=tf.float32)
    function_name= [rotate_image_tf_2,rotate_image_tf_3,rotate_image_tf_4,rotate_image_tf_5,rotate_image_tf_7,
                    transfer_LR_tf,transfer_TB_tf,#0-2
                    add_gaussian_Noise_tf,add_brightness_tf,add_contrast,add_saturation,#3-6
                    add_hue,fisheye_in,fisheye_out,#7-9
                    Wave,TwirlDeal,#10-11
                    motionBlur,meanBlur,#12-13
                    bilateralBlur]#16-18
    for index in range(0, len(function_name), 1):
        # print("================")
        # print("index:", index)
        if index in py_list:
            images_all = tf.py_func(function_name[index], [images],tf.float32)
            images_all.set_shape([batch_size, 224, 224, 3])
        else:
            images_all = function_name[index](images)
        net_new = model.predict(images_all, False)
        top_k_op = tf.cast(tf.nn.in_top_k(net_new, labels_original, 1),
                              dtype=tf.float32)
        L1_new = tf.reduce_sum(tf.abs(net_new-prob_ori),1)
        labels_l1 = tf.concat((labels_l1, tf.reshape(L1_new, (batch_size, 1))), 1)
        labels_match = tf.concat((labels_match, tf.reshape(top_k_op, (batch_size, 1))), 1)
    labels_l1=labels_l1[:,1:]
    labels_match=labels_match[:,1:]
    labels_all = tf.concat((labels_l1,labels_match),1)
    return labels_all

