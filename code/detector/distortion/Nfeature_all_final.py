## coding=utf-8
import numpy as np
import skimage
import tensorflow as tf
from PIL import Image
import cv2 as cv
import numpy.matlib
import copy
import math

'''
juntao's code
'''
def fisheye_out(imgs):
   img_temp=copy.deepcopy(imgs)
   gamma =1.1
   for i in range(imgs.shape[0]):
     if(i%100==0):
       print(str(i)+" finished.")
     img=imgs[i]
     row, col, channel = img.shape
     img_out = img * 1.0
     R=(min(row, col)/2)
     center_x = (col-1)/2.0
     center_y = (row-1)/2.0
     xx = np.arange (col) 
     yy = np.arange (row)
     x_mask = numpy.matlib.repmat (xx, row, 1)
     y_mask = numpy.matlib.repmat (yy, col, 1)
     y_mask = np.transpose(y_mask)
     xx_dif = x_mask - center_x
     yy_dif = center_y - y_mask
     r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
     theta = np.arctan(yy_dif / xx_dif)
     mask_1 = xx_dif < 0
     theta = theta * (1 - mask_1) + (theta + math.pi) * mask_1
     r_new = R*np.power(r/R, gamma)
     x_new = r_new * np.cos(theta) + center_x
     y_new = center_y - r_new * np.sin(theta) 
     int_x = np.floor (x_new)
     int_x = int_x.astype(int)
     int_y = np.floor (y_new)
     int_y = int_y.astype(int)
     for ii in range(row):
       for jj in range (col):
         new_xx = int_x [ii, jj]
         new_yy = int_y [ii, jj]
         if x_new [ii, jj] < 0 or x_new [ii, jj] > col -1 :
           continue
         if y_new [ii, jj] < 0 or y_new [ii, jj] > row -1 :
           continue
         img_out[ii, jj, :] = img[new_yy, new_xx, :]
     img_temp[i]=img_out
   return img_temp.astype(np.float32)

def fisheye_in(imgs):
   img_temp=copy.deepcopy(imgs)
   gamma =0.9
   for i in range(imgs.shape[0]):
     if(i%100==0):
       print(str(i)+" finished.")
     img=imgs[i]
     row, col, channel = img.shape
     img_out = img * 1.0
     R=(min(row, col)/2)
     center_x = (col-1)/2.0
     center_y = (row-1)/2.0
     xx = np.arange (col) 
     yy = np.arange (row)
     x_mask = numpy.matlib.repmat (xx, row, 1)
     y_mask = numpy.matlib.repmat (yy, col, 1)
     y_mask = np.transpose(y_mask)
     xx_dif = x_mask - center_x
     yy_dif = center_y - y_mask
     r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
     theta = np.arctan(yy_dif / xx_dif)
     mask_1 = xx_dif < 0
     theta = theta * (1 - mask_1) + (theta + math.pi) * mask_1
     r_new = R*np.power(r/R, gamma)
     x_new = r_new * np.cos(theta) + center_x
     y_new = center_y - r_new * np.sin(theta) 
     int_x = np.floor (x_new)
     int_x = int_x.astype(int)
     int_y = np.floor (y_new)
     int_y = int_y.astype(int)
     for ii in range(row):
       for jj in range (col):
         new_xx = int_x [ii, jj]
         new_yy = int_y [ii, jj]
         if x_new [ii, jj] < 0 or x_new [ii, jj] > col -1 :
           continue
         if y_new [ii, jj] < 0 or y_new [ii, jj] > row -1 :
           continue
         img_out[ii, jj, :] = img[new_yy, new_xx, :]
     img_temp[i]=img_out
   return img_temp.astype(np.float32)
  
def Wave(imgs):

  img_temp=copy.deepcopy(imgs)
  A=2
  B=1
  for i in range(imgs.shape[0]):
    img=imgs[i]
    row, col, channel = img.shape
    img_out = img * 1.0
    if(i%100==0):
      print(str(i)+" finished.")
    center_x = (col-1)/2.0
    center_y = (row-1)/2.0

    xx = np.arange (col) 
    yy = np.arange (row)

    x_mask = numpy.matlib.repmat (xx, row, 1)
    y_mask = numpy.matlib.repmat (yy, col, 1)
    y_mask = np.transpose(y_mask)

    xx_dif = x_mask - center_x
    yy_dif = center_y - y_mask

    theta = np.arctan2(yy_dif,  xx_dif)
    r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
    r1 = r + A*col*0.01*np.sin(B*0.1*r)

    x_new = r1 * np.cos(theta) + center_x
    y_new = center_y - r1 * np.sin(theta) 

    int_x = np.floor (x_new)
    int_x = int_x.astype(int)
    int_y = np.floor (y_new)
    int_y = int_y.astype(int)

    for ii in range(row):
        for jj in range (col):
            new_xx = int_x [ii, jj]
            new_yy = int_y [ii, jj]

            if x_new [ii, jj] < 0 or x_new [ii, jj] > col -1 :
                continue
            if y_new [ii, jj] < 0 or y_new [ii, jj] > row -1 :
                continue
            img_out[ii, jj, :] = img[new_yy, new_xx, :]
    img_temp[i]=img_out
  return img_temp.astype(np.float32)

'''
zhenyu's code
'''
def TwirlDeal(img, Num):
    img1= img.copy()
    row, col, channel = img1.shape
    xx = np.arange(col)
    yy = np.arange(row)
    x_mask = numpy.matlib.repmat(xx, row, 1)
    y_mask = numpy.matlib.repmat(yy, col, 1)
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

        img1[:, j, :] = img[int_y, int_x, :].sum(axis=1) / Num
    return img1

def meanBlur(images):
    images_new = np.zeros(shape=images.shape)
    print (images.shape)
    for l in range(images.shape[0]):#images.shape[1]
        src = images[l]
        # print ("ddd:",src.shape,type(src))
        src1 = cv.blur(src, (3, 3))  # 
        src1.resize((224, 224, 3))
        images_new[l]= src1
    return images_new.astype(np.float32)

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

def motionBlurDeal(img, size):
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    # applying the kernel to the input image
    output = cv.filter2D(img, -1, kernel_motion_blur)
    return output

def MotionBlur(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):  # images.shape[1]
        src = images[l]
        src1 = motionBlurDeal(src, 4)
        src1.resize((224, 224, 3))
        images_new[l] = src1
    return images_new.astype(np.float32)

'''
xurong's code
'''

def rotate_image_30(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):
        rows, cols, channel = images[l].shape
        M = cv.getRotationMatrix2D((cols/2, rows/2), 30, 1)
        images_new[l] = cv.warpAffine(images[l], M, (cols, rows))
        
    return images_new.astype(np.float32)

def rotate_image_60(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):
        rows, cols, channel = images[l].shape
        M = cv.getRotationMatrix2D((cols/2, rows/2), 60, 1)
        images_new[l] = cv.warpAffine(images[l], M, (cols, rows))
        
    return images_new.astype(np.float32)

def rotate_image_90(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):
        rows, cols, channel = images[l].shape
        M = cv.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        images_new[l] = cv.warpAffine(images[l], M, (cols, rows))
        
    return images_new.astype(np.float32)



def shift_image(images,x,y):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):
        M = np.float32([[1, 0, x], [0, 1, y]])
        images_new[l] = cv.warpAffine(images[l], M, (images[l].shape[1], images[l].shape[0]))
    return images_new.astype(np.float32)

def transfer_feature_LR(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):
        images_new[l] = cv.flip(images[l],1)## Flipped Horizontally
    return images_new.astype(np.float32)
                
def transfer_feature_TB(images):
    images_new = np.zeros(shape=images.shape)
    for l in range(images.shape[0]):
        images_new[l]=cv.flip(images[l],0)#Flipped Vertically
        
    return images_new.astype(np.float32)

def add_gaussian_Noise_feature(images):
    images_new = np.zeros(shape=images.shape)
    temp = images*255.0
    for l in range(images.shape[0]):
        images_new[l]= skimage.util.random_noise(temp[l].astype("uint8"),"gaussian")
    return images_new.astype(np.float32)

def add_localvar_Noise_feature(images):
    images_new = np.zeros(shape=images.shape)
    temp = images*255.0
    for l in range(images.shape[0]):
        images_new[l] = skimage.util.random_noise(temp[l].astype("uint8"),"localvar")
    return images_new.astype(np.float32)

def add_poisson_Noise_feature(images):
    images_new = np.zeros(shape=images.shape)
    temp = images*255.0
    for l in range(images.shape[0]):
        images_new[l] = skimage.util.random_noise(temp[l].astype("uint8"),"poisson")
    return images_new.astype(np.float32)

def add_salt_Noise_feature(images):
    images_new = np.zeros(shape=images.shape)
    temp = images*255.0
    for l in range(images.shape[0]):
        images_new[l] = skimage.util.random_noise(temp[l].astype("uint8"),"salt")
    return images_new.astype(np.float32)

def add_pepper_Noise_feature(images):
    images_new = np.zeros(shape=images.shape)
    temp = images*255.0
    for l in range(images.shape[0]):
        images_new[l] = skimage.util.random_noise(temp[l].astype("uint8"),"pepper")
    return images_new.astype(np.float32)

def add_sp_Noise_feature(images):
    images_new = np.zeros(shape=images.shape)
    temp = images*255.0
    for l in range(images.shape[0]):
        images_new[l] = skimage.util.random_noise(temp[l].astype("uint8"),"s&p")
    return images_new.astype(np.float32)

def add_speckle_Noise_feature(images):
    images_new = np.zeros(shape=images.shape)
    temp = images*255.0
    for l in range(images.shape[0]):
        images_new[l] = skimage.util.random_noise(temp[l].astype("uint8"),"speckle")
    return images_new.astype(np.float32)

def cropN(imagenpy,N):
    img=Image.fromarray(imagenpy.astype('uint8'))
    box=((224-N)/2,((224-N)/2),(224+N)/2,((224+N)/2))##left,upper,right,down
    img=img.crop(box)
    img = img.resize((224,224))
    img = np.array(img)
    return img
    
def extract_feature(model,images):

    function_name= [meanBlur,transfer_feature_LR,rotate_image_30,#0-2
                    rotate_image_60,rotate_image_90,transfer_feature_TB,twirl,#3-6
                    bilateralBlur,fisheye_out,fisheye_in,#7-9
                    add_gaussian_Noise_feature,add_localvar_Noise_feature,#10-11
                    add_poisson_Noise_feature,add_salt_Noise_feature,#12-13
                    add_pepper_Noise_feature,add_sp_Noise_feature,#14-15
                    add_speckle_Noise_feature,Wave,MotionBlur]#16-18

    total_size = images.shape[0]
    feature_vec = tf.Variable(np.zeros([total_size,1]),dtype=tf.int64)
    # feature_vec = np.zeros([total_size,1])
    # feature_vec = tf.get_variable("fec", shape=[total_size,1], dtype=tf.int64, initializer=tf.zeros_initializer)
    # sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    # sess.run(tf.global_variables_initializer())
    a_ori = model.predict(images)
    labels_ori = tf.argmax(a_ori, 1)

    for index in range(len(function_name)):
            print ("index:",index)
            images_new = tf.py_func(function_name[index],[images],tf.float32)
            images_new.set_shape([total_size, 224, 224, 3])
            # print (images.shape,images_new.shape,type(images_new))
            # print("ok")
            a_new = model.predict(images_new)
            # print(labels_ori)
            labels_new = tf.argmax(a_new, 1)
            # print (labels_new)
            res = tf.cast(tf.equal(labels_ori,labels_new),dtype=tf.int64)
            feature_vec = tf.concat((feature_vec,tf.reshape(res,(total_size,1))),1)
    # print (feature_vec.shape)
    return feature_vec