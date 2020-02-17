
##coding=utf-8
'''

@author: lixurong
'''
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC #支持向量机
from sklearn.linear_model import LinearRegression# 线性回归
from sklearn.ensemble import RandomForestClassifier#随机森林
from sklearn.ensemble import ExtraTreesClassifier #
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn import neighbors ##KNN
from sklearn.neural_network import MLPClassifier#感知机
from sklearn import naive_bayes#朴素贝叶斯
# from sklearn.cross_validation import cross_val_score # K折交叉验证模块
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.externals import joblib
import random
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

def read_feature(para):
    new = np.load(para)
    # new=np.hstack((new[:,:4],new[:,9:17],np.reshape(new[:,19],(10000,1)),np.reshape(new[:,20],(10000,1)),new[:,23:]))
    # print (new.shape)
    return new

def makeTest(img,label,test_start,test_end):
    test_X =np.copy(img[test_start:test_end])
    test_Y =label[test_start:test_end]
    return test_X,test_Y

if __name__=="__main__":

    models = [SVC,LinearRegression,RandomForestClassifier,ExtraTreesClassifier,DecisionTreeClassifier,MLPClassifier,RFE,AdaBoostClassifier,GradientBoostingClassifier]
    models_string = ["SVC", "LinearRegression", "RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier",
              "MLPClassifier", "RFE", "AdaBoostClassifier", "GradientBoostingClassifier"]
    for i in range(6):
        model = models[i]()#GradientBoostingClassifier(n_estimators=500)
        print(models_string[i])
        label = np.zeros(2000)
        label[1000:] = 0
       # 前1000个是对抗，为1,2,3
       # 后1000个是原始，为0

        name_all_match = ["clean", "untarget_CWk0", "untarget_CWk10", "untarget_CWk20", "untarget_CWk30", "untarget_CWk40",
                          "FGSM", "IGSM", "LL", "NEXT", "deepfool", "tgsm_ll", "tgsm_next_8","tgsm_next_16"]
        train_X1 = np.load("./features/1000-vggres-same-image_feature.npy")
        # train_X1=np.hstack((train_X1[:,:4],
        #                    train_X1[:,9:17],
        #                     np.reshape(train_X1[:, 19], (10000, 1)),np.reshape(train_X1[:,20],(10000,1)),train_X1[:,23:]))
        # train_X2 = np.hstack((train_X2[:,:4],
        #                        train_X2[:, 9:17],
        #                       np.reshape(train_X2[:, 19],(10000,1)), np.reshape(train_X2[:,20],(10000,1)), train_X2[:, 23:]))
        # train_X3 = np.hstack((train_X3[:,:4],
        #                        train_X3[:, 9:17],
        #                       np.reshape(train_X3[:, 19], (10000, 1)), np.reshape(train_X3[:,20],(10000,1)), train_X3[:, 23:]))
        #
        Feature_dir = "./features/"
        label_dir = "./label/"
        File_list = os.listdir(Feature_dir)

        x = 100
        train_X=train_X1[x:1000]
        train_Y=label[x+1000:]

        x = 800
        for file in File_list:
            if(file.split("_")[0]==".DS"):
                 continue
            file_path=Feature_dir+file
            label_path= label_dir+"ML_labels_"+file.split("_")[0]+".npy"
            if(file.split("-")[0]=="CW"):
                label[:1000] = (1-np.load(label_path))*2
            elif(file.split("-")[0]=="FGSM"):
                label[:1000] = (1-np.load(label_path))*3
            elif(file.split("-")[0]=="IGSM"):
                label[:1000] = (1-np.load(label_path))*4
            elif(file.split("-")[0]=="EAD"):
                label[:1000] = (1-np.load(label_path))*5
            train_X2=np.load(file_path)
            train_X = np.vstack((train_X,train_X2[x:1000]))

            train_Y= np.hstack((train_Y,label[x:1000]))
            # print(train_X.shape)
            #
        model.fit(train_X, train_Y)
        print("train finish")

        for file in File_list:
            if(file.split("_")[0]==".DS"):
                 continue
            file_path=Feature_dir+file
            label_path= label_dir+"ML_labels_"+file.split("_")[0]+".npy"
            if(file.split("-")[0]=="CW"):
                label[:1000] = (1-np.load(label_path))*2
            elif(file.split("-")[0]=="FGSM"):
                label[:1000] = (1-np.load(label_path))*3
            elif(file.split("-")[0]=="IGSM"):
                label[:1000] = (1-np.load(label_path))*4
            elif(file.split("-")[0]=="EAD"):
                label[:1000] = (1-np.load(label_path))*5
            else:
                label[:1000] = (1-np.load(label_path))
            train_X2=np.load(file_path)

            test_X,test_Y = makeTest(train_X2, label[:1000], 0, x)

            print(file.split("-")[0], model.score(test_X, test_Y))