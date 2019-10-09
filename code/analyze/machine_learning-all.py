
##coding=utf-8
'''

@author: lixurong
'''
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC 
from sklearn.linear_model import LinearRegression# 
from sklearn.ensemble import RandomForestClassifier#
from sklearn.ensemble import ExtraTreesClassifier #
from sklearn.tree import DecisionTreeClassifier#
from sklearn import neighbors ##KNN
from sklearn.neural_network import MLPClassifier#
from sklearn import naive_bayes#
#from sklearn.cross_validation import cross_val_score #
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.externals import joblib
import os
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
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
    model = joblib.load("./ML_models/model_RF.m")
    fdir="./modified_result/38_features/"
    fs=["38F_IGSM-Res50-Slim-20.npy"]
    for i in range(len(fs)):
        temp=np.load(fdir+fs[i])
        labels=np.load("./modified_result/adv_mark/advmark_"+fs[i].split("_")[1])
        # labels = np.load("./results/ML_label.npy")
        print("===================")

        # for h in range(5):
        #     a = random.randint(0, 37)
        #     temp[:, a] = clean[:, a]
        print("test: ", fs[i])
        test_X, test_Y = makeTest(temp, labels, 0, 1000)
        scores=model.score(test_X, test_Y)
        print(test_X.shape, test_Y.shape)
        print("TP:")
        print("random:",scores)
        f=open("black_machine.txt","a+")
        f.write(fs[i].split(".")[0]+"-"+str(scores)+"\n")
        f.close()

    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores.mean())



