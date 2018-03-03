# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 17:51:24 2018

@author: John Kwok
"""

import GetDataUtil
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing

#%%
'''
第一步：将原始数据文件整合保存在npy数组文件中。得到“DataSet.npy”
'''
GetDataUtil.saveDataToNP("DataSet/trim")

#%%
'''
第二步：使用插值法，使原始数据长度对齐，默认采用300为目标长度（即时间窗口3s）。
生成文件“JustifiedData.npy”，并且返回data
[data]：字典集合，{"Acc":A,"Gyr":G,"Label":label}
'''
data = GetDataUtil.interpolation(np.load("DataSet.npy"))

#%%
'''第三步：获取数据，并且归一化、标准化'''
X_train, X_test, y_train, y_test = GetDataUtil.getTrainTestSet()
# 数据预处理（此处只使用了归一化、标准化，Flatten）
# 后续研究可以加入其他预处理：1.滤波 2.稀疏滤波 3.特征升维 等
X_train = preprocessing.scale(X_train.reshape(-1,1800))
X_test = preprocessing.scale(X_test.reshape(-1,1800))

#%%
#numpy测试
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print(a.shape)
print(a)
b = a.reshape(-1,6)
print(b.shape)
print(b)
c = b.reshape(2,3)
print(c)

#%%
'''贝叶斯分类器（高斯）'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(classification_report(y_test, y_pred))


#%%
'''贝叶斯分类器'''
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
print(classification_report(y_test, y_pred))

#%%
'''SVM模型'''
from sklearn import svm
SVM = svm.SVC()
SVM.fit(X_train, y_train)
SVM_labels = SVM.predict(X_test)
print(classification_report(y_test, SVM_labels))

#%%
'''测试画图'''
import math
import matplotlib.pyplot as plt
x = np.arange(10)
print(x)
y = x+1
plt.plot(x,y)
plt.show()
#%%
'''程序二：使用简单全连接神经网络进行训练'''
import GetDataUtil
import numpy as np
from sklearn.metrics import classification_report
from sklearn import preprocessing
'''获取数据，并且归一化、标准化'''
X_train, X_test, y_train, y_test = GetDataUtil.getTrainTestSet()
# 数据预处理
X_train = preprocessing.scale(X_train.reshape(-1,1800))
X_test = preprocessing.scale(X_test.reshape(-1,1800))
#
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train-1,5)
y_test = np_utils.to_categorical(y_test-1,5)
#%%

from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.normalization import BatchNormalization
a = Input(shape = (1800,),name="input_layer")

b = Dense(1024,activation='relu',name="D1")(a)
bb = BatchNormalization(axis = 1)(b)

c = Dense(1024,activation='relu',name="D2")(bb)

out = Dense(5,activation='softmax',name="out_layer")(c)
model = Model(inputs = a,outputs = out)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,
          batch_size=64,
          epochs=10,
          verbose=1
          )
#%%
import sklearn
predict = model.predict(X_test)
print(sklearn.metrics.confusion_matrix(np.argmax(y_test,axis=1), np.argmax(predict,axis = 1)))
#%%
def convertData(filePath = "DataSet/trim/右转/1513994245"):
    return GetDataUtil.getAandG(GetDataUtil.readFile(filePath))
#%%
%timeit result = convertData("DataSet/trim/加速/1513496477")
#%%
%timeit predict = model.predict(X_test)

#%%
#测试numpy广播
import numpy as np
a = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])
print(a.shape)
b = np.array([[1],[2]])
b = b[:,:,np.newaxis]
#b = b.repeat(3,axis = 1)

print(b)
print(b.shape)
a = a/b
print(a)

#%%
from sklearn.model_selection import train_test_split
def getTrainTestSet(dataPath = "JustifiedData.npy",test_size = 0.1):
    X = []
    Y = []
    dataSet = np.load(dataPath)
    for data in dataSet:
        S = data["Acc"]
        S = np.concatenate((S,data["Gyr"]),axis = 0)
        #S = S.reshape((6,300))
        X.append(S)
        Y.append(data["Label"])
#     print(np.array(X).shape)
#     print(np.array(Y).shape)
    X = np.array(X)
    Y = np.array(Y)
#     X_train, X_test, y_train, y_test
    return train_test_split(X,Y,test_size = test_size,random_state = 0)
X_train, X_test, y_train, y_test = getTrainTestSet()
print(X_train.shape)
print(X_test.shape)
#%%
import numpy as np
a = np.load("DataSet.npy")

print(a[0]["Acc"].shape)
b = np.load("JustifiedData.npy")
print(b[1]["Gyr"].shape)

'''
数据预处理函数
1.特征构造
2.标准化
'''
def dataProcess(data):
    print("数据处理中，请稍后...")
    # 特征构造
    x_2 = np.power(data, 2)
    x_3 = np.power(data, 3)
    sin = np.sin(data)
    cos = np.cos(data)
    data = np.concatenate((data,x_2,x_3,sin,cos),axis = 1)
    print("特征构造完毕！")
    # 标准化
    #print(data.shape)
    mean = np.mean(data,axis = (1,2))
    std = np.std(data,axis = (1,2))
    mean = mean[:,np.newaxis,np.newaxis]
    std = std[:,np.newaxis,np.newaxis]
    data = (data-mean)/std
    print("数据标准化完毕！")
    print("数据形状：")
    print(data.shape)
    return data
print(X_train.shape)
X_train = dataProcess(X_train)
print(X_train.shape)