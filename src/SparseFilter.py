# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:20:51 2018
稀疏滤波器
@author: John Kwok
"""
#%%
import numpy as np
import tensorflow as tf
class SparseFilter:
    '稀疏滤波器类'

    '''
    构造函数：初始化各个参数，以及相关tf运算
    X-输入的数据，每个列向量为一个样本，行数为样本原始特征数，列数为样本数
    D-目标特征数
    '''
    def __init__(self,X,D):
        self.X = X
      #  print(X.shape[0])

        self.W = tf.Variable(tf.truncated_normal(shape=(X.shape[0],D),
                                                 stddev=0.1,
                                                 dtype=tf.float64),
                             name="SF_W")
        print(self.W.shape)
        F = tf.matmul(self.W,self.X,transpose_a = True)
        Fs = tf.sqrt(tf.square(F) + 1e-8)
        #Fs = tf.log(tf.square(F) + 1)
        L2Row = tf.norm(Fs,
                        ord=2,
                        axis = 1,
                        keep_dims=True)
        Fs = tf.div(Fs,L2Row)
        L2Col = tf.norm(Fs,
                        ord = 2,
                        axis = 0,
                        keep_dims=True)
        Fs = tf.div(Fs,L2Col)
        self.cost = tf.norm(Fs,
                            ord=1,
                            keep_dims=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=100,
                                          beta1=0.9,
                                          beta2=0.999,
                                          epsilon=1e-08,
                                          use_locking=False,
                                          name='Adam').minimize(self.cost)

    '''训练函数：用于训练'''
    def train(self,numIter):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(numIter):
                sess.run(self.opt)
                print(sess.run(self.cost))
            return sess.run(self.W)            
        
                    
#%%
            
#%%
'''
载入文件插值、对齐好的数据文件，并返回训练集、测试集
参数：
    dataPath 数据文件地址
    test_size 测试集占比
输出：
    X_train,X_test, y_train, y_test
    shape = (6,300)
'''
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
'''
数据预处理函数
1.特征构造
2.标准化
'''
def dataProcess(data,logPrint = True):
    if(logPrint):
        print("processing...")
        '''
    # 特征构造
    x_2 = np.power(data, 2)
    x_3 = np.power(data, 3)
    sin = np.sin(data)
    cos = np.cos(data)
    data = np.concatenate((data,x_2,x_3,sin,cos),axis = 1)
    if(logPrint):
        print("特征构造完毕！")
        '''
    # 标准化
    #print(data.shape)
    mean = np.mean(data,axis = (1,2))
    #std = np.std(data,axis = (1,2))
    mean = mean[:,np.newaxis,np.newaxis]
    #std = std[:,np.newaxis,np.newaxis]
    #data = (data-mean)/std
    data = data-mean
    if(logPrint):
        #print("数据标准化完毕！(未去标准差)")
        #print("数据形状：")
        print(data.shape)
    return data


X_train, X_test, y_train, y_test = getTrainTestSet()
X_train = dataProcess(X_train)
X_test = dataProcess(X_test)
#print("数据初始化完毕！")

#%%

X = X_train.reshape(-1,1800).T
sf = SparseFilter(X,32*32)
W = sf.train(100)
a = np.transpose(W).dot(X)
np.save("SFW.npy",W)
#%%
print("finished")
print(W.shape)
        
       



 
#%%
'''
# 测试
#X = np.random.randn(20,10)
sf = SparseFilter(X,32*32)
W = sf.train(100)

a = np.transpose(W).dot(X)
print(a.shape)
#%%
print(np.transpose(W).dtype)
                

#%%
'''
'''
tf.matmul()  为矩阵乘法

tf.multiply() 为矩阵点乘

np.dot() 为矩阵乘法

np.multiply() 为矩阵点乘
'''
'''
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3],[4,5,6]])
c = np.array([[1,2],[3,4],[3,4]])
print(np.dot(a,c))
print(np.multiply(a,b))
'''