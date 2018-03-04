#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 09:20:22 2018

@author: gyhu
"""
#%%
import numpy as np
import keras
import GetDataUtil

#%%
'''1. Get the train and test dataset.'''
X_train_origin, X_test_origin, y_train, y_test = GetDataUtil.getTrainTestSet(dataPath = "../DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_ACC_005_10000.npy",test_size = 0.1)
#X_train_origin, X_test_origin, y_train, y_test = GetDataUtil.getTrainTestSet(dataPath = "../DataSet_NPSave/JustifiedData.npy",test_size = 0.1)


#%%
'''2. Data preprocessing'''
def DataPreprocess(data):
    print("Data Preprocessing,Please wait...")
    '''
    # 加速度归一化 0~1
    
    data[:,:3,:] /= 16
    data[:,:3,:] *= 32768.0
    data[:,:3,:] += 32768.0
    data[:,:3,:] /= 65535.0
    
    # 角速度归一化 0~1
    data[:,3:6,:] /= 2000
    data[:,3:6,:] *= 32768.0
    data[:,3:6,:] += 32768.0
    data[:,3:6,:] /= 65535.0
    '''
    data[:,:3,:] = (data[:,:3,:] - np.mean(data[:,:3,:]))/np.std(data[:,:3,:])
    data[:,3:6,:] = (data[:,3:6,:] - np.mean(data[:,3:6,:]))/np.std(data[:,3:6,:])
    
    # 特征构造
    sin = np.sin(data * np.pi / 2)
    cos = np.cos(data * np.pi / 2)
    X_2 = np.power(data,2)
    X_3 = np.power(data,3)   
    ACC_All = np.sqrt((np.power(data[:,0,:],2)+
                      np.power(data[:,1,:],2)+
                      np.power(data[:,2,:],2))/3)[:,np.newaxis,:]    
    Ay_Gz = (data[:,1,:] * data[:,5,:])[:,np.newaxis,:]
    Ay_2_Gz = (np.power(data[:,1,:],2) * data[:,5,:])[:,np.newaxis,:]
    Ay_Gz_2 = (np.power(data[:,5,:],2) * data[:,1,:])[:,np.newaxis,:]
    Ax_Gy = (data[:,0,:] * data[:,4,:])[:,np.newaxis,:]
    Ax_2_Gy = (np.power(data[:,0,:],2) * data[:,4,:])[:,np.newaxis,:]
    Ax_Gy_2 = (np.power(data[:,4,:],2) * data[:,0,:])[:,np.newaxis,:]
    
    Ax_Ay_Az = (data[:,0,:]*data[:,1,:]*data[:,2,:])[:,np.newaxis,:]
    
    newData = np.concatenate((data,sin,cos,X_3,X_2,ACC_All,Ay_Gz,Ay_2_Gz,Ay_Gz_2,Ax_Gy,
                           Ax_2_Gy,Ax_Gy_2,Ax_Ay_Az),axis = 1)
    
    # data *= 255
    print(np.min(data))
    print(np.max(data))
    
    print("Finished!")
    return newData

X_train = DataPreprocess(X_train_origin)
X_test = DataPreprocess(X_test_origin)
#%%
'''3. 将标签转换为one_hot稀疏值'''
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train-1,5)
y_test = np_utils.to_categorical(y_test-1,5)

#%%
'''卷积神经网络input需要dim = 4'''
X_train = X_train[:,np.newaxis,:,:]
X_test = X_test[:,np.newaxis,:,:]
# (9000, 1, 38, 300)
print(X_train.shape)

#%%
'''4. 神经网络模型 Labels = ["加速","碰撞","匀速","左转","右转"]'''
import time
import sklearn
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal
from keras import regularizers
from keras.layers.core import Activation,Flatten
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.layers.core import Dropout
# 
inputs = Input(shape = (1,38,300),name="Input")

x = Conv2D(filters = 128,
           kernel_size = (2,2),#尝试方形kernel 纵向也可能相关
           strides = (1,1),
           padding = 'same',
           use_bias = True,# 尝试加一下bias
           kernel_initializer = TruncatedNormal(stddev=0.0001),
           kernel_regularizer = regularizers.l2(0.1),
           name = "Conv1")(inputs)
#x = BatchNormalization(axis = 1,name = "BN1")(x)# 尝试去掉BN
x = Activation('tanh',name = 'relu1')(x)
x = MaxPooling2D(pool_size = (3,3),strides = (1,1),padding = 'same')(x)
# x = Dropout(0.1)(x)#0.1zuoyou
x = Conv2D(filters = 64,
           kernel_size = (3,3),
           strides = (1,1),
           padding = 'same',
           use_bias = True,
           kernel_initializer = TruncatedNormal(stddev=0.001),
           kernel_regularizer = regularizers.l2(0.1),
           name = "Conv2")(x)
# x = BatchNormalization(axis = 1,name = 'BN2')(x)
x = Activation('tanh',name = 'relu2')(x)
x = MaxPooling2D(pool_size = (3,3),strides = (1,1),padding = 'same')(x)
#x = Dropout(0.1)(x)
#
#x = Conv2D(filters = 128,
#           kernel_size = (1,5),
#           strides = (1,3),
#           padding = 'same',
#           use_bias = False,
#           kernel_initializer = TruncatedNormal(stddev=0.001),
#           kernel_regularizer = regularizers.l2(0.01),
#           name = "Conv3")(x)
#x = BatchNormalization(axis = 1,name = "BN3")(x)
#x = Activation('relu',name = 'relu3')(x)
#x = MaxPooling2D(pool_size = (1,3),strides = (1,2),padding = 'same')(x)
#x = Dropout(0.2)(x)

x = Flatten()(x)

x = Dense(1024,
          kernel_initializer = TruncatedNormal(stddev=0.01),
          kernel_regularizer = regularizers.l2(0.01),
#          activation='relu',
          name = "D1")(x)
# x = BatchNormalization(axis = 1,name="D1_BN")(x)
x = Activation('tanh',name = 'D1_relu')(x)
# x = Dropout(0.1)(x)

x = Dense(512,
#          activation='relu',
          kernel_initializer = TruncatedNormal(stddev=0.01),
          kernel_regularizer = regularizers.l2(0.01),
          name="D2")(x)
# x = BatchNormalization(axis = 1,name="D2_BN")(x)
x = Activation('tanh',name = 'D2_relu')(x)

x = Dense(256,
          activation='relu',
          kernel_initializer = TruncatedNormal(stddev=0.1),
          kernel_regularizer = regularizers.l2(0.01),
          name="D3")(x)

# x = BatchNormalization(axis = 1,name="D3_BN")(x)
x = Activation('tanh',name = 'D3_relu')(x)

out = Dense(5,activation='softmax',name="OutPut")(x)

model = Model(inputs = inputs,outputs = out)
model.compile(optimizer = 'Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("正在训练网络，请耐心等候......")
#X_train,y_train = shuffle(X_train,y_train,random_state=None)

startTime = time.clock()
trainLog = model.fit(X_train,
          y_train,
          validation_split = 0.03,
          batch_size=512,
          epochs=10,
          verbose=1
          )
endTime = time.clock()
# 注意，这里的时间window和linux不同
print("网络训练已完成 耗时%f 秒"%((float)(endTime - startTime)/10))

# 绘制模型的结构图 此处还出现点问题，待解a决
plot_model(model,
           to_file='model.png',
           show_shapes=True,
           show_layer_names=True)

'''注意：sklearn中的标签都不是稀疏的'''
from sklearn.metrics import classification_report
predict = model.predict(X_test)
print("测试集预测结果：")
print(classification_report(np.argmax(y_test,axis = 1),np.argmax(predict,axis = 1)))
# 打印混淆矩阵
print("测试集混淆矩阵：")
print(sklearn.metrics.confusion_matrix(np.argmax(y_test,axis = 1),np.argmax(predict,axis = 1)))

#%%
# from keras.models import load_model
# 保存模型
model.save('my_model.h5')

#%%
print(trainLog)
plot_model(model,
           to_file='model.png',
           show_shapes=True,
           show_layer_names=True)

