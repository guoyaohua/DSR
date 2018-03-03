# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:51:07 2018
模型工具类，提供模型训练，模型保存，模型导入操作
@author: John Kwok
"""
import GetDataUtil
import numpy as np
#%%
'''
# 声明训练集，数据集
X_train = []
X_test = []
y_train = []
y_test = []
'''
# 获取 训练/测试 数据集
X_train, X_test, y_train, y_test = GetDataUtil.dataInit()
#%%
# 将标签转换为one_hot稀疏值
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train-1,5)
y_test = np_utils.to_categorical(y_test-1,5)
#%%
#卷积神经网络input需要dim = 4
X_train = X_train[:,np.newaxis,:,:]
X_test = X_test[:,np.newaxis,:,:]
print(X_train.shape)
#%%
import time
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input,Dense
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import TruncatedNormal
from keras import regularizers
from keras.layers.core import Activation,Flatten
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
# 两层神经网络
inputs = Input(shape = (1,30,300),name="Input")
x = Conv2D(filters = 32,
           kernel_size = (1,3),
           strides = (1,2),
           use_bias = False,
           kernel_initializer = TruncatedNormal(stddev=0.0001),
           kernel_regularizer = regularizers.l2(0.01),
           name = "Conv1")(inputs)
x = BatchNormalization(axis = 1,name = "BN1")(x)
x = Activation('tanh',name = 'Tanh1')(x)
x = Conv2D(filters = 16,
           kernel_size = (1,5),
           strides = (1,3),
           use_bias = False,
           kernel_initializer = TruncatedNormal(stddev=0.01),
           kernel_regularizer = regularizers.l2(0.01),
           name = "Conv1")(inputs)
x = BatchNormalization(axis = 1,name = 'BN2')(x)
x = Activation('tanh',name = 'Tanh2')(x)
x = Flatten()(x)

x = Dense(1024,
          kernel_initializer = TruncatedNormal(stddev=0.01),
          kernel_regularizer = regularizers.l2(0.01),
          name = "D1")(x)
x = BatchNormalization(axis = 1,name="BN1")(x)
x = Activation('tanh',name = 'Tanh3')(x)
x = Dense(1024,
          activation='tanh',
          kernel_initializer = TruncatedNormal(stddev=0.01),
          kernel_regularizer = regularizers.l2(0.01),
          name="D2")(x)
out = Dense(5,activation='softmax',name="OutPut")(x)

model = Model(inputs = inputs,outputs = out)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("正在训练网络，请耐心等候......")
startTime = time.clock()
model.fit(X_train,
          y_train,
          batch_size=64,
          epochs=20,
          verbose=1
          )
endTime = time.clock()
print("")
# 绘制模型的结构图 此处还出现点问题，待解决
plot_model(model,
           to_file='model.png',
           show_shapes=True,
           show_layer_names=True)

'''注意：sklearn中的标签都不是稀疏的'''
from sklearn.metrics import classification_report
predict = model.predict(X_test)

print(classification_report(np.argmax(y_test,axis = 1),np.argmax(predict,axis = 1)))
# 打印混淆矩阵
# print(sklearn.metrics.confusion_matrix(np.argmax(y_test,axis = 1),np.argmax(predict,axis = 1)))

#%%
# from keras.models import load_model
# 保存模型
model.save('my_model.h5')







