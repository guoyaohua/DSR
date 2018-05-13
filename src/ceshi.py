#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:53:33 2018

@author: gyhu
"""

#%%
import numpy as np
import SCNN_Layer
import time
from keras.layers import Input,Flatten,Dense,Recurrent
from keras.optimizers import Adam
from keras.models import Model

x_train = np.random.randn(64,3,128,128)
y = np.random.randint(5,size = 64)



'''将标签转换为one_hot稀疏值'''
from keras.utils import np_utils
label = np_utils.to_categorical(y,5)



inputs = Input(shape = (3,128,128),name="Input")


x = SCNN_Layer.SCNN_Layer()(inputs)
x = Flatten()(x)
out = Dense(5,activation='softmax',name="OutPut")(x)



model = Model(inputs = inputs,outputs = out)
model.compile(optimizer = 'Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("正在训练网络，请耐心等候......")

#print(x_train.shape)
startTime = time.clock()
trainLog = model.fit(x_train,
          label,
          validation_split = 0.1,
          batch_size=16,
          epochs=10,
          verbose=1
          )
endTime = time.clock()
# 注意，这里的时间window和linux不同
print("网络训练已完成 耗时%f 秒"%((float)(endTime - startTime)/10))







#%%
