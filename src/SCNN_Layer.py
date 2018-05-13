#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:09:15 2018

@author: Guo Yaohua
"""

from keras.engine.topology import Layer
import numpy as np
import keras.backend as K
from keras.utils import conv_utils
from keras import regularizers,initializers,activations,constraints
import tensorflow as tf

class SCNN_Layer(Layer):
    
    def __init__(self,oritation = 'VH',
#                 filters, 
                 kernel_size = 9, 
#                 strides=(1, 1), 
#                 padding='valid', 
                 data_format="channels_first", 
                 dilation_rate=(1, 1), # 指定dilated convolution中的膨胀比例
                 activation=None, 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None,
                 **kwargs):
        print("init")
        self.oritation = oritation
        self.kernel_size = kernel_size
        #self.data_format = data_format
        self.use_bias = use_bias
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        #self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        
        super(SCNN_Layer,self).__init__(**kwargs)
        
    def build(self,input_shape):
        
        print("build")
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        self.input_chennel = input_shape[channel_axis]
        # 定义卷集核大小形状,此处的1代表每次切一片
        kernel_shape = (self.input_chennel,self.kernel_size) + (1, self.input_chennel)
        print(kernel_shape)
        # 定义四个方向的卷集核
        self.kernel_D= self.add_weight(shape = kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel_D',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
##        self.kernel_U= self.add_weight(shape=kernel_shape,
##                                      initializer=self.kernel_initializer,
##                                      name='kernel_U',
##                                      regularizer=self.kernel_regularizer,
##                                      constraint=self.kernel_constraint)
##        self.kernel_R= self.add_weight(shape=kernel_shape,
##                                      initializer=self.kernel_initializer,
##                                      name='kernel_R',
##                                      regularizer=self.kernel_regularizer,
##                                      constraint=self.kernel_constraint)
##        self.kernel_L= self.add_weight(shape=kernel_shape,
##                                      initializer=self.kernel_initializer,
##                                      name='kernel_L',
##                                      regularizer=self.kernel_regularizer,
##                                      constraint=self.kernel_constraint)
#        
#        
#        
#        # 定义四个方向的bias
        if self.use_bias:
            self.bias_D = self.add_weight(shape=(self.input_chennel,),
                                        initializer=self.bias_initializer,
                                        name='bias_D',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            print(self.bias_D.shape)
##            self.bias_U = self.add_weight(shape=(input_shape[1],),
##                                        initializer=self.bias_initializer,
##                                        name='bias_U',
##                                        regularizer=self.bias_regularizer,
##                                        constraint=self.bias_constraint)
##            self.bias_R = self.add_weight(shape=(input_shape[1],),
##                                        initializer=self.bias_initializer,
##                                        name='bias_R',
##                                        regularizer=self.bias_regularizer,
##                                        constraint=self.bias_constraint)
##            self.bias_L = self.add_weight(shape=(input_shape[1],),
##                                        initializer=self.bias_initializer,
##                                        name='bias_L',
##                                        regularizer=self.bias_regularizer,
##                                        constraint=self.bias_constraint)
        else:
            self.bias = None
#        
        super(SCNN_Layer, self).build(input_shape) # Be sure to call this somewhere!
        
        
        def call(self,x):
#            # 1. 沿H方向纵向切片 B C H W
#            # down
            for i in range(x.shape[2]):
                print(i)
                x_slice = x[:,:,i,:] # B*C*W
#
                x_slice = tf.transpose(x_slice,(0,2,1))# B*W*C
                
                x_slice = K.asymmetric_temporal_padding(x_slice, left_pad=4, right_pad=4) # B*W*C
                
                x_slice = x_slice[:,np.newaxis,:,:] 

                x_slice = tf.transpose(x_slice,(0,1,3,2)) # B 1 C W
                conv_out = K.conv2d(x_slice,
                                    self.kernel_D,
                                    strides=(1,1),
                                    padding='valid',
                                    data_format=self.data_format,
                                    dilation_rate=self.dilation_rate)
                if self.use_bias:
                    conv_out = K.bias_add(
                        conv_out,
                        self.bias_D,
                        data_format=self.data_format)
        
                if self.activation is not None:
                    conv_out = self.activation(conv_out)
                
                conv_out = tf.reshape(conv_out,(conv_out.shape[0],conv_out.shape[1],conv_out.shape[3]))
                print(conv_out)
                K.update(x[:,:,i,:],conv_out)
                # when is not the last slice
                if i <x.shape[2]-1:
                    K.update_add(x[:,:,i+1,:],conv_out)
            print(x.shape)
#            # up
#            for i in inputs.shape[2]:
#                i = inputs.shape[2]-i-1
#                x_slice = inputs[:,:,i,:] # B*C*W
#                x_slice = np.transpose(x_slice,(0,2,1))# B*W*C
#                x_slice = K.asymmetric_temporal_padding(x_slice, left_pad=4, right_pad=5) # B*W*C
#                x_slice = x_slice[:,np.newaxis,:,:] 
#                x_slice = np.transpose(x_slice,(0,1,3,2)) # B 1 C W
#                
#                conv_out = K.conv2d(x_slice,
#                                    self.kernel_D,
#                                    strides=(1,1),
#                                    padding='valid',
#                                    data_format=self.data_format,
#                                    dilation_rate=self.dilation_rate)
#                if self.use_bias:
#                    conv_out = K.bias_add(
#                        conv_out,
#                        self.bias_U,
#                        data_format=self.data_format)
#        
#                if self.activation is not None:
#                    conv_out = self.activation(conv_out)
#                print(inputs.shape)       
#                conv_out.resize(conv_out.shape[0],conv_out.shape[1],conv_out.shape[3])
#                K.update(inputs[:,:,i,:],conv_out)
#                print(inputs.shape)
#                # when is not the last slice
#                if i < inputs.shape[2]-1:
#                    K.update_add(inputs[:,:,i-1,:],conv_out)         
            return conv_out
    
        def compute_output_shape(input_shape):
            return input_shape
        
