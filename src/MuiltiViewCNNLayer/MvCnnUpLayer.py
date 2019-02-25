# -*- coding: utf-8 -*-

import tensorflow as tf

class MvCnnUpLayer(tf.keras.layers.Layer):
  '''
    Muiti-View-CNN-Up 向上传递信息

    状态释义：
        state = 1：(when i == H-1 && H > 2)
            无slice_bottom,存在slice_conv,slice_add,slice_top
        state = 2: (when i == 1 && H > 2)
            无slice_top，存在slice_bottom,slice_conv,slice_add
        state = 3: (when i == 0 && H > 1)
            无slice_add,slice_top，存在slice_bottom,slice_conv
        state = 4: (when H == 1)
            无slice_top,slice_add,slice_bottom，存在slice_conv
        state = 5: (when i == H-1 && H == 2)
            无slice_top,slice_bottom，存在slice_conv,slice_add
        state = 6:
            存在slice_top,slice_conv,slice_add,slice_bottom
  '''
  def __init__(self,SCNN_KERNEL_LENGTH, **kwargs):
    # SCNN 卷集核的宽度
    self.SCNN_KERNEL_LENGTH = SCNN_KERNEL_LENGTH
    # 切片后需要pad的个数
    PAD = SCNN_KERNEL_LENGTH - 1  
    # 切片需要后需要PAD
    self.PADDING = [[0, 0],
                    [0, 0],
                    [int(PAD / 2), PAD - int(PAD/2)],
                    [0, 0]]
    super(MvCnnUpLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    # 声明参数
    self.w = self.add_weight(name='w_mvcnn_u',
                        shape=tf.TensorShape(([input_shape.as_list()[3], self.SCNN_KERNEL_LENGTH, 1, input_shape.as_list()[3]])),
                        initializer='random_uniform',
                        regularizer=tf.keras.regularizers.l2(),
                        trainable=True)
    self.b = self.add_weight(name="b_mvcnn_u",
                            shape=tf.TensorShape((input_shape.as_list()[3])),
                            initializer=tf.keras.initializers.zeros(),
                            trainable=True)
                          
    # Be sure to call this at the end
    super(MvCnnUpLayer, self).build(input_shape)

  def call(self, inputs):
    FEATURE_MAP_H = inputs.shape[1]
    # 用于动态生成网络层
    MVCNN_U = {}

    for i in range(FEATURE_MAP_H-1, -1, -1):
        # State 1 : 无slice_bottom,存在slice_conv,slice_add,slice_top
        if i == FEATURE_MAP_H-1 and FEATURE_MAP_H > 2:
            MVCNN_U['slice_conv_'+str(i)] = tf.slice(inputs,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            MVCNN_U['slice_add_'+str(i)] = tf.slice(inputs,
                                                   begin=[0, i-1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            MVCNN_U['slice_top_'+str(i)] = tf.slice(inputs,
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i-1, -1, -1])
            MVCNN_U['tran_bef_'+str(i)] = tf.transpose(MVCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            MVCNN_U['pad_'+str(i)] = tf.pad(MVCNN_U['tran_bef_'+str(i)],
                                           paddings=self.PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            MVCNN_U['conv_'+str(i)] = tf.nn.conv2d(MVCNN_U['pad_'+str(i)],
                                                  filter=self.w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            MVCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(MVCNN_U['conv_'+str(i)], self.b),
                                                          name=None)
            MVCNN_U['add_'+str(i)] = tf.add(MVCNN_U['conv_bias_relu_'+str(i)],
                                           MVCNN_U['slice_add_'+str(i)])
            MVCNN_U['concat_'+str(i)] = tf.concat(values=[MVCNN_U['slice_top_'+str(i)],
                                                         MVCNN_U['add_' +
                                                                str(i)],
                                                         MVCNN_U['conv_bias_relu_'+str(i)]], axis=1)
        # State 2 : 无slice_top，存在slice_bottom,slice_conv,slice_add
        elif i == 1 and FEATURE_MAP_H > 2:
            MVCNN_U['slice_add_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            MVCNN_U['slice_conv_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            MVCNN_U['slice_bottom_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            MVCNN_U['tran_bef_'+str(i)] = tf.transpose(MVCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            MVCNN_U['pad_'+str(i)] = tf.pad(MVCNN_U['tran_bef_'+str(i)],
                                           paddings=self.PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            MVCNN_U['conv_'+str(i)] = tf.nn.conv2d(MVCNN_U['pad_'+str(i)],
                                                  filter=self.w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            MVCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(MVCNN_U['conv_'+str(i)], self.b),
                                                          name=None)
            MVCNN_U['add_'+str(i)] = tf.add(MVCNN_U['conv_bias_relu_'+str(i)],
                                           MVCNN_U['slice_add_'+str(i)])
            MVCNN_U['concat_'+str(i)] = tf.concat(values=[MVCNN_U['add_'+str(i)],
                                                         MVCNN_U['conv_bias_relu_' +
                                                                str(i)],
                                                         MVCNN_U['slice_bottom_'+str(i)]], axis=1)
        # State 3 : 无slice_add,slice_top，存在slice_bottom,slice_conv
        elif i == 0 and FEATURE_MAP_H > 1:
            MVCNN_U['slice_conv_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            MVCNN_U['slice_bottom_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            MVCNN_U['tran_bef_'+str(i)] = tf.transpose(MVCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            MVCNN_U['pad_'+str(i)] = tf.pad(MVCNN_U['tran_bef_'+str(i)],
                                           paddings=self.PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            MVCNN_U['conv_'+str(i)] = tf.nn.conv2d(MVCNN_U['pad_'+str(i)],
                                                  filter=self.w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            MVCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(MVCNN_U['conv_'+str(i)], self.b),
                                                          name=None)
            MVCNN_U['concat_'+str(i)] = tf.concat(values=[MVCNN_U['conv_bias_relu_'+str(i)],
                                                         MVCNN_U['slice_bottom_'+str(i)]], axis=1)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_H == 1:
            MVCNN_U['slice_conv_'+str(i)] = tf.slice(inputs,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, -1, -1, -1])
            MVCNN_U['tran_bef_'+str(i)] = tf.transpose(MVCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            MVCNN_U['pad_'+str(i)] = tf.pad(MVCNN_U['tran_bef_'+str(i)],
                                           paddings=self.PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            MVCNN_U['conv_'+str(i)] = tf.nn.conv2d(MVCNN_U['pad_'+str(i)],
                                                  filter=self.w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            MVCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(MVCNN_U['conv_'+str(i)], self.b),
                                                          name=None)
            MVCNN_U['concat_'+str(i)] = MVCNN_U['conv_bias_relu_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == FEATURE_MAP_H-1 and FEATURE_MAP_H == 2:
            MVCNN_U['slice_add_'+str(i)] = tf.slice(inputs,
                                                   begin=[0, i, 0, 0],
                                                   size=[-1, 1, -1, -1])
            MVCNN_U['slice_conv_'+str(i)] = tf.slice(inputs,
                                                    begin=[0, i+1, 0, 0],
                                                    size=[-1, 1, -1, -1])
            MVCNN_U['tran_bef_'+str(i)] = tf.transpose(MVCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            MVCNN_U['pad_'+str(i)] = tf.pad(MVCNN_U['tran_bef_'+str(i)],
                                           paddings=self.PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            MVCNN_U['conv_'+str(i)] = tf.nn.conv2d(MVCNN_U['pad_'+str(i)],
                                                  filter=self.w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            MVCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(MVCNN_U['conv_'+str(i)], self.b),
                                                          name=None)
            MVCNN_U['add_'+str(i)] = tf.add(MVCNN_U['conv_bias_relu_'+str(i)],
                                           MVCNN_U['slice_add_'+str(i)])
            MVCNN_U['concat_'+str(i)] = tf.concat(values=[MVCNN_U['add_'+str(i)],
                                                         MVCNN_U['conv_bias_relu_'+str(i)]], axis=1)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            MVCNN_U['slice_top_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i-1, -1, -1])
            MVCNN_U['slice_add_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                   begin=[0, i-1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            MVCNN_U['slice_conv_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            MVCNN_U['slice_bottom_'+str(i)] = tf.slice(MVCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            MVCNN_U['tran_bef_'+str(i)] = tf.transpose(MVCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            MVCNN_U['pad_'+str(i)] = tf.pad(MVCNN_U['tran_bef_'+str(i)],
                                           paddings=self.PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            MVCNN_U['conv_'+str(i)] = tf.nn.conv2d(MVCNN_U['pad_'+str(i)],
                                                  filter=self.w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            MVCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(MVCNN_U['conv_'+str(i)], self.b),
                                                          name=None)
            MVCNN_U['add_'+str(i)] = tf.add(MVCNN_U['conv_bias_relu_'+str(i)],
                                           MVCNN_U['slice_add_'+str(i)])
            MVCNN_U['concat_'+str(i)] = tf.concat(values=[MVCNN_U['slice_top_'+str(i)],
                                                         MVCNN_U['add_' +
                                                                str(i)],
                                                         MVCNN_U['conv_bias_relu_' +
                                                                str(i)],
                                                         MVCNN_U['slice_bottom_'+str(i)]], axis=1)
    return MVCNN_U['concat_'+str(0)]


  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    base_config = super(MvCnnUpLayer, self).get_config()
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
