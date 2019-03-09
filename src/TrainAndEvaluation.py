# -*- coding: utf-8 -*-

import argparse
import os
import time
from MuiltiViewCNNLayer.MvCnnLeftLayer import MvCnnLeftLayer
from MuiltiViewCNNLayer.MvCnnRightLayer import MvCnnRightLayer
from MuiltiViewCNNLayer.MvCnnUpLayer import MvCnnUpLayer
from MuiltiViewCNNLayer.MvCnnDownLayer import MvCnnDownLayer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import GetDataUtil

'''
定义超参
'''
BATCH_SIZE = 64  # 批大小
EPOCHS = 10  # 训练EPOCH次数
HIDDEN_UNIT = 512
KERNEL_SIZE = 3  # 通用卷集核大小
COVN_1_CHANNELS = 32  # 第一层卷积层的输出chennel个数
COVN_2_CHANNELS = 16  # 第二层卷积层的输出chennel个数
INPUT_CHENNELS = 3  # 输入图片的chennels个数
SCNN_KERNEL_LENGTH = 9  # SCNN 卷集核的宽度
PAD = SCNN_KERNEL_LENGTH - 1  # 切片后需要pad的个数
# 切片需要后需要PAD
PADDING = [[0, 0],
           [0, 0],
           [int(PAD / 2), PAD - int(PAD/2)],
           [0, 0]]

CONV1_FILTERS = 64
CONV2_FILTERS = 32


def DataPreprocess(data):
    '''
        数据处理，特征构造
    '''
    print("Data Preprocessing,Please wait...")

    data[:, :3, :] = (data[:, :3, :] - np.mean(data[:, :3, :])
                      )/np.std(data[:, :3, :])
    data[:, 3:6, :] = (data[:, 3:6, :] -
                       np.mean(data[:, 3:6, :]))/np.std(data[:, 3:6, :])

    # 特征构造
    sin = np.sin(data * np.pi / 2)
    cos = np.cos(data * np.pi / 2)
    X_2 = np.power(data, 2)
    X_3 = np.power(data, 3)
    ACC_All = np.sqrt((np.power(data[:, 0, :], 2) +
                       np.power(data[:, 1, :], 2) +
                       np.power(data[:, 2, :], 2))/3)[:, np.newaxis, :]
    Ay_Gz = (data[:, 1, :] * data[:, 5, :])[:, np.newaxis, :]
    Ay_2_Gz = (np.power(data[:, 1, :], 2) * data[:, 5, :])[:, np.newaxis, :]
    Ay_Gz_2 = (np.power(data[:, 5, :], 2) * data[:, 1, :])[:, np.newaxis, :]
    Ax_Gy = (data[:, 0, :] * data[:, 4, :])[:, np.newaxis, :]
    Ax_2_Gy = (np.power(data[:, 0, :], 2) * data[:, 4, :])[:, np.newaxis, :]
    Ax_Gy_2 = (np.power(data[:, 4, :], 2) * data[:, 0, :])[:, np.newaxis, :]

    Ax_Ay_Az = (data[:, 0, :]*data[:, 1, :]*data[:, 2, :])[:, np.newaxis, :]

    newData = np.concatenate((data, sin, cos, X_3, X_2, ACC_All, Ay_Gz, Ay_2_Gz, Ay_Gz_2, Ax_Gy,
                              Ax_2_Gy, Ax_Gy_2, Ax_Ay_Az), axis=1)

    print("Data Prosess Finished!")

    return newData


def LoadData():
    '''
        数据加载
    '''
    # 数据读取
    x_train_DA, y_train_DA = GetDataUtil.splitDataAndLabel(
        dataPath="../DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_ACC_005_10000.npy")
    x_train, y_train = GetDataUtil.splitDataAndLabel(
        dataPath="../DataSet_NPSave/Train_Data_Orig.npy")
    x_test, y_test = GetDataUtil.splitDataAndLabel(
        dataPath="../DataSet_NPSave/Test_Data_Orig.npy")

    y_train_DA = y_train_DA-1
    y_train = y_train-1
    y_test = y_test-1

    # 特征构造
    x_train_FA = DataPreprocess(x_train)
    x_test_FA = DataPreprocess(x_test)
    x_train_DA_FA = DataPreprocess(x_train_DA)

    x_train = x_train[:, :, :, np.newaxis]
    x_train_DA = x_train_DA[:, :, :, np.newaxis]
    x_train_FA = x_train_FA[:, :, :, np.newaxis]
    x_train_DA_FA = x_train_DA_FA[:, :, :, np.newaxis]

    x_test = x_test[:, :, :, np.newaxis]
    x_test_FA = x_test_FA[:, :, :, np.newaxis]

    print("Load data Done!")
    print("x_train:", x_train.shape)
    print("x_train_FA:", x_train_FA.shape)
    print("y_train:", y_train.shape)
    print("x_train_DA:", x_train_DA.shape)
    print("x_train_DA_FA:", x_train_DA_FA.shape)
    print("y_train_DA:", y_train_DA.shape)
    print("x_test:", x_test.shape)
    print("x_test_FA:", x_test_FA.shape)
    print("y_test:", y_test.shape)

    return x_train, x_train_FA, x_train_DA, x_train_DA_FA, x_test, x_test_FA, y_train, y_train_DA, y_test

def SCNN_D_F(SCNN_D_input):
    '''
        SCNN_Down 向下传递信息

        状态释义：
            state = 1：(when i == 0 && H > 2)
                无slice_top，存在slice_conv,slice_add,slice_bottom
            state = 2: (when i == H-2 && i > 0)
                无slice_bottom，存在slice_top,slice_conv,slice_add
            state = 3: (when i == H-1 && i > 0)
                无slice_add,slice_bottom，存在slice_top,slice_conv
            state = 4: (when H == 1)
                无slice_top,slice_add,slice_bottom，存在slice_conv
            state = 5: (when i == 0 && H == 2)
                无slice_top,slice_bottom，存在slice_conv,slice_add
            state = 6:
                存在slice_top,slice_conv,slice_add,slice_bottom
    '''
    CHANNELS = SCNN_D_input.shape[3]
    FEATURE_MAP_H = SCNN_D_input.shape[1]
    # 声明参数
    w = tf.get_variable(name='w_scnn_d',
                        shape=[CHANNELS,
                               SCNN_KERNEL_LENGTH,
                               1,
                               CHANNELS],
                        initializer=tf.keras.initializers.zeros(),
                        trainable = False,
                        regularizer=tf.keras.regularizers.l2())
    b = tf.Variable(tf.zeros(CHANNELS),
                    trainable = False,
                    name="b_scnn_d")

    # 用于动态生成网络层
    SCNN_D = {}

    for i in range(FEATURE_MAP_H):
        # State 1 : 无slice_top，存在slice_conv,slice_add,slice_bottom
        if i == 0 and FEATURE_MAP_H > 2:
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['slice_add_'+str(i)] = tf.slice(SCNN_D_input,
                                                   begin=[0, i+1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_D['slice_bottom_'+str(i)] = tf.slice(SCNN_D_input,
                                                      begin=[0, i+2, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['add_'+str(i)] = tf.add(SCNN_D['conv_bias_relu_'+str(i)],
                                           SCNN_D['slice_add_'+str(i)])
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['conv_bias_relu_'+str(i)],
                                                         SCNN_D['add_' +
                                                                str(i)],
                                                         SCNN_D['slice_bottom_'+str(i)]], axis=1)
        # State 2 : 无slice_bottom，存在slice_top,slice_conv,slice_add
        elif i == FEATURE_MAP_H - 2 and i > 0:
            SCNN_D['slice_top_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['slice_add_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, i+1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['add_'+str(i)] = tf.add(SCNN_D['conv_bias_relu_'+str(i)],
                                           SCNN_D['slice_add_'+str(i)])
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['slice_top_'+str(i)],
                                                         SCNN_D['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_D['add_'+str(i)]], axis=1)
        # State 3 : 无slice_add,slice_bottom，存在slice_top,slice_conv
        elif i == FEATURE_MAP_H - 1 and i > 0:
            SCNN_D['slice_top_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['slice_top_'+str(i)],
                                                         SCNN_D['conv_bias_relu_'+str(i)]], axis=1)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_H == 1:
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, -1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['concat_'+str(i)] = SCNN_D['conv_bias_relu_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == 0 and FEATURE_MAP_H == 2:
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['slice_add_'+str(i)] = tf.slice(SCNN_D_input,
                                                   begin=[0, i+1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['add_'+str(i)] = tf.add(SCNN_D['conv_bias_relu_'+str(i)],
                                           SCNN_D['slice_add_'+str(i)])
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['conv_bias_relu_'+str(i)],
                                                         SCNN_D['add_'+str(i)]], axis=1)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            SCNN_D['slice_top_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['slice_add_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, i+1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_D['slice_bottom_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                      begin=[0, i+2, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['add_'+str(i)] = tf.add(SCNN_D['conv_bias_relu_'+str(i)],
                                           SCNN_D['slice_add_'+str(i)])
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['slice_top_'+str(i)],
                                                         SCNN_D['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_D['add_' +
                                                                str(i)],
                                                         SCNN_D['slice_bottom_'+str(i)]], axis=1)

    return SCNN_D['concat_'+str(FEATURE_MAP_H-1)]


def SCNN_U_F(SCNN_U_input):
    '''
        SCNN_Up 向上传递信息

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
    CHANNELS = SCNN_U_input.shape[3]
    FEATURE_MAP_H = SCNN_U_input.shape[1]
    # 声明参数
    w = tf.get_variable(name='w_scnn_u',
                        shape=[CHANNELS,
                               SCNN_KERNEL_LENGTH,
                               1,
                               CHANNELS],
                        initializer=tf.keras.initializers.zeros(),
                        trainable = False,
                        regularizer=tf.keras.regularizers.l2())
    b = tf.Variable(tf.zeros(CHANNELS),
                    trainable = False,
                    name="b_scnn_u")

    # 用于动态生成网络层
    SCNN_U = {}

    for i in range(FEATURE_MAP_H-1, -1, -1):
        # State 1 : 无slice_bottom,存在slice_conv,slice_add,slice_top
        if i == FEATURE_MAP_H-1 and FEATURE_MAP_H > 2:
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['slice_add_'+str(i)] = tf.slice(SCNN_U_input,
                                                   begin=[0, i-1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_U['slice_top_'+str(i)] = tf.slice(SCNN_U_input,
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i-1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['add_'+str(i)] = tf.add(SCNN_U['conv_bias_relu_'+str(i)],
                                           SCNN_U['slice_add_'+str(i)])
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['slice_top_'+str(i)],
                                                         SCNN_U['add_' +
                                                                str(i)],
                                                         SCNN_U['conv_bias_relu_'+str(i)]], axis=1)
        # State 2 : 无slice_top，存在slice_bottom,slice_conv,slice_add
        elif i == 1 and FEATURE_MAP_H > 2:
            SCNN_U['slice_add_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['slice_bottom_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['add_'+str(i)] = tf.add(SCNN_U['conv_bias_relu_'+str(i)],
                                           SCNN_U['slice_add_'+str(i)])
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['add_'+str(i)],
                                                         SCNN_U['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_U['slice_bottom_'+str(i)]], axis=1)
        # State 3 : 无slice_add,slice_top，存在slice_bottom,slice_conv
        elif i == 0 and FEATURE_MAP_H > 1:
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['slice_bottom_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['conv_bias_relu_'+str(i)],
                                                         SCNN_U['slice_bottom_'+str(i)]], axis=1)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_H == 1:
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, -1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['concat_'+str(i)] = SCNN_U['conv_bias_relu_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == FEATURE_MAP_H-1 and FEATURE_MAP_H == 2:
            SCNN_U['slice_add_'+str(i)] = tf.slice(SCNN_U_input,
                                                   begin=[0, i, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U_input,
                                                    begin=[0, i+1, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['add_'+str(i)] = tf.add(SCNN_U['conv_bias_relu_'+str(i)],
                                           SCNN_U['slice_add_'+str(i)])
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['add_'+str(i)],
                                                         SCNN_U['conv_bias_relu_'+str(i)]], axis=1)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            SCNN_U['slice_top_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i-1, -1, -1])
            SCNN_U['slice_add_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                   begin=[0, i-1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['slice_bottom_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['add_'+str(i)] = tf.add(SCNN_U['conv_bias_relu_'+str(i)],
                                           SCNN_U['slice_add_'+str(i)])
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['slice_top_'+str(i)],
                                                         SCNN_U['add_' +
                                                                str(i)],
                                                         SCNN_U['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_U['slice_bottom_'+str(i)]], axis=1)
    return SCNN_U['concat_'+str(0)]


def SCNN_R_F(SCNN_R_input):
    '''
        SCNN_Right 向右传递信息
        状态释义：
            state = 1：(when i == 0 && H > 2)
                无slice_top，存在slice_conv,slice_add,slice_bottom
            state = 2: (when i == H-2 && i > 0)
                无slice_bottom，存在slice_top,slice_conv,slice_add
            state = 3: (when i == H-1 && i > 0)
                无slice_add,slice_bottom，存在slice_top,slice_conv
            state = 4: (when H == 1)
                无slice_top,slice_add,slice_bottom，存在slice_conv
            state = 5: (when i == 0 && H == 2)
                无slice_top,slice_bottom，存在slice_conv,slice_add
            state = 6:
                存在slice_top,slice_conv,slice_add,slice_bottom
    '''
    CHANNELS = SCNN_R_input.shape[3]
    FEATURE_MAP_W = SCNN_R_input.shape[2]
    # 声明参数
    w = tf.get_variable(name='w_scnn_r',
                        shape=[CHANNELS,
                               SCNN_KERNEL_LENGTH,
                               1,
                               CHANNELS],
                        initializer=tf.keras.initializers.zeros(),
                        trainable = False,
                        regularizer=tf.keras.regularizers.l2())
    b = tf.Variable(tf.zeros(CHANNELS),
                    trainable = False,
                    name="b_scnn_r")

    # 用于动态生成网络层
    SCNN_R = {}

    '''

     '''
    for i in range(FEATURE_MAP_W):
        # State 1 : 无slice_top，存在slice_conv,slice_add,slice_bottom
        if i == 0 and FEATURE_MAP_W > 2:
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['slice_add_'+str(i)] = tf.slice(SCNN_R_input,
                                                   begin=[0, 0, i+1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_R['slice_bottom_'+str(i)] = tf.slice(SCNN_R_input,
                                                      begin=[0, 0, i+2, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['add_'+str(i)] = tf.add(SCNN_R['tran_aft_'+str(i)],
                                           SCNN_R['slice_add_'+str(i)])

            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['tran_aft_'+str(i)],
                                                         SCNN_R['add_' +
                                                                str(i)],
                                                         SCNN_R['slice_bottom_'+str(i)]], axis=2)
        # State 2 : 无slice_bottom，存在slice_top,slice_conv,slice_add
        elif i == FEATURE_MAP_W - 2 and i > 0:
            SCNN_R['slice_top_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i, -1])
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['slice_add_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, i+1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['add_'+str(i)] = tf.add(SCNN_R['tran_aft_'+str(i)],
                                           SCNN_R['slice_add_'+str(i)])
            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['slice_top_'+str(i)],
                                                         SCNN_R['tran_aft_' +
                                                                str(i)],
                                                         SCNN_R['add_'+str(i)]], axis=2)
        # State 3 : 无slice_add,slice_bottom，存在slice_top,slice_conv
        elif i == FEATURE_MAP_W - 1 and i > 0:
            SCNN_R['slice_top_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i, -1])
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['slice_top_'+str(i)],
                                                         SCNN_R['tran_aft_'+str(i)]], axis=2)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_W == 1:
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, -1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['concat_'+str(i)] = SCNN_R['tran_aft_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == 0 and FEATURE_MAP_W == 2:
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['slice_add_'+str(i)] = tf.slice(SCNN_R_input,
                                                   begin=[0, 0, i+1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['add_'+str(i)] = tf.add(SCNN_R['tran_aft_'+str(i)],
                                           SCNN_R['slice_add_'+str(i)])
            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['tran_aft_'+str(i)],
                                                         SCNN_R['add_'+str(i)]], axis=2)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            SCNN_R['slice_top_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i, -1])
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['slice_add_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, i+1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_R['slice_bottom_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                      begin=[0, 0, i+2, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['add_'+str(i)] = tf.add(SCNN_R['tran_aft_'+str(i)],
                                           SCNN_R['slice_add_'+str(i)])
            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['slice_top_'+str(i)],
                                                         SCNN_R['tran_aft_' +
                                                                str(i)],
                                                         SCNN_R['add_' +
                                                                str(i)],
                                                         SCNN_R['slice_bottom_'+str(i)]], axis=2)

    return SCNN_R['concat_'+str(FEATURE_MAP_W-1)]


def SCNN_L_F(SCNN_L_input):
    '''
        SCNN_Left 向左传递信息

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
    CHANNELS = SCNN_L_input.shape[3]
    FEATURE_MAP_W = SCNN_L_input.shape[2]
    # 声明参数
    w = tf.get_variable(name='w_scnn_l',
                        shape=[CHANNELS,
                               SCNN_KERNEL_LENGTH,
                               1,
                               CHANNELS],
                        initializer=tf.keras.initializers.zeros(),
                        trainable = False,
                        regularizer=tf.keras.regularizers.l2())
    b = tf.Variable(tf.zeros(CHANNELS),
                    trainable = False,
                    name="b_scnn_l")

    # 用于动态生成网络层
    SCNN_L = {}

    for i in range(FEATURE_MAP_W-1, -1, -1):
        # State 1 : 无slice_bottom,存在slice_conv,slice_add,slice_top
        if i == FEATURE_MAP_W-1 and FEATURE_MAP_W > 2:
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['slice_add_'+str(i)] = tf.slice(SCNN_L_input,
                                                   begin=[0, 0, i-1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_L['slice_top_'+str(i)] = tf.slice(SCNN_L_input,
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i-1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['add_'+str(i)] = tf.add(SCNN_L['conv_bias_relu_'+str(i)],
                                           SCNN_L['slice_add_'+str(i)])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['slice_top_'+str(i)],
                                                         SCNN_L['add_' +
                                                                str(i)],
                                                         SCNN_L['conv_bias_relu_'+str(i)]], axis=2)
        # State 2 : 无slice_top，存在slice_bottom,slice_conv,slice_add
        elif i == 1 and FEATURE_MAP_W > 2:
            SCNN_L['slice_add_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i, -1])
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['slice_bottom_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                      begin=[0, 0, i+1, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['add_'+str(i)] = tf.add(SCNN_L['conv_bias_relu_'+str(i)],
                                           SCNN_L['slice_add_'+str(i)])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['add_'+str(i)],
                                                         SCNN_L['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_L['slice_bottom_'+str(i)]], axis=2)
        # State 3 : 无slice_add,slice_top，存在slice_bottom,slice_conv
        elif i == 0 and FEATURE_MAP_W > 1:
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['slice_bottom_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                      begin=[0, 0, i+1, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['conv_bias_relu_'+str(i)],
                                                         SCNN_L['slice_bottom_'+str(i)]], axis=2)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_W == 1:
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, -1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['concat_'+str(i)] = SCNN_L['conv_bias_relu_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == FEATURE_MAP_W-1 and FEATURE_MAP_W == 2:
            SCNN_L['slice_add_'+str(i)] = tf.slice(SCNN_L_input,
                                                   begin=[0, 0, i, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L_input,
                                                    begin=[0, 0, i+1, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['add_'+str(i)] = tf.add(SCNN_L['conv_bias_relu_'+str(i)],
                                           SCNN_L['slice_add_'+str(i)])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['add_'+str(i)],
                                                         SCNN_L['conv_bias_relu_'+str(i)]], axis=2)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            SCNN_L['slice_top_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i-1, -1])
            SCNN_L['slice_add_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                   begin=[0, 0, i-1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['slice_bottom_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                      begin=[0, 0, i+1, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['add_'+str(i)] = tf.add(SCNN_L['conv_bias_relu_'+str(i)],
                                           SCNN_L['slice_add_'+str(i)])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['slice_top_'+str(i)],
                                                         SCNN_L['add_' +
                                                                str(i)],
                                                         SCNN_L['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_L['slice_bottom_'+str(i)]], axis=2)
    return SCNN_L['concat_'+str(0)]


def SCNN_D(SCNN_D_input):
    '''
        SCNN_Down 向下传递信息

        状态释义：
            state = 1：(when i == 0 && H > 2)
                无slice_top，存在slice_conv,slice_add,slice_bottom
            state = 2: (when i == H-2 && i > 0)
                无slice_bottom，存在slice_top,slice_conv,slice_add
            state = 3: (when i == H-1 && i > 0)
                无slice_add,slice_bottom，存在slice_top,slice_conv
            state = 4: (when H == 1)
                无slice_top,slice_add,slice_bottom，存在slice_conv
            state = 5: (when i == 0 && H == 2)
                无slice_top,slice_bottom，存在slice_conv,slice_add
            state = 6:
                存在slice_top,slice_conv,slice_add,slice_bottom
    '''
    CHANNELS = SCNN_D_input.shape[3]
    FEATURE_MAP_H = SCNN_D_input.shape[1]
    # 声明参数
    w = tf.get_variable(name='w_scnn_d',
                        shape=[CHANNELS,
                               SCNN_KERNEL_LENGTH,
                               1,
                               CHANNELS],
                        initializer=tf.keras.initializers.glorot_normal(),
                        regularizer=tf.keras.regularizers.l2())
    b = tf.Variable(tf.zeros(CHANNELS),
                    name="b_scnn_d")

    # 用于动态生成网络层
    SCNN_D = {}

    for i in range(FEATURE_MAP_H):
        # State 1 : 无slice_top，存在slice_conv,slice_add,slice_bottom
        if i == 0 and FEATURE_MAP_H > 2:
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['slice_add_'+str(i)] = tf.slice(SCNN_D_input,
                                                   begin=[0, i+1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_D['slice_bottom_'+str(i)] = tf.slice(SCNN_D_input,
                                                      begin=[0, i+2, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['add_'+str(i)] = tf.add(SCNN_D['conv_bias_relu_'+str(i)],
                                           SCNN_D['slice_add_'+str(i)])
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['conv_bias_relu_'+str(i)],
                                                         SCNN_D['add_' +
                                                                str(i)],
                                                         SCNN_D['slice_bottom_'+str(i)]], axis=1)
        # State 2 : 无slice_bottom，存在slice_top,slice_conv,slice_add
        elif i == FEATURE_MAP_H - 2 and i > 0:
            SCNN_D['slice_top_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['slice_add_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, i+1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['add_'+str(i)] = tf.add(SCNN_D['conv_bias_relu_'+str(i)],
                                           SCNN_D['slice_add_'+str(i)])
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['slice_top_'+str(i)],
                                                         SCNN_D['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_D['add_'+str(i)]], axis=1)
        # State 3 : 无slice_add,slice_bottom，存在slice_top,slice_conv
        elif i == FEATURE_MAP_H - 1 and i > 0:
            SCNN_D['slice_top_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['slice_top_'+str(i)],
                                                         SCNN_D['conv_bias_relu_'+str(i)]], axis=1)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_H == 1:
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, -1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['concat_'+str(i)] = SCNN_D['conv_bias_relu_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == 0 and FEATURE_MAP_H == 2:
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['slice_add_'+str(i)] = tf.slice(SCNN_D_input,
                                                   begin=[0, i+1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['add_'+str(i)] = tf.add(SCNN_D['conv_bias_relu_'+str(i)],
                                           SCNN_D['slice_add_'+str(i)])
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['conv_bias_relu_'+str(i)],
                                                         SCNN_D['add_'+str(i)]], axis=1)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            SCNN_D['slice_top_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            SCNN_D['slice_conv_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_D['slice_add_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                   begin=[0, i+1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_D['slice_bottom_'+str(i)] = tf.slice(SCNN_D['concat_'+str(i-1)],
                                                      begin=[0, i+2, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_D['tran_bef_'+str(i)] = tf.transpose(SCNN_D['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_D['pad_'+str(i)] = tf.pad(SCNN_D['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_D['conv_'+str(i)] = tf.nn.conv2d(SCNN_D['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_D['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_D['conv_'+str(i)], b),
                                                          name=None)
            SCNN_D['add_'+str(i)] = tf.add(SCNN_D['conv_bias_relu_'+str(i)],
                                           SCNN_D['slice_add_'+str(i)])
            SCNN_D['concat_'+str(i)] = tf.concat(values=[SCNN_D['slice_top_'+str(i)],
                                                         SCNN_D['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_D['add_' +
                                                                str(i)],
                                                         SCNN_D['slice_bottom_'+str(i)]], axis=1)

    return SCNN_D['concat_'+str(FEATURE_MAP_H-1)]


def SCNN_U(SCNN_U_input):
    '''
        SCNN_Up 向上传递信息

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
    CHANNELS = SCNN_U_input.shape[3]
    FEATURE_MAP_H = SCNN_U_input.shape[1]
    # 声明参数
    w = tf.get_variable(name='w_scnn_u',
                        shape=[CHANNELS,
                               SCNN_KERNEL_LENGTH,
                               1,
                               CHANNELS],
                        initializer=tf.keras.initializers.glorot_normal(),
                        regularizer=tf.keras.regularizers.l2())
    b = tf.Variable(tf.zeros(CHANNELS),
                    name="b_scnn_u")

    # 用于动态生成网络层
    SCNN_U = {}

    for i in range(FEATURE_MAP_H-1, -1, -1):
        # State 1 : 无slice_bottom,存在slice_conv,slice_add,slice_top
        if i == FEATURE_MAP_H-1 and FEATURE_MAP_H > 2:
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['slice_add_'+str(i)] = tf.slice(SCNN_U_input,
                                                   begin=[0, i-1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_U['slice_top_'+str(i)] = tf.slice(SCNN_U_input,
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i-1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['add_'+str(i)] = tf.add(SCNN_U['conv_bias_relu_'+str(i)],
                                           SCNN_U['slice_add_'+str(i)])
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['slice_top_'+str(i)],
                                                         SCNN_U['add_' +
                                                                str(i)],
                                                         SCNN_U['conv_bias_relu_'+str(i)]], axis=1)
        # State 2 : 无slice_top，存在slice_bottom,slice_conv,slice_add
        elif i == 1 and FEATURE_MAP_H > 2:
            SCNN_U['slice_add_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i, -1, -1])
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['slice_bottom_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['add_'+str(i)] = tf.add(SCNN_U['conv_bias_relu_'+str(i)],
                                           SCNN_U['slice_add_'+str(i)])
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['add_'+str(i)],
                                                         SCNN_U['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_U['slice_bottom_'+str(i)]], axis=1)
        # State 3 : 无slice_add,slice_top，存在slice_bottom,slice_conv
        elif i == 0 and FEATURE_MAP_H > 1:
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['slice_bottom_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['conv_bias_relu_'+str(i)],
                                                         SCNN_U['slice_bottom_'+str(i)]], axis=1)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_H == 1:
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U_input,
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, -1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['concat_'+str(i)] = SCNN_U['conv_bias_relu_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == FEATURE_MAP_H-1 and FEATURE_MAP_H == 2:
            SCNN_U['slice_add_'+str(i)] = tf.slice(SCNN_U_input,
                                                   begin=[0, i, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U_input,
                                                    begin=[0, i+1, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['add_'+str(i)] = tf.add(SCNN_U['conv_bias_relu_'+str(i)],
                                           SCNN_U['slice_add_'+str(i)])
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['add_'+str(i)],
                                                         SCNN_U['conv_bias_relu_'+str(i)]], axis=1)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            SCNN_U['slice_top_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, i-1, -1, -1])
            SCNN_U['slice_add_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                   begin=[0, i-1, 0, 0],
                                                   size=[-1, 1, -1, -1])
            SCNN_U['slice_conv_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                    begin=[0, i, 0, 0],
                                                    size=[-1, 1, -1, -1])
            SCNN_U['slice_bottom_'+str(i)] = tf.slice(SCNN_U['concat_'+str(i+1)],
                                                      begin=[0, i+1, 0, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_U['tran_bef_'+str(i)] = tf.transpose(SCNN_U['slice_conv_'+str(i)],
                                                      perm=[0, 3, 2, 1])
            SCNN_U['pad_'+str(i)] = tf.pad(SCNN_U['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_U['conv_'+str(i)] = tf.nn.conv2d(SCNN_U['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_U['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_U['conv_'+str(i)], b),
                                                          name=None)
            SCNN_U['add_'+str(i)] = tf.add(SCNN_U['conv_bias_relu_'+str(i)],
                                           SCNN_U['slice_add_'+str(i)])
            SCNN_U['concat_'+str(i)] = tf.concat(values=[SCNN_U['slice_top_'+str(i)],
                                                         SCNN_U['add_' +
                                                                str(i)],
                                                         SCNN_U['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_U['slice_bottom_'+str(i)]], axis=1)
    return SCNN_U['concat_'+str(0)]


def SCNN_R(SCNN_R_input):
    '''
        SCNN_Right 向右传递信息
        状态释义：
            state = 1：(when i == 0 && H > 2)
                无slice_top，存在slice_conv,slice_add,slice_bottom
            state = 2: (when i == H-2 && i > 0)
                无slice_bottom，存在slice_top,slice_conv,slice_add
            state = 3: (when i == H-1 && i > 0)
                无slice_add,slice_bottom，存在slice_top,slice_conv
            state = 4: (when H == 1)
                无slice_top,slice_add,slice_bottom，存在slice_conv
            state = 5: (when i == 0 && H == 2)
                无slice_top,slice_bottom，存在slice_conv,slice_add
            state = 6:
                存在slice_top,slice_conv,slice_add,slice_bottom
    '''
    CHANNELS = SCNN_R_input.shape[3]
    FEATURE_MAP_W = SCNN_R_input.shape[2]
    # 声明参数
    w = tf.get_variable(name='w_scnn_r',
                        shape=[CHANNELS,
                               SCNN_KERNEL_LENGTH,
                               1,
                               CHANNELS],
                        initializer=tf.keras.initializers.glorot_normal(),
                        regularizer=tf.keras.regularizers.l2())
    b = tf.Variable(tf.zeros(CHANNELS),
                    name="b_scnn_r")

    # 用于动态生成网络层
    SCNN_R = {}

    for i in range(FEATURE_MAP_W):
        # State 1 : 无slice_top，存在slice_conv,slice_add,slice_bottom
        if i == 0 and FEATURE_MAP_W > 2:
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['slice_add_'+str(i)] = tf.slice(SCNN_R_input,
                                                   begin=[0, 0, i+1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_R['slice_bottom_'+str(i)] = tf.slice(SCNN_R_input,
                                                      begin=[0, 0, i+2, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['add_'+str(i)] = tf.add(SCNN_R['tran_aft_'+str(i)],
                                           SCNN_R['slice_add_'+str(i)])

            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['tran_aft_'+str(i)],
                                                         SCNN_R['add_' +
                                                                str(i)],
                                                         SCNN_R['slice_bottom_'+str(i)]], axis=2)
        # State 2 : 无slice_bottom，存在slice_top,slice_conv,slice_add
        elif i == FEATURE_MAP_W - 2 and i > 0:
            SCNN_R['slice_top_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i, -1])
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['slice_add_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, i+1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['add_'+str(i)] = tf.add(SCNN_R['tran_aft_'+str(i)],
                                           SCNN_R['slice_add_'+str(i)])
            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['slice_top_'+str(i)],
                                                         SCNN_R['tran_aft_' +
                                                                str(i)],
                                                         SCNN_R['add_'+str(i)]], axis=2)
        # State 3 : 无slice_add,slice_bottom，存在slice_top,slice_conv
        elif i == FEATURE_MAP_W - 1 and i > 0:
            SCNN_R['slice_top_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i, -1])
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['slice_top_'+str(i)],
                                                         SCNN_R['tran_aft_'+str(i)]], axis=2)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_W == 1:
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, -1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['concat_'+str(i)] = SCNN_R['tran_aft_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == 0 and FEATURE_MAP_W == 2:
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['slice_add_'+str(i)] = tf.slice(SCNN_R_input,
                                                   begin=[0, 0, i+1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['add_'+str(i)] = tf.add(SCNN_R['tran_aft_'+str(i)],
                                           SCNN_R['slice_add_'+str(i)])
            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['tran_aft_'+str(i)],
                                                         SCNN_R['add_'+str(i)]], axis=2)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            SCNN_R['slice_top_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i, -1])
            SCNN_R['slice_conv_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_R['slice_add_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                   begin=[0, 0, i+1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_R['slice_bottom_'+str(i)] = tf.slice(SCNN_R['concat_'+str(i-1)],
                                                      begin=[0, 0, i+2, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_R['tran_bef_'+str(i)] = tf.transpose(SCNN_R['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_R['pad_'+str(i)] = tf.pad(SCNN_R['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_R['conv_'+str(i)] = tf.nn.conv2d(SCNN_R['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_R['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_R['conv_'+str(i)], b),
                                                          name=None)
            SCNN_R['tran_aft_'+str(i)] = tf.transpose(SCNN_R['conv_bias_relu_'+str(i)],
                                                      perm=[0, 2, 1, 3])
            SCNN_R['add_'+str(i)] = tf.add(SCNN_R['tran_aft_'+str(i)],
                                           SCNN_R['slice_add_'+str(i)])
            SCNN_R['concat_'+str(i)] = tf.concat(values=[SCNN_R['slice_top_'+str(i)],
                                                         SCNN_R['tran_aft_' +
                                                                str(i)],
                                                         SCNN_R['add_' +
                                                                str(i)],
                                                         SCNN_R['slice_bottom_'+str(i)]], axis=2)

    return SCNN_R['concat_'+str(FEATURE_MAP_W-1)]


def SCNN_L(SCNN_L_input):
    '''
        SCNN_Left 向左传递信息

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
    CHANNELS = SCNN_L_input.shape[3]
    FEATURE_MAP_W = SCNN_L_input.shape[2]
    # 声明参数
    w = tf.get_variable(name='w_scnn_l',
                        shape=[CHANNELS,
                               SCNN_KERNEL_LENGTH,
                               1,
                               CHANNELS],
                        initializer=tf.keras.initializers.glorot_normal(),
                        regularizer=tf.keras.regularizers.l2())
    b = tf.Variable(tf.zeros(CHANNELS),
                    name="b_scnn_l")

    # 用于动态生成网络层
    SCNN_L = {}

    for i in range(FEATURE_MAP_W-1, -1, -1):
        # State 1 : 无slice_bottom,存在slice_conv,slice_add,slice_top
        if i == FEATURE_MAP_W-1 and FEATURE_MAP_W > 2:
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['slice_add_'+str(i)] = tf.slice(SCNN_L_input,
                                                   begin=[0, 0, i-1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_L['slice_top_'+str(i)] = tf.slice(SCNN_L_input,
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i-1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['add_'+str(i)] = tf.add(SCNN_L['conv_bias_relu_'+str(i)],
                                           SCNN_L['slice_add_'+str(i)])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['slice_top_'+str(i)],
                                                         SCNN_L['add_' +
                                                                str(i)],
                                                         SCNN_L['conv_bias_relu_'+str(i)]], axis=2)
        # State 2 : 无slice_top，存在slice_bottom,slice_conv,slice_add
        elif i == 1 and FEATURE_MAP_W > 2:
            SCNN_L['slice_add_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i, -1])
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['slice_bottom_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                      begin=[0, 0, i+1, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1])
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['add_'+str(i)] = tf.add(SCNN_L['conv_bias_relu_'+str(i)],
                                           SCNN_L['slice_add_'+str(i)])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['add_'+str(i)],
                                                         SCNN_L['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_L['slice_bottom_'+str(i)]], axis=2)
        # State 3 : 无slice_add,slice_top，存在slice_bottom,slice_conv
        elif i == 0 and FEATURE_MAP_W > 1:
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['slice_bottom_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                      begin=[0, 0, i+1, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['conv_bias_relu_'+str(i)],
                                                         SCNN_L['slice_bottom_'+str(i)]], axis=2)
        # State 4 : 无slice_top,slice_add,slice_bottom，存在slice_conv
        elif FEATURE_MAP_W == 1:
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L_input,
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, -1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['concat_'+str(i)] = SCNN_L['conv_bias_relu_'+str(i)]
        # State 5 : 无slice_top,slice_bottom，存在slice_conv,slice_add
        elif i == FEATURE_MAP_W-1 and FEATURE_MAP_W == 2:
            SCNN_L['slice_add_'+str(i)] = tf.slice(SCNN_L_input,
                                                   begin=[0, 0, i, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L_input,
                                                    begin=[0, 0, i+1, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['add_'+str(i)] = tf.add(SCNN_L['conv_bias_relu_'+str(i)],
                                           SCNN_L['slice_add_'+str(i)])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['add_'+str(i)],
                                                         SCNN_L['conv_bias_relu_'+str(i)]], axis=2)
        # State 6 : 存在slice_top,slice_conv,slice_add,slice_bottom
        else:
            SCNN_L['slice_top_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                   begin=[0, 0, 0, 0],
                                                   size=[-1, -1, i-1, -1])
            SCNN_L['slice_add_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                   begin=[0, 0, i-1, 0],
                                                   size=[-1, -1, 1, -1])
            SCNN_L['slice_conv_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                    begin=[0, 0, i, 0],
                                                    size=[-1, -1, 1, -1])
            SCNN_L['slice_bottom_'+str(i)] = tf.slice(SCNN_L['concat_'+str(i+1)],
                                                      begin=[0, 0, i+1, 0],
                                                      size=[-1, -1, -1, -1])
            SCNN_L['tran_bef_'+str(i)] = tf.transpose(SCNN_L['slice_conv_'+str(i)],
                                                      perm=[0, 3, 1, 2])
            SCNN_L['pad_'+str(i)] = tf.pad(SCNN_L['tran_bef_'+str(i)],
                                           paddings=PADDING,
                                           mode='CONSTANT',
                                           constant_values=0)
            SCNN_L['conv_'+str(i)] = tf.nn.conv2d(SCNN_L['pad_'+str(i)],
                                                  filter=w,
                                                  strides=[1, 1, 1, 1],
                                                  padding='VALID',
                                                  use_cudnn_on_gpu=True,
                                                  data_format='NHWC',
                                                  dilations=[1, 1, 1, 1],
                                                  name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.nn.relu(tf.nn.bias_add(SCNN_L['conv_'+str(i)], b),
                                                          name=None)
            SCNN_L['conv_bias_relu_'+str(i)] = tf.transpose(SCNN_L['conv_bias_relu_'+str(i)],
                                                            perm=[0, 2, 1, 3])
            SCNN_L['add_'+str(i)] = tf.add(SCNN_L['conv_bias_relu_'+str(i)],
                                           SCNN_L['slice_add_'+str(i)])
            SCNN_L['concat_'+str(i)] = tf.concat(values=[SCNN_L['slice_top_'+str(i)],
                                                         SCNN_L['add_' +
                                                                str(i)],
                                                         SCNN_L['conv_bias_relu_' +
                                                                str(i)],
                                                         SCNN_L['slice_bottom_'+str(i)]], axis=2)
    return SCNN_L['concat_'+str(0)]


def CreateCNNModel(feature_dim):
    '''
        CNN 网络构造

        Imput:
            feature_dim     特征维度
    '''
    inputs = tf.keras.Input(shape=(feature_dim, 300, 1),
                            name='Input_Layer',
                            dtype=tf.float32)

    conv_1 = tf.keras.layers.Conv2D(filters=CONV1_FILTERS,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    data_format='channels_last',
                                    use_bias=True,
                                    kernel_initializer=tf.keras.initializers.random_uniform(),
                                    bias_initializer=tf.zeros_initializer())(inputs)

    maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid',
                                          data_format='channels_last')(conv_1)

    conv_2 = tf.keras.layers.Conv2D(filters=CONV2_FILTERS,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    data_format='channels_last',
                                    use_bias=True,
                                    kernel_initializer=tf.keras.initializers.random_uniform(),
                                    bias_initializer=tf.zeros_initializer())(maxpool_1)

    maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid',
                                          data_format='channels_last')(conv_2)

    fc_input = tf.keras.layers.Flatten()(maxpool_2)

    fc_1 = tf.keras.layers.Dense(units=512,
                                 activation='tanh',
                                 use_bias=True,
                                 kernel_initializer=tf.keras.initializers.random_uniform(),
                                 bias_initializer=tf.zeros_initializer())(fc_input)

    dropout_1 = tf.keras.layers.Dropout(0.1)(fc_1)

    fc_2 = tf.keras.layers.Dense(units=128,
                                 activation='tanh',
                                 use_bias=True,
                                 kernel_initializer=tf.keras.initializers.random_uniform(),
                                 bias_initializer=tf.zeros_initializer())(dropout_1)

    dropout_2 = tf.keras.layers.Dropout(0.1)(fc_2)

    logits = tf.keras.layers.Dense(units=5,
                                   activation='softmax',
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.random_uniform(),
                                   bias_initializer=tf.zeros_initializer())(dropout_2)

    model = tf.keras.Model(inputs=inputs, outputs=logits)

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def CreateMVCNNModel(feature_dim):
    '''
        MV-CNN 网络构造

        Imput:
            feature_dim     特征维度
    '''
    inputs = tf.keras.Input(shape=(feature_dim, 300, 1),
                            name='Input_Layer',
                            dtype=tf.float32)
    # '''
    # with tf.name_scope('SCNN_D_%d' % feature_dim):
    #     SCNN_D_out = tf.keras.layers.Lambda(SCNN_D)(inputs)
    # with tf.name_scope('SCNN_U_%d' % feature_dim):
    #     SCNN_U_out = tf.keras.layers.Lambda(SCNN_U)(inputs)
    # with tf.name_scope('SCNN_R_%d' % feature_dim):
    #     SCNN_R_out = tf.keras.layers.Lambda(SCNN_R)(inputs)
    # with tf.name_scope('SCNN_L_%d' % feature_dim):
    #     SCNN_L_out = tf.keras.layers.Lambda(SCNN_L)(inputs)
    # '''

    # '''
    # with tf.name_scope('SCNN_D_%d' % feature_dim):
    #     SCNN_D_out = tf.keras.layers.Lambda(SCNN_D_F)(inputs)
    # with tf.name_scope('SCNN_U_%d' % feature_dim):
    #     SCNN_U_out = tf.keras.layers.Lambda(SCNN_U_F)(inputs)
    # with tf.name_scope('SCNN_R_%d' % feature_dim):
    #     SCNN_R_out = tf.keras.layers.Lambda(SCNN_R_F)(inputs)
    # with tf.name_scope('SCNN_L_%d' % feature_dim):
    #     SCNN_L_out = tf.keras.layers.Lambda(SCNN_L_F)(inputs)
    # '''
    SCNN_L_out = MvCnnLeftLayer(SCNN_KERNEL_LENGTH)(inputs)
    SCNN_R_out = MvCnnRightLayer(SCNN_KERNEL_LENGTH)(inputs)
    SCNN_U_out = MvCnnUpLayer(SCNN_KERNEL_LENGTH)(inputs)
    SCNN_D_out = MvCnnDownLayer(SCNN_KERNEL_LENGTH)(inputs)
    
    concat = tf.keras.layers.concatenate(
        [inputs, SCNN_D_out, SCNN_U_out, SCNN_R_out, SCNN_L_out], axis=-1)
    
    # concat = tf.keras.layers.concatenate([inputs, SCNN_L_out], axis=-1)

    conv_1 = tf.keras.layers.Conv2D(filters=CONV1_FILTERS,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    data_format='channels_last',
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=tf.keras.initializers.glorot_normal(),
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=tf.keras.regularizers.l2(),
                                    bias_regularizer=tf.keras.regularizers.l2())(concat)

    maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid',
                                          data_format='channels_last')(conv_1)

    conv_2 = tf.keras.layers.Conv2D(filters=CONV2_FILTERS,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    data_format='channels_last',
                                    activation='relu',
                                    use_bias=True,
                                    kernel_initializer=tf.keras.initializers.glorot_normal(),
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=tf.keras.regularizers.l2(),
                                    bias_regularizer=tf.keras.regularizers.l2())(maxpool_1)

    maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid',
                                          data_format='channels_last')(conv_2)

    fc_input = tf.keras.layers.Flatten()(maxpool_2)

    fc_1 = tf.keras.layers.Dense(units=512,
                                 activation='tanh',
                                 use_bias=True,
                                 kernel_initializer=tf.keras.initializers.glorot_normal(),
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=tf.keras.regularizers.l2(),
                                 bias_regularizer=tf.keras.regularizers.l2())(fc_input)

    dropout_1 = tf.keras.layers.Dropout(0.1)(fc_1)

    fc_2 = tf.keras.layers.Dense(units=128,
                                 activation='tanh',
                                 use_bias=True,
                                 kernel_initializer=tf.keras.initializers.glorot_normal(),
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=tf.keras.regularizers.l2(),
                                 bias_regularizer=tf.keras.regularizers.l2())(dropout_1)

    dropout_2 = tf.keras.layers.Dropout(0.1)(fc_2)

    logits = tf.keras.layers.Dense(units=5,
                                   activation='softmax',
                                   use_bias=True,
                                   kernel_initializer=tf.keras.initializers.glorot_normal(),
                                   bias_initializer=tf.zeros_initializer())(dropout_2)

    model = tf.keras.Model(inputs=inputs, outputs=logits)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    tf.keras.backend.get_session().run(tf.global_variables_initializer())
    return model

def CreateSimpleRNNModel(feature_dim):
    '''
        Simple RNN 网络构造

        Imput:
            feature_dim     特征维度
    '''
    inputs = tf.keras.Input(shape=(feature_dim,300),
                            name='Input_Layer',
                            dtype=tf.float32)

    rnn = tf.keras.layers.SimpleRNN(128, 
                                    activation='tanh', 
                                    use_bias=True, 
                                    kernel_initializer='glorot_uniform', 
                                    recurrent_initializer='orthogonal', 
                                    bias_initializer='zeros', 
                                    kernel_regularizer=None, 
                                    recurrent_regularizer=None, 
                                    bias_regularizer=None, 
                                    activity_regularizer=None, 
                                    kernel_constraint=None, 
                                    recurrent_constraint=None, 
                                    bias_constraint=None, 
                                    dropout=0.0, 
                                    recurrent_dropout=0.0, 
                                    return_sequences=False, 
                                    return_state=False, 
                                    go_backwards=False, 
                                    stateful=False, 
                                    unroll=False)(inputs)

    logits = tf.keras.layers.Dense(units=5,
                                    activation='softmax',
                                    use_bias=True,
                                    kernel_initializer=tf.keras.initializers.random_uniform(),
                                    bias_initializer=tf.zeros_initializer())(rnn)

    model = tf.keras.Model(inputs=inputs, outputs=logits)

    model.compile(optimizer='sgd',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def CreateLSTMModel(feature_dim):
    '''
        LSTM 网络构造

        Imput:
            feature_dim     特征维度
    '''
    inputs = tf.keras.Input(shape=(feature_dim, 300),
                            name='Input_Layer',
                            dtype=tf.float32)

    rnn = tf.keras.layers.LSTM(128, 
                                activation='tanh', 
                                use_bias=True, 
                                kernel_initializer='glorot_uniform', 
                                recurrent_initializer='orthogonal', 
                                bias_initializer='zeros', 
                                kernel_regularizer=None, 
                                recurrent_regularizer=None, 
                                bias_regularizer=None, 
                                activity_regularizer=None, 
                                kernel_constraint=None, 
                                recurrent_constraint=None, 
                                bias_constraint=None, 
                                dropout=0.0, 
                                recurrent_dropout=0.0, 
                                return_sequences=False, 
                                return_state=False, 
                                go_backwards=False, 
                                stateful=False, 
                                unroll=False)(inputs)

    logits = tf.keras.layers.Dense(units=5,
                                    activation='softmax',
                                    use_bias=True,
                                    kernel_initializer=tf.keras.initializers.random_uniform(),
                                    bias_initializer=tf.zeros_initializer())(rnn)

    model = tf.keras.Model(inputs=inputs, outputs=logits)

    model.compile(optimizer='sgd',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model

def SaveLog(logfile, str):
    '''
        日志打印保存

        Input：
            logfile     日志文件对象
            str         日志内容
    '''
    print(str)
    logfile.write(str+'\n')

def ModelCrossValidation(creatModel,dim, x_train, y_train, x_test, y_test, skf, logfolder, name):
    '''
        模型交叉验证方法

        Input：
            creatModel      模型构造方法
            dim             特征维度
            x_train         训练数据
            y_train         训练标签
            x_test          测试数据
            y_test          测试标签
            skf             K折交叉方法
            logfolder       输出路径
            name            测试名称
    '''
    # 打开日志输出
    logfile = open(logfolder+name+'.txt', 'w', encoding='utf-8')
    idx = 1
    logPicFolder = logfolder+"trainLogPic/"+name+"/"
    os.makedirs(logPicFolder)
    for train_idx, valid_idx in skf.split(x_train, y_train):
        # 模型导入
        model = creatModel(dim)
        if idx == 1:
        #     # 打印网络结构
            model.summary()
        #     # # 绘制模型的结构图 此处还出现点问题，待解决
        #     # tf.keras.utils.plot_model(model,
        #     #                           to_file='model.png',
        #     #                           show_shapes=True,
        #     #                           show_layer_names=True)
        SaveLog(logfile, "正在 第%d次 训练网络" % idx + name+"，请耐心等候......")
        y_train_onehot = np_utils.to_categorical(y_train, 5)
        if x_train.ndim == 4:
            x = x_train[train_idx, :, :, :]
            val_x = x_train[valid_idx, :, :, :]
        else:
            x = x_train[train_idx, :, :]
            val_x = x_train[valid_idx, :, :]
        y = y_train_onehot[train_idx, :]
        val_y = y_train_onehot[valid_idx, :]
        startTime = time.clock()
        history = model.fit(x=x,
                            y=y,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            validation_data=(val_x, val_y),
                            shuffle=True)

        endTime = time.clock()
        # 注意，这里的时间window和linux不同
        SaveLog(logfile, "网络训练已完成 耗时%f 秒" % ((float)(endTime - startTime)))
        # 注意：sklearn中的标签都不是稀疏的
        predict = model.predict(x_test)
        SaveLog(logfile, "测试集预测结果：")
        SaveLog(logfile, classification_report(
            y_test, np.argmax(predict, axis=1)))
        # 打印混淆矩阵
        SaveLog(logfile, "测试集混淆矩阵：")
        SaveLog(logfile, str(sklearn.metrics.confusion_matrix(
            y_test, np.argmax(predict, axis=1))))
        
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        acc = history.history['acc']

        # 生成统计图片
        Labels = ["val_loss", "val_acc", "loss", "acc"]
        # colorLabel = ["r",'g',"b"]
        plt.rcParams['font.sans-serif'] = ['SIMHEI']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        print("正在生成图片，请稍后......")

        for i in range(4):
            plt.figure(figsize=(6, 4))
            plt.title(name+str(idx)+Labels[i])
            plt.plot(np.arange(1, EPOCHS+1, 1),
                     history.history[Labels[i]], '-r*')
            plt.grid(True, linestyle="-.", color="gray", linewidth="0.5")
            plt.xticks(np.arange(1, EPOCHS+1, 1))
            plt.xlabel('Epochs')
            plt.gcf().savefig(logPicFolder+Labels[i] + '_'+str(idx) +
                              ".png", dpi=200, bbox_inches='tight')

        print("生成图片完毕！")

        # 保存日志文件
        log_pd = pd.DataFrame(history.history)
        log = np.array(log_pd)
        np.save(logPicFolder+'log_'+str(idx), log)

        # 保存模型
        model.save(logfolder+name+'_'+str(idx)+'.h5')

        # 清空session，重置图
        tf.keras.backend.clear_session()
        tf.reset_default_graph()
        idx += 1

    logfile.close()


if __name__ == "__main__":
    log_folder = "./experiment/" + time.strftime('%Y-%m-%d %H.%M.%S', time.localtime(time.time()))+'/'

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                    help="Batch Size")
    parser.add_argument("-e", "--epoch", type=int, default=10,
                    help="Train Epoch")
    parser.add_argument("-g", "--gpu", type=str, default="0",
                    help="GPU Device")
    parser.add_argument("-l", "--logdir", type=str, default=log_folder,
                    help="Log Dir")
    parser.add_argument("-m", "--model", type=str, default="0123456789abcdef",
                    help="select model")
    args = parser.parse_args()
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # load data
    x_train, x_train_FA, x_train_DA, x_train_DA_FA, x_test, x_test_FA, y_train, y_train_DA, y_test = LoadData()
    # cross validation
    # n-fold=10
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    log_folder = args.logdir
    os.makedirs(log_folder)
    # Experiment 1
    # orign CNN
    if "0" in args.model:
        ModelCrossValidation(CreateCNNModel,6, x_train, y_train, x_test, y_test, skf, log_folder, "orignCNN")
               
    # Experiment 2
    # orign RNN
    if "1" in args.model:
        ModelCrossValidation(CreateSimpleRNNModel,6, np.squeeze(x_train), y_train, np.squeeze(x_test),y_test, skf, log_folder, "orignSimpleRNN")
    
    # Experiment 3
    # orign LSTM
    if "2" in args.model:
        ModelCrossValidation(CreateLSTMModel,6, np.squeeze(x_train), y_train, np.squeeze(x_test),y_test, skf, log_folder, "orignLSTM")

    # Experiment 4
    # orign MV-CNN
    if "3" in args.model:
        ModelCrossValidation(CreateMVCNNModel,6, x_train, y_train, x_test, y_test, skf, log_folder, "orignMVCNN")

    # Experiment 5
    # orign+DA CNN
    if "4" in args.model:
        ModelCrossValidation(CreateCNNModel,6, x_train_DA, y_train_DA, x_test, y_test, skf, log_folder, "orignDACNN")
               
    # Experiment 6
    # orign+DA RNN
    if "5" in args.model:
        ModelCrossValidation(CreateSimpleRNNModel,6, np.squeeze(x_train_DA), y_train_DA, np.squeeze(x_test), y_test, skf, log_folder, "orignDASimpleRNN")
    
    # Experiment 7
    # orign+DA LSTM
    if "6" in args.model:
        ModelCrossValidation(CreateLSTMModel,6, np.squeeze(x_train_DA), y_train_DA, np.squeeze(x_test), y_test, skf, log_folder, "orignDALSTM")

    # Experiment 8
    # orign+DA MV-CNN
    if "7" in args.model:
        ModelCrossValidation(CreateMVCNNModel,6, x_train_DA, y_train_DA, x_test, y_test, skf, log_folder, "orignDAMVCNN")

    # Experiment 9
    # orign+FA CNN
    if "8" in args.model:
        ModelCrossValidation(CreateCNNModel,38, x_train_FA, y_train, x_test_FA, y_test, skf, log_folder, "orignFACNN")
               
    # Experiment 10
    # orign+FA RNN
    if "9" in args.model:
        ModelCrossValidation(CreateSimpleRNNModel,38, np.squeeze(x_train_FA), y_train, np.squeeze(x_test_FA), y_test, skf, log_folder, "orignFASimpleRNN")
    
    # Experiment 11
    # orign+FA LSTM
    if "a" in args.model:
        ModelCrossValidation(CreateLSTMModel,38, np.squeeze(x_train_FA), y_train, np.squeeze(x_test_FA), y_test, skf, log_folder, "orignFALSTM")

    # Experiment 12
    # orign+FA MV-CNN
    if "b" in args.model:
        ModelCrossValidation(CreateMVCNNModel,38, x_train_FA, y_train, x_test_FA, y_test, skf, log_folder, "orignFAMVCNN")

    # Experiment 13
    # orign+DA+FA CNN
    if "c" in args.model:
        ModelCrossValidation(CreateCNNModel,38, x_train_DA_FA, y_train_DA, x_test_FA, y_test, skf, log_folder, "orignDAFACNN")
               
    # Experiment 14
    # orign+DA+FA RNN
    if "d" in args.model:
        ModelCrossValidation(CreateSimpleRNNModel,38, np.squeeze(x_train_DA_FA), y_train_DA, np.squeeze(x_test_FA), y_test, skf, log_folder, "orignDAFASimpleRNN")
    
    # Experiment 15
    # orign+DA+FA LSTM
    if "e" in args.model:
        ModelCrossValidation(CreateLSTMModel,38, np.squeeze(x_train_DA_FA), y_train_DA, np.squeeze(x_test_FA), y_test, skf, log_folder, "orignDAFALSTM")
    
    # Experiment 16
    # orign+DA+FA MV-CNN
    if "f" in args.model:
        ModelCrossValidation(CreateMVCNNModel,38, x_train_DA_FA, y_train_DA, x_test_FA, y_test, skf, log_folder, "orignDAFAMVCNN")




    '''
    # Experiment 1
    # orign CNN
    ModelCrossValidation(CreateCNNModel,6, x_train, y_train, x_test,
                 y_test, skf, log_folder, "orign+CNN")
               
    # Experiment 2
    # orign RNN
    ModelCrossValidation(CreateSimpleRNNModel,6, np.squeeze(x_train), y_train, np.squeeze(x_test),
                 y_test, skf, log_folder, "orign+SimpleRNN")
    
    # Experiment 3
    # orign LSTM
    ModelCrossValidation(CreateLSTMModel,6, np.squeeze(x_train), y_train, np.squeeze(x_test),
                 y_test, skf, log_folder, "orign+LSTM")

    # Experiment 4
    # orign MV-CNN
    ModelCrossValidation(CreateMVCNNModel,6, x_train, y_train, x_test,
                y_test, skf, log_folder, "orign+MV_CNN")

    # Experiment 5
    # orign+DA CNN
    ModelCrossValidation(CreateCNNModel,6, x_train_DA, y_train_DA, x_test,
                 y_test, skf, log_folder, "orign+DA+CNN")
               
    # Experiment 6
    # orign+DA RNN
    ModelCrossValidation(CreateSimpleRNNModel,6, np.squeeze(x_train_DA), y_train_DA, np.squeeze(x_test),
                 y_test, skf, log_folder, "orign+DA+SimpleRNN")
    
    # Experiment 7
    # orign+DA LSTM
    ModelCrossValidation(CreateLSTMModel,6, np.squeeze(x_train_DA), y_train_DA, np.squeeze(x_test),
                 y_test, skf, log_folder, "orign+DA+LSTM")

    # Experiment 8
    # orign+DA MV-CNN
    ModelCrossValidation(CreateMVCNNModel,6, x_train_DA, y_train_DA, x_test,
                y_test, skf, log_folder, "orign+DA+MV_CNN")

    # Experiment 9
    # orign+FA CNN
    ModelCrossValidation(CreateCNNModel,38, x_train_FA, y_train, x_test_FA,
                 y_test, skf, log_folder, "orign+FA+CNN")
               
    # Experiment 10
    # orign+FA RNN
    ModelCrossValidation(CreateSimpleRNNModel,38, np.squeeze(x_train_FA), y_train, np.squeeze(x_test_FA),
                 y_test, skf, log_folder, "orign+FA+SimpleRNN")
    
    # Experiment 11
    # orign+FA LSTM
    ModelCrossValidation(CreateLSTMModel,38, np.squeeze(x_train_FA), y_train, np.squeeze(x_test_FA),
                 y_test, skf, log_folder, "orign+FA+LSTM")

    # Experiment 12
    # orign+FA MV-CNN
    ModelCrossValidation(CreateMVCNNModel,38, x_train_FA, y_train, x_test_FA,
                y_test, skf, log_folder, "orign+FA+MV_CNN")
    
    # Experiment 13
    # orign+DA+FA CNN
    ModelCrossValidation(CreateCNNModel,38, x_train_DA_FA, y_train_DA, x_test_FA,
                 y_test, skf, log_folder, "orign+DA+FA+CNN")
               
    # Experiment 14
    # orign+DA+FA RNN
    ModelCrossValidation(CreateSimpleRNNModel,38, np.squeeze(x_train_DA_FA), y_train_DA, np.squeeze(x_test_FA),
                 y_test, skf, log_folder, "orign+DA+FA+SimpleRNN")
    
    # Experiment 15
    # orign+DA+FA LSTM
    ModelCrossValidation(CreateLSTMModel,38, np.squeeze(x_train_DA_FA), y_train_DA, np.squeeze(x_test_FA),
                 y_test, skf, log_folder, "orign+DA+FA+LSTM")
    
    # Experiment 16
    # orign+DA+FA MV-CNN
    ModelCrossValidation(CreateMVCNNModel,38, x_train_DA_FA, y_train_DA, x_test_FA,
                 y_test, skf, log_folder, "orign+DA+FA+MV_CNN")
    '''