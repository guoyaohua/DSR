# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:03:14 2018
批量生成图片程序（从数据集中随机选取100个样本（每个动作20个），画图）
@author: John Kwok
"""

#%% 
'''操作一：生成图片'''
import GetDataUtil
import numpy as np
from scipy import signal

# 加载文件
#orinDataSet = np.load("DataSet.npy")
#orinDataSet = np.load("DataSet_NPSave/AugDataAll.npy")
#orinDataSet = np.load("DataSet_NPSave/NoiseAugmentatedData.npy")
#orinDataSet = np.load("DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_10000.npy")
orinDataSet = np.load("DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_ACC_01_10000.npy")

#orinDataSet = np.load("JustifiedData.npy")

# Labels = ["加速","碰撞","匀速","左转","右转"]

accelerate_data = []
collision_data = []
uniform_speed_data = []
left_turn_data = []
right_turn_data = []

for data in orinDataSet:
     if data["Label"] == 1:
          accelerate_data.append(data)
     elif data["Label"] == 2:
          collision_data.append(data)
     elif data["Label"] == 3:
          uniform_speed_data.append(data)
     elif data["Label"] == 4:
          left_turn_data.append(data)
     elif data["Label"] == 5:
          right_turn_data.append(data)
# 转换为numpy
accelerate_data = np.array(accelerate_data)
collision_data = np.array(collision_data)
uniform_speed_data = np.array(uniform_speed_data)
left_turn_data = np.array(left_turn_data)
right_turn_data = np.array(right_turn_data)

''' 
shape:
(146,)
(224,)
(266,)
(196,)
(200,)
'''
print(accelerate_data.shape)
print(collision_data.shape)
print(uniform_speed_data.shape)
print(left_turn_data.shape)
print(right_turn_data.shape)
# 随机选取
accelerate_data = np.random.choice(accelerate_data, size=20)
collision_data = np.random.choice(collision_data, size=20)
uniform_speed_data = np.random.choice(uniform_speed_data, size=20)
left_turn_data = np.random.choice(left_turn_data, size=20)
right_turn_data = np.random.choice(right_turn_data, size=20)

print(accelerate_data.shape)
print(collision_data.shape)
print(uniform_speed_data.shape)
print(left_turn_data.shape)
print(right_turn_data.shape)
# 数组拼接
selectedData = np.concatenate((accelerate_data,
                               collision_data,
                               uniform_speed_data,
                               left_turn_data,
                               right_turn_data))
# (100,)
print(selectedData.shape)
# 生成图片
GetDataUtil.generatePic(selectedData,picSavePath = "Pic_All_ACC_01_100")

#%%
'''此部分为生成滤波后的波形图'''
# 初始化巴特沃思滤波器 低通 截止频率 1HZ
b,a = signal.butter(3,0.02,'low')

for i in range(len(selectedData)):
     selectedData[i]["Acc"] = signal.filtfilt(b,a,selectedData[i]["Acc"])
     selectedData[i]["Gyr"] = signal.filtfilt(b,a,selectedData[i]["Gyr"])
# 生成图片
GetDataUtil.generatePic(selectedData,picSavePath = "Pic_lowPass_100")