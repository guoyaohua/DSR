# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:10:49 2018
主程序：用于各部分操作的调用
@author: John Kwok
"""
#%%
'''操作二：数据增强'''
import GetDataUtil
import numpy as np
import math

# 加载文件
orinDataSet = np.load("DataSet.npy")

justifiedData = GetDataUtil.interpolation(orinDataSet,
                                          sample = 300,
                                          kind ="cubic",
                                          savePath="JustifiedData.npy")

accelerate_data = []
collision_data = []
uniform_speed_data = []
left_turn_data = []
right_turn_data = []

for data in justifiedData:
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

print(accelerate_data.shape)
print(collision_data.shape)
print(uniform_speed_data.shape)
print(left_turn_data.shape)
print(right_turn_data.shape)
#%%
import matplotlib.pyplot as plt
# 生成单一样本数据波形图函数
def showPic(data,picName,picSavePath = "Pic_temp"):
    Labels = ["加速","碰撞","匀速","左转","右转"]
    axisLabel = ["X","Y","Z"]
    colorLabel = ["r",'g',"b"]
    curLabel = -1
    plt.rcParams['font.sans-serif']=['SIMHEI'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    print("正在生成图片，请稍后......")
    curLabel = Labels[data["Label"]-1]#得到中文标签
    print(curLabel)
    plt.figure(figsize=(12,6))
    plt.suptitle(curLabel,fontsize=40,x=0.5,y=0.97)
    for i in range(3):
         plt.subplot(3,2,i*2+1)
         plt.title("加速度-"+axisLabel[i])
         plt.plot(range(len(data["Acc"][i])),data["Acc"][i],color=colorLabel[i])
         plt.subplot(3,2,i*2+2)
         plt.title("角速度-"+axisLabel[i])
         plt.plot(range(len(data["Gyr"][i])),data["Gyr"][i],color=colorLabel[i])
    plt.gcf().savefig(picSavePath+"/"+picName+".png",dpi = 120,bbox_inches='tight')
    print("生成图片完毕！")
    plt.close('all')
#测试
    '''
#print(accelerate_data[0]["Label"])
showPic(accelerate_data[3],"加速_3")
showPic(accelerate_data[9],"加速_9")
data = accelerate_data[1]
data["Acc"] = 0.5*accelerate_data[3]["Acc"]+0.5*accelerate_data[9]["Acc"]
data["Gyr"] = 0.5*accelerate_data[3]["Gyr"]+0.5*accelerate_data[9]["Gyr"]
data["Acc"][0] = accelerate_data[3]["Acc"][0]
showPic(data,"合成——3+9")
'''
#%%
def DataArgument_1(rawData,label,expNum=500,weight=0.8,savePath="DataSet_NPSave/AugmentatedData"):
     AugmentatedData = []
     rawLenth = len(rawData)
     augNum = expNum - rawLenth
     Labels = ["加速","碰撞","匀速","左转","右转"]
     print(rawLenth)
     print("正在生成新数据，请稍后...")
     while len(AugmentatedData)<augNum:
          
          idx = np.random.randint(rawLenth,size = 2)
          # 注意：这里一定是copy()
          data = rawData[idx[0]].copy()
          data["Acc"] = weight*np.array(rawData[idx[0]]["Acc"])+(1-weight)*np.array(rawData[idx[1]]["Acc"])
          data["Gyr"] = weight*rawData[idx[0]]["Gyr"]+(1-weight)*rawData[idx[1]]["Gyr"]
          if label == 1:#加速 主特征轴 ACC-X     
               data["Acc"][0] = rawData[idx[0]]["Acc"][0]
          elif label == 2:#碰撞 主特征轴 ACC-X Gyr-Y
               data["Acc"][0] = rawData[idx[0]]["Acc"][0]
               data["Gyr"][1] = rawData[idx[0]]["Gyr"][1]
#          elif label == 3:#匀速 
               #主特征轴 无
          elif label == 4:#左转 主特征轴 ACC-Y Gyr-Z
               data["Acc"][1] = rawData[idx[0]]["Acc"][1]
               data["Gyr"][2] = rawData[idx[0]]["Gyr"][2]
          elif label == 5:#右转 主特征轴 ACC-Y Gyr-Z
               data["Acc"][1] = rawData[idx[0]]["Acc"][1]
               data["Gyr"][2] = rawData[idx[0]]["Gyr"][2]
          #将生成的数据加入集合中
          AugmentatedData.append(data)
          #print("已完成%d%%"%(int(len(AugmentatedData)*100/augNum)))
     print("数据增强已完成，目前数据个数%d"%(len(AugmentatedData)))
     AugmentatedData = np.array(AugmentatedData)
     np.save(savePath+Labels[label-1],AugmentatedData)
     AandO = np.concatenate((rawData,AugmentatedData))
     print(AandO.shape)
     np.save("DataSet_NPSave/Aug1+orgin"+Labels[label-1],AandO)
     
     return AugmentatedData

# 测试
#DataArgument_1(accelerate_data,1,expNum=len(accelerate_data)+1)
#DataArgument_1(collision_data,2)
#DataArgument_1(uniform_speed_data,3)
#DataArgument_1(left_turn_data,4)
#DataArgument_1(right_turn_data,5)  

AugDataAll = np.concatenate((DataArgument_1(accelerate_data,1),
                             DataArgument_1(collision_data,2),
                             DataArgument_1(uniform_speed_data,3),
                             DataArgument_1(left_turn_data,4),
                             DataArgument_1(right_turn_data,5)))
AugAndOrginData = np.concatenate((np.load("DataSet_NPSave/Aug1+orgin加速.npy"),
                                  np.load("DataSet_NPSave/Aug1+orgin碰撞.npy"),
                                  np.load("DataSet_NPSave/Aug1+orgin匀速.npy"),
                                  np.load("DataSet_NPSave/Aug1+orgin左转.npy"),
                                  np.load("DataSet_NPSave/Aug1+orgin右转.npy"),))
print(AugDataAll.shape)
print(AugAndOrginData.shape)
np.save("DataSet_NPSave/AugDataAll",AugDataAll)
np.save("DataSet_NPSave/AugAndOrginData",AugAndOrginData)

#%% 
'''操作一：生成图片'''
import GetDataUtil
import numpy as np
from scipy import signal
import math
# 加载文件
#orinDataSet = np.load("DataSet.npy")
orinDataSet = np.load("DataSet_NPSave/AugDataAll.npy")
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
GetDataUtil.generatePic(selectedData,picSavePath = "Pic_Aug_100")

#%%
# 初始化巴特沃思滤波器 低通 截止频率 1HZ
b,a = signal.butter(3,0.02,'low')

for i in range(len(selectedData)):
     selectedData[i]["Acc"] = signal.filtfilt(b,a,selectedData[i]["Acc"])
     selectedData[i]["Gyr"] = signal.filtfilt(b,a,selectedData[i]["Gyr"])
# 生成图片
GetDataUtil.generatePic(selectedData,picSavePath = "Pic_lowPass_100")