# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:06:04 2018
数据增强程序
@author: John Kwok
"""

#%%
'''第一步：初始化，数据准备'''
import GetDataUtil
import numpy as np


# 加载文件
justifiedData = np.load("DataSet_NPSave/JustifiedData.npy")
print(len(justifiedData))
accelerate_data = []
collision_data = []
uniform_speed_data = []
left_turn_data = []
right_turn_data = []
# 分离各个动作到数组
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
'''
算法一：基于非主特征轴加权融合数据增强算法
'''
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
     # 将生成的新数据集保存
     np.save(savePath+Labels[label-1],AugmentatedData)
     AandO = np.concatenate((rawData,AugmentatedData))
     print(AandO.shape)
     # 将生成的数据集和原始数据集拼接后保存
     np.save("DataSet_NPSave/Aug1+orgin"+Labels[label-1],AandO)
     
     return AugmentatedData

# 测试
#DataArgument_1(accelerate_data,1,expNum=len(accelerate_data)+1)
#DataArgument_1(collision_data,2)
#DataArgument_1(uniform_speed_data,3)
#DataArgument_1(left_turn_data,4)
#DataArgument_1(right_turn_data,5)  

# 对各动作分别进行数据增强，并将增强后的数据拼接起来，AugDataAll为新生成的数据集（不包括原始数据）
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
'''
算法二：背景噪声融合数据增强算法
'''
import GetDataUtil
import numpy as np
'''
背景噪声融合数据增强算法实现函数：
rawData : 原始数据，字典
noiseData ：噪声数据，数组 
'''
def DataArgument_2(rawData,noiseData,savePath="DataSet_NPSave/NoiseAugmentatedData"):
     AugmentatedData = []
     noiseLength = noiseData.shape[1]
     print("正在生成数据，请稍后...")
     for i in range(len(rawData)):
          newData = rawData[i].copy()
          # 随机切割
          idx = np.random.randint(noiseLength-300,size = 1)

          newData["Acc"][:2] = newData["Acc"][:2]+noiseData[:2,idx[0]:idx[0]+300]
          newData["Acc"][2] = newData["Acc"][2]+noiseData[2,idx[0]:idx[0]+300]-1 # 减去噪声中的重力
          newData["Gyr"] = newData["Gyr"]+noiseData[3:6,idx[0]:idx[0]+300]
          AugmentatedData.append(newData)
     print("已完毕，共生成%d个新数据。"%(len(AugmentatedData)))
     AugmentatedData = np.array(AugmentatedData)
     np.save(savePath,AugmentatedData)
     return AugmentatedData

# 生成数据
noiseDataDic = {}

noiseData = np.concatenate(GetDataUtil.getAandG(GetDataUtil.readFile("DataSet/静止/2017-12-23-匀速")),axis = 0)
print(noiseData.shape)

rawData = np.load("DataSet_NPSave/AugAndOrginData.npy")
AugmentatedData = DataArgument_2(rawData,
                                 noiseData,
                                 savePath="DataSet_NPSave/NoiseAugmentatedData")
print(AugmentatedData.shape)

# 将通过噪声融合生成的数据和原始数据合并保存

np.save("DataSet_NPSave/NPAWF_Noise_orgin_all_5000",
        np.concatenate((rawData,AugmentatedData)))
print("文件保存在：DataSet_NPSave/NPAWF_Noise_orgin_all_5000.npy")
#%%
'''
算法三：随即剪切数据增强
rawData : 原始数据，字典
'''
def DataArgument_3(rawData,savePath="DataSet_NPSave/RandomCropAugmentatedData"):
     AugmentatedData = []
     print("正在生成数据，请稍后...")
     for data in rawData:
          newData = data.copy()
          idx = np.random.randint(150,size = 1)
          if data["Label"] == 2: # 该样本是“碰撞”
               while np.min(newData["Acc"][0]) != np.min(newData["Acc"][0,idx[0]:idx[0]+150]):
                    idx = np.random.randint(150,size = 1)
          newData["Acc"] =  newData["Acc"][:,idx[0]:idx[0]+150]
          newData["Gyr"] =  newData["Gyr"][:,idx[0]:idx[0]+150]
          AugmentatedData.append(newData)
     print("已完毕，共生成%d个新数据。"%(len(AugmentatedData)))
     AugmentatedData = np.array(AugmentatedData)
     # 插值处理
     return GetDataUtil.interpolation(AugmentatedData,
                                      sample = 300,
                                      kind ="cubic",
                                      savePath=savePath)


rawData = np.load("DataSet_NPSave/NPAWF_Noise_orgin_all_5000.npy")
AugmentatedData = DataArgument_3(rawData,
                                 savePath="DataSet_NPSave/RandomCropAugmentatedData")
print(AugmentatedData.shape)

# 将通过噪声融合生成的数据和原始数据合并保存

np.save("DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_10000",
        np.concatenate((rawData,AugmentatedData)))
print("文件保存在：DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_10000.npy")

     
#%%
import matplotlib.pyplot as plt

'''辅助函数： 生成单一样本数据波形图函数'''
def showPic(data,picName,picSavePath = "Pic_temp"):
    Labels = ["加速","碰撞","匀速","左转","右转","静止噪声"]
    axisLabel = ["X","Y","Z"]
    colorLabel = ["r",'g',"b"]
    curLabel = -1
    plt.rcParams['font.sans-serif']=['SIMHEI'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    print("正在生成图片，请稍后......")
    curLabel = Labels[data["Label"]-1]#得到中文标签
    print(curLabel)
    plt.figure(figsize=(13,10))
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
    
#%%   
#测试：
import GetDataUtil
import numpy as np
noiseDataDic = {}

noiseData = np.concatenate(GetDataUtil.getAandG(GetDataUtil.readFile("DataSet/静止/2017-12-23-匀速")),axis = 0)
print(noiseData.shape)

noiseDataDic["Acc"] = noiseData[:3]
noiseDataDic["Gyr"] = noiseData[3:6]
noiseDataDic["Label"] = 6
showPic(noiseDataDic,"noise",picSavePath = "Pic_temp")