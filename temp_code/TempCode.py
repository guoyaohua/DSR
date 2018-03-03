# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:18:33 2018

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
orinDataSet = np.load("DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_10000.npy")

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

print(accelerate_data.shape)
print(collision_data.shape)
print(uniform_speed_data.shape)
print(left_turn_data.shape)
print(right_turn_data.shape)

count = 0
for data in accelerate_data:
     data["Acc"][0] += 0.05
     count += 1
print("已完成%d个样本增加0.05"%(count))
     
# 数组拼接
AllData_Acc_005 = np.concatenate((accelerate_data,
                                  collision_data,
                                  uniform_speed_data,
                                  left_turn_data,
                                  right_turn_data))
np.save("DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_ACC_005_10000.npy",AllData_Acc_005)

count = 0
for data in accelerate_data:
     data["Acc"][0] += 0.05
     count += 1
print("已完成%d个样本增加0.1"%(count))

# 数组拼接
AllData_Acc_01 = np.concatenate((accelerate_data,
                                  collision_data,
                                  uniform_speed_data,
                                  left_turn_data,
                                  right_turn_data))
np.save("DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_ACC_01_10000.npy",AllData_Acc_01)


#%%
import numpy as np
noiseData = np.concatenate(GetDataUtil.getAandG(GetDataUtil.readFile("DataSet/静止/2017-12-23-匀速")),axis = 0)
idx = np.random.randint(noiseData.shape[1]-300,size = 1)
print(noiseData[2,idx[0]:idx[0]+300]-1)
#%%
#dataSet = np.load("DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_10000.npy")
dataSet = np.load("DataSet_NPSave/JustifiedData.npy")
count = 0
for data in dataSet:
     if np.max(data["Gyr"]) > 200:
          print("存在%d"%(np.max(data["Gyr"])))
          count += 1
print(count)
#%%
import numpy as np
import GetDataUtil
#orginDataSet = GetDataUtil.saveDataToNP("DataSet/trim",savePath = "DataSet_NPSave/DataSet.npy")
#print(len(orginDataSet))
#np_JutifiedDataSet = GetDataUtil.interpolation(orginDataSet,sample = 300,kind ="cubic",savePath="DataSet_NPSave/JustifiedData.npy")
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


DataSet = np.load("DataSet_NPSave/RandomCrop_NPAWF_Noise_orgin_all_10000.npy")
count = 0
for data in DataSet:
     if np.max(data["Gyr"]) > 200:
          print("存在%d"%(np.max(data["Gyr"])))
          count += 1
          #showPic(data,"pic_%d"%(count),picSavePath = "Pic_temp")
print(count)

#%%
a = {"a":np.array([1,2,3]),"b":np.array([4,5,6])}
b = {"a":np.array([1,2,3]),"b":np.array([4,5,6])}
c = {"a" : np.array([3,2,2]),"b" : np.array([5,4,3])}
print(c)
d = c.copy()
c["a"][1] = 100000
print(d == c)
print(d)
print(c)
#%%
import matplotlib.pyplot as plt
from scipy import interpolate
mu, sigma = 0, 0.1
#s = np.random.normal(mu, sigma, 1000)
plt.rcParams['font.sans-serif']=['SIMHEI'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
F = interpolate.interp1d(range(0,len(s[300:800])),s[300:800],axis = 0,kind ="cubic")
X_new =np.linspace(0,len(s[300:800])-1,1000)
s_New=F(X_new)
plt.figure(0,figsize=(12,2))
plt.plot(range(1000),s_New,linewidth = '1',color='lime')
plt.title("剪切数据")
plt.show()



#%%
data = np.load("DataSet_NPSave/AugmentatedData左转.npy")
plt.rcParams['font.sans-serif']=['SIMHEI'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure(0,figsize=(5,2))
#plt.plot(range(300),0.8*data[32]["Acc"][1],linewidth = '1',color='orangered')
#plt.plot(range(300),0.2*data[56]["Acc"][1],linewidth = '1',color='lime')
plt.plot(range(300),0.8*data[32]["Acc"][1]+0.2*data[56]["Acc"][1],linewidth = '1',color='red')
plt.title("新数据")
plt.show()
