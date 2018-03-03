# coding = utf-8 
# -*- coding: utf-8 -*-
# 2018年2月18日 17:40:55
# 优化数据解析方法
"""
Created on Tue Jan 16 13:01:23 2018
数据文件读取工具包
@author: John Kwok
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#%%
'''加载文件'''
def readFile(filePath):
    file = open(filePath,'rb')
    data = file.read()
    file.close()
    return data
#%%
'''得到所有文件名,所在目录名'''
def getFileList(rootPath):
    fileList = []
    for root, dirs, files in os.walk(rootPath,topdown = False):#优先遍历子目录
        if len(files) > 0:
            fileList.append((root,files));
    return fileList
#%%
'''把得到的加速度,角速度,标签，写入到文件中'''
def write2file(X,Y,X_path,Y_path):
    np.savetxt(X_path,X,delimiter=',')
    np.savetxt(Y_path,Y,delimiter=',')
#%% 
'''旧版本数据解析方法：
从文件中获取加速度和角速度'''
def getAandG_old(data):
    data_len = len(data)
    a = [[] for i in range(3)]  # 加速度
    w = [[] for i in range(3)]  # 角速度
    index = 0
    while index < data_len:  
        if data[index] == 0x55:  # 包头
            if index + 7 < data_len: # 该数据包完整
                temp_a = data[index+3]
                temp_b = data[index+5]
                temp_c = data[index+7]
                if data[index+3] > 127:  # 说明是负数
                    temp_a = temp_a - 256
                if data[index+5] >127:
                    temp_b = temp_b - 256
                if data[index+7] >127:
                    temp_c = temp_c - 256
                if data[index+1] == 0x51: # 加速度输出
                    a[0].append(temp_a*256+data[index+2]) # x轴加速度
                    a[1].append(temp_b*256+data[index+4]) # y轴加速度
                    a[2].append(temp_c*256+data[index+6]) # z轴加速度
                    index += 11
                    continue
                elif data[index+1] == 0x52: # 角速度输出
                    w[0].append(temp_a*256+data[index+2]) # x轴角速度
                    w[1].append(temp_b*256+data[index+4]) # y轴角速度
                    w[2].append(temp_c*256+data[index+6]) # z轴角速度
                    index += 11
                    continue
                elif data[index+1] == 0x53: # 角度输出
                    index += 11
                else:  # 该数据包损坏
                    index += 1
            else:  # 没有完整的数据了
                break
        else:  # 索引值+1直至寻找到包头
            index += 1
    a = np.array(a)
    w = np.array(w)
    a = a / 32768.0 * 16    #单位为g
    w = w / 32768.0 * 2000  # 单位为°/s
    #数据对齐
    len_diff = len(a[0])-len(w[0])
    if len_diff !=0:
        if len_diff>0:#加速度数据长
            w = np.hstack((w,np.zeros((3,len_diff))))
        elif len_diff<0:#角速度数据长
            a = np.hstack((a,np.zeros((3,-len_diff))))   
    if len(a[0])!=len(w[0]):
        print("数据未对齐")#判断是否对齐
    
    #print(a.shape)
    #print(w.shape)
    return a,w
#%%
'''
新版数据解析函数
'''
def getAandG(data):
    data_len = len(data)
    a = [[] for i in range(3)]  # 加速度
    w = [[] for i in range(3)]  # 角速度
    index = 0
    while index < data_len :
        if data[index] == 0x55: # 包头
            if index + 7 < data_len: # 该数据包完整
                if data[index+1] == 0x51: # 该数据包为加速度数据
                    a[0].append((data[index+3]<<8|data[index+2]))
                    a[1].append((data[index+5]<<8|data[index+4]))
                    a[2].append((data[index+7]<<8|data[index+6]))
                    index += 11
                    continue
                elif data[index+1] == 0x52: #该数据包为角速度数据
                    w[0].append((data[index+3]<<8|data[index+2]))
                    w[1].append((data[index+5]<<8|data[index+4]))
                    w[2].append((data[index+7]<<8|data[index+6]))
                    index += 11
                    continue
                elif data[index+1] == 0x53: # 该数据包为角度数据
                    index += 11
                else:  # 数据包损坏
                    index += 1
            else: # 该数据包不完整,丢弃该数据
                break
        else:
            index += 1 # 索引值+1直至寻找到包头
            
    a = np.array(a,dtype = 'int16')
    w = np.array(w,dtype = 'int16')
    a = a / 32768 * 16    #单位为g
    w = w / 32768.0 * 2000  # 单位为°/s
    #数据对齐,将短的轴按末尾数值补齐
    len_diff = len(a[0])-len(w[0])
    if len_diff !=0:
        if len_diff>0:# 加速度数据长
            w = np.concatenate((w,np.full((3,len_diff),w[:,len(w[0])-1].reshape(3,1))),axis = 1)
        elif len_diff<0:# 角速度数据长
            a = np.concatenate((a,np.full((3,-len_diff),a[:,len(a[0])-1].reshape(3,1))),axis = 1)
    if len(a[0])!=len(w[0]):
        print("数据未对齐")#判断是否对齐
    #print(a.shape)
    #print(w.shape)
    return a,w
#%%
'''
读取指定目录原始数据，将其整合成为numpy，
'''
def saveDataToNP(rootPath ,savePath = "DataSet_NPSave/DataSet.npy"):
    DataSet = []
    Labels = {"加速":1,"碰撞":2,"匀速":3,"左转":4,"右转":5}
    fileList = getFileList(rootPath)
    print("正在生成文件，请稍后...")  
    for root,files in fileList:
        print(root.split('\\')[-1])
        label = Labels[root.split('\\')[-1]]
        for filePath in files:
            filePath = root+ '/' + filePath
            fileData = readFile(filePath)
            if(len(fileData) == 0):
                continue
            A,G = getAandG(fileData)
            DataSet.append({"Acc":A,"Gyr":G,"Label":label})
    DataSet = np.array(DataSet)   
    np.save(savePath,DataSet)
    print("生成文件完毕！")  
    return DataSet
#%%
'''
生成图片
'''
def generatePic(DataSet,picSavePath = "Pic"):
    Labels = ["加速","碰撞","匀速","左转","右转"]
    axisLabel = ["X","Y","Z"]
    colorLabel = ["r",'g',"b"]
    curLabel = -1
    count = 1
    plt.rcParams['font.sans-serif']=['SIMHEI'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    print("正在生成图片，请稍后......")
    for data in DataSet:
        if curLabel != Labels[data["Label"]-1]:
            count = 1
            curLabel = Labels[data["Label"]-1]#得到中文标签
        else:
            count += 1
        plt.figure(curLabel+"%d"%(count),figsize=(18,12))
        plt.suptitle(curLabel,fontsize=40,x=0.5,y=0.97)
        for i in range(3):
            plt.subplot(3,2,i*2+1)
            plt.title("加速度-"+axisLabel[i])
            plt.plot(range(len(data["Acc"][i])),data["Acc"][i],color=colorLabel[i])
            plt.subplot(3,2,i*2+2)
            plt.title("角速度-"+axisLabel[i])
            plt.plot(range(len(data["Gyr"][i])),data["Gyr"][i],color=colorLabel[i])
        plt.gcf().savefig(picSavePath+"/"+curLabel+"_%d.png"%(count),dpi = 200,bbox_inches='tight')
        print("正在生成.."+curLabel+"_%d.png"%(count))
        plt.close('all')
    print("生成图片完毕！")
#%%
'''
插值函数：
sample 为插值后的采样点数
kind 为插值方式
    #"nearest","zero"为阶梯插值  
    #slinear 线性插值  
    #"quadratic","cubic" 为2阶、3阶B样条曲线插值
savaPath 为数据保存路径
'''
def interpolation(originData,sample = 300,kind ="cubic",savePath="DataSet_NPSave/JustifiedData.npy"):
    JutifiedDataSet = []
    print("正在生成转换文件，请稍后...") 
    for data in originData:
        F_Acc = interpolate.interp1d(range(0,len(data["Acc"][0])),data["Acc"],axis = 1,kind = kind)
        F_Gyr = interpolate.interp1d(range(0,len(data["Gyr"][0])),data["Gyr"],axis = 1,kind = kind)
        X_new =np.linspace(0,len(data["Acc"][0])-1,sample)
        Acc_New=F_Acc(X_new)
        Gyr_New=F_Gyr(X_new)
        JutifiedDataSet.append({"Acc":Acc_New,"Gyr":Gyr_New,"Label":data["Label"]})
    np_JutifiedDataSet = np.array(JutifiedDataSet)
    np.save(savePath,np_JutifiedDataSet)
    print("转换完毕，已生成文件"+savePath) 
    return np_JutifiedDataSet
#%%
'''
载入文件插值、对齐好的数据文件，并返回训练集、测试集
参数：
    dataPath 数据文件地址
    test_size 测试集占比
输出：
    X_train,X_test, y_train, y_test
    shape = (6,300)
'''
from sklearn.utils import shuffle
def getTrainTestSet(dataPath = "JustifiedData.npy",test_size = 0.1):
    X = []
    Y = []
    dataSet = np.load(dataPath)
    for data in dataSet:
        S = data["Acc"]
        S = np.concatenate((S,data["Gyr"]),axis = 0)
        #S = S.reshape((6,300))
        X.append(S)
        Y.append(data["Label"])
#     print(np.array(X).shape)
#     print(np.array(Y).shape)
    X = np.array(X)
    Y = np.array(Y)
    np.save("orin-X",X)
    np.save("orin-Y",Y)
    #打乱顺序
    X,Y = shuffle(X,Y,random_state=0)
#     X_train, X_test, y_train, y_test
    return train_test_split(X,Y,test_size = test_size,random_state = 0)

#%%
'''
数据初始化函数:
    第一步：将原始数据文件整合保存在npy数组文件中。得到“DataSet.npy”
    第二步：使用插值法，使原始数据长度对齐，默认采用300为目标长度(即时间窗口3s)
    第三步：获取数据，并且归一化、标准化
【返回值】：X_train, X_test, y_train, y_test
'''
def dataInit():
    # 第一步
    saveDataToNP("DataSet/trim")
    # 第二步
    # 生成文件“JustifiedData.npy”，并且返回data
    # [data]：字典集合，{"Acc":A,"Gyr":G,"Label":label}
    interpolation(np.load("DataSet.npy"))
    # 第三步
    X_train, X_test, y_train, y_test = getTrainTestSet()
    # 数据预处理（此处只使用了归一化、标准化，Flatten）
    # 后续研究可以加入其他预处理：1.滤波 2.稀疏滤波 3.特征升维 等
    #X_train = preprocessing.scale(X_train.reshape(-1,1800))
    #X_test = preprocessing.scale(X_test.reshape(-1,1800))
    X_train = dataProcess(X_train)
    X_test = dataProcess(X_test)
    print("数据初始化完毕！")
    return X_train, X_test, y_train, y_test

'''
数据预处理函数（旧版本，需要改进）
1.特征构造
2.标准化
'''
def dataProcess(data,logPrint = True):
    if(logPrint):
        print("数据处理中，请稍后...")
       
    # 特征构造
    x_2 = np.power(data, 2)
    x_3 = np.power(data, 3)
    sin = np.sin(data)
    cos = np.cos(data)
    data = np.concatenate((data,x_2,x_3,sin,cos),axis = 1)
    if(logPrint):
        print("特征构造完毕！")
      
    # 标准化
    #print(data.shape)
    mean = np.mean(data,axis = (1,2))
    std = np.std(data,axis = (1,2))
    mean = mean[:,np.newaxis,np.newaxis]
    std = std[:,np.newaxis,np.newaxis]
    data = (data-mean)/std
    #data = data-mean
    if(logPrint):
        print("数据标准化完毕！(未去标准差)")
        print("数据形状：")
        print(data.shape)
    return data