# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:01:23 2018
实时数据接收端
@author: John Kwok
"""
from keras.models import load_model
import socket, msvcrt, time, GetDataUtil
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

# 导入模型
model = load_model('my_model.h5')
Labels = ["加速","碰撞","匀速","左转","右转"]

#ip_port = ('10.10.100.254', 59225)
ip_port = ('192.168.109.1',6688)
'''
AF = Address Family
PF = Protocol Family
AF_INET = PF_INET

SOCK_STREAM是基于TCP的，数据传输比较有保障。
SOCK_DGRAM是基于UDP的，专门用于局域网。
基于广播SOCK_STREAM 是数据流,一般是tcp/ip协议的编程
SOCK_DGRAM分是数据包,是udp协议网络编程
'''
# Socket 配置
sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#sk.bind(ip_port)
sk.settimeout(None)
sk.connect(ip_port)
data = b''
raw_data = b''
start = time.time()
print('数据正在接收...\n按"Esc"键结束程序...')

'''
数据处理函数，每次得到9900字节十六进制数据
1.转换为Acc,Gyr
2.数据预处理
3.预测模型
'''
def handleWindowData(data):
    #1.转换
    a,w = GetDataUtil.getAandG(data)
    #print(a[:,a.shape[1]-1])
    # 此处用于保证所阶段
    if a.shape[1] > 300:
        a = a[:,:300]
        w = w[:,:300]
        #print("数据长于300")
    elif a.shape[1] < 300:
        #print("数据短于300")
        for i in range(300 - a.shape[1]):
            a = np.concatenate((a,a[:,a.shape[1]-1].reshape(3,1)),axis = 1)
            w = np.concatenate((w,w[:,w.shape[1]-1].reshape(3,1)),axis = 1)
    #print(a.shape)
    #print(w.shape)
    X = np.concatenate((a,w),axis = 0)
    #print(X.shape)
    #2.数据预处理
    X = X[np.newaxis,:,:]
    X = GetDataUtil.dataProcess(X,logPrint = False)
    X = X[:,np.newaxis,:,:]
    #3.预测
    label = model.predict(X)
    t = time.strftime('%Y-%m-%d %H:%M:%S')
    print(t+"  预测结果："+Labels[np.argmax(label)]+'\n')
    
# 画布初始化
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
dataRecord = {"acc_x":np.array([]),
              "acc_y":np.array([]),
              "acc_z":np.array([]),
              "gyr_x":np.array([]),
              "gyr_y":np.array([]),
              "gyr_z":np.array([])} #用于存放六轴实时传来的数据，更新图
plt.ion()

fig = plt.figure(figsize = (16,13))
acc_x_subfig = fig.add_subplot(321)
acc_y_subfig = fig.add_subplot(323)
acc_z_subfig = fig.add_subplot(325)
gyr_x_subfig = fig.add_subplot(322)
gyr_y_subfig = fig.add_subplot(324)
gyr_z_subfig = fig.add_subplot(326)
accX_line, = acc_x_subfig.plot([], [], linestyle="-", color="r")
accY_line, = acc_y_subfig.plot([], [], linestyle="-", color="g")
accZ_line, = acc_z_subfig.plot([], [], linestyle="-", color="b")
gyrX_line, = gyr_x_subfig.plot([], [], linestyle="-", color="r")
gyrY_line, = gyr_y_subfig.plot([], [], linestyle="-", color="g")
gyrZ_line, = gyr_z_subfig.plot([], [], linestyle="-", color="b")
#用于记录已显示的样本点总数的辅助变量
count = 0

def p(a, w,maxLenth = 500):
    # points为截止到此时，接收到的数据点个数
    dataRecord["acc_x"] = np.concatenate((dataRecord["acc_x"],a[0]),axis = 0)
    dataRecord["acc_y"] = np.concatenate((dataRecord["acc_y"],a[1]),axis = 0)
    dataRecord["acc_z"] = np.concatenate((dataRecord["acc_z"],a[2]),axis = 0)
    dataRecord["gyr_x"] = np.concatenate((dataRecord["gyr_x"],w[0]),axis = 0)
    dataRecord["gyr_y"] = np.concatenate((dataRecord["gyr_y"],w[1]),axis = 0)
    dataRecord["gyr_z"] = np.concatenate((dataRecord["gyr_z"],w[2]),axis = 0)
    start = 0
    points = len(dataRecord["acc_x"])
    if(points > maxLenth):
        start = points - maxLenth
        points -= start
        global count
        count += start
        #print(start)
        dataRecord["acc_x"] = dataRecord["acc_x"][start:]
        dataRecord["acc_y"] = dataRecord["acc_y"][start:]
        dataRecord["acc_z"] = dataRecord["acc_z"][start:]
        dataRecord["gyr_x"] = dataRecord["gyr_x"][start:]
        dataRecord["gyr_y"] = dataRecord["gyr_y"][start:]
        dataRecord["gyr_z"] = dataRecord["gyr_z"][start:]
    #print(len(dataRecord["acc_y"]))

    acc_x_subfig.set_xlim(count, count+points)
    acc_y_subfig.set_xlim(count, count+points)
    acc_z_subfig.set_xlim(count, count+points)
    gyr_x_subfig.set_xlim(count, count+points)
    gyr_y_subfig.set_xlim(count, count+points)
    gyr_z_subfig.set_xlim(count, count+points)

    acc_x_subfig.set_title("加速度-X")
    acc_y_subfig.set_title("加速度-Y")
    acc_z_subfig.set_title("加速度-Z")
    gyr_x_subfig.set_title("角速度-X")
    gyr_y_subfig.set_title("角速度-Y")
    gyr_z_subfig.set_title("角速度-Z")
    # print(np.min(dataRecord["acc_x"]))
    acc_x_subfig.set_ylim(np.min(dataRecord["acc_x"]), np.max(dataRecord["acc_x"]) + 1)
    acc_y_subfig.set_ylim(np.min(dataRecord["acc_y"]), np.max(dataRecord["acc_y"]) + 1)
    acc_z_subfig.set_ylim(np.min(dataRecord["acc_z"]), np.max(dataRecord["acc_z"]) + 1)
    gyr_x_subfig.set_ylim(np.min(dataRecord["gyr_x"]), np.max(dataRecord["gyr_x"]) + 1)
    gyr_y_subfig.set_ylim(np.min(dataRecord["gyr_y"]), np.max(dataRecord["gyr_y"]) + 1)
    gyr_z_subfig.set_ylim(np.min(dataRecord["gyr_z"]), np.max(dataRecord["gyr_z"]) + 1)
    '''    
    print(points)
    print(np.size(dataRecord["acc_x"]))
    print(np.size(dataRecord["acc_y"]))
    print(np.size(dataRecord["acc_z"]))
    print(np.size(dataRecord["gyr_x"]))
    print(np.size(dataRecord["gyr_y"]))
    print(np.size(dataRecord["gyr_z"]))
    '''  
   
    #print("count:%d  points:%d"%(count,points))
    accX_line.set_data(range(count,count+points), dataRecord["acc_x"])
    accY_line.set_data(range(count,count+points), dataRecord["acc_y"])
    accZ_line.set_data(range(count,count+points), dataRecord["acc_z"])
    gyrX_line.set_data(range(count,count+points), dataRecord["gyr_x"])
    gyrY_line.set_data(range(count,count+points), dataRecord["gyr_y"])
    gyrZ_line.set_data(range(count,count+points), dataRecord["gyr_z"])
    plt.pause(0.0001)
    acc_x_subfig.figure.canvas.draw()
    acc_y_subfig.figure.canvas.draw()
    acc_z_subfig.figure.canvas.draw()
    gyr_x_subfig.figure.canvas.draw()
    gyr_y_subfig.figure.canvas.draw()
    gyr_z_subfig.figure.canvas.draw()

# 发送端模块，100HZ 每次采样生成33字节数据，
# 时间窗口如果取3s 则一次接收9900字节
while True:
    recv = sk.recv(10240)
    data += recv
    raw_data += recv
    # 动态画出六轴数据图
    if len(raw_data) >= 3300:
        a,w = GetDataUtil.getAandG(raw_data[:3300])
        raw_data = raw_data[3300:]
        # 更新图
        p(a,w)
        plt.ioff()
        plt.tight_layout()
        plt.show()
        
    # 分类预测
    if len(data) >= 9900:
        # lenth_before = len(data)
        handleWindowData(data[:9900])
        data = data[9900:]
        # lenth_after = len(data)
        # print("Before:%d  after:%d  差值：%d"%(lenth_before,lenth_after,lenth_before-lenth_after))
    # t = time.strftime('%Y-%m-%d %H:%M:%S')
    # print(t + " 数据长度：%d" % len(bytes.decode(data)))
    # 用于读取一次按键操作,如果是“退出健”ASCII = 27 则终止程序
    if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
        break



'''
file_name = '../data/raw/' + str(int(time.time()))
with open(file_name, 'wb') as fd:
    fd.write(data)
'''