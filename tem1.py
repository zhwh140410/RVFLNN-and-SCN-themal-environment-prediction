# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 15:41:26 2017

@author: acer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn import svm
import time
from sklearn.preprocessing import MinMaxScaler
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\STKAITI.ttf")

def sigm(z):
    return 1.0/(1.0+np.exp(-z))
'''data_E = {'node_numbers':[20,50,100,200,400,600,1000,1500,2000],
          'E_train':[3.836,3.647,3.522,3.438,3.393,3.333,3.215,3.107,3.135],
          'E_test':[5.068,4.654,4.521,4.419,4.422,4.611,4.517,4.445,4.529]}
frame = pd.DataFrame(data_E)    
frame.plot(figsize=(6,3),x = 'node_numbers',kind = 'bar',)
plt.legend(bbox_to_anchor=(0.24, 1.0),prop={'size':10})
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel(u'temperature[\u2103]',fontproperties='SimHei')
plt.figure(dpi=240)
'''
x = np.arange(2160)
data = pd.read_csv('20170810.csv')#温度读取
df = pd.read_csv('out2m.csv')#人流读取
df1 = df['dx'][0:2160]
df2 = df['sx'][0:2160]

tem1 = data['1T'][0:2160]
tem2 = data['2T'][0:2160]
tem3 = data['3T'][0:2160]
tem4 = data['4T'][0:2160]
data_2 = {'tem1':tem1,'tem2':tem2,'tem3':tem3,'tem4':tem4}
data_22 = pd.DataFrame(data_2)
data_22.plot()

plt.legend(loc='best')
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties=font)
plt.ylabel('temperature',fontproperties=font)

#巴特沃斯滤波
b,a = ss.butter(3,0.08,'low')  
t1 = ss.filtfilt(b,a,tem1)-4
t2 = ss.filtfilt(b,a,tem2)-4
t3 = ss.filtfilt(b,a,tem3)-4
t4 = ss.filtfilt(b,a,tem4)-4 
df3 = ss.medfilt(df1,19)
t2 = t2-0.2
'''#中值滤波
tem_1 = ss.medfilt(tem1,13)
tem_2 = ss.medfilt(tem2,13)
tem_3 = ss.medfilt(tem3,19)
tem_4 = ss.medfilt(tem4,13)
tem_11 = np.diff(tem_1)
tem_22 = np.diff(tem_2)
tem_33 = np.diff(tem_3)
tem_44 = np.diff(tem_4)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(t1,lw=1.5,color='b')
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(tem_1,lw=1.5,label="med",color='g')
#多项式拟合
a1 = np.polyfit(x,tem_1,44)
a11 = np.polyval(a1,x)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(a11,lw=1.5,label="nihe",color='r')'''

'''data_1 = {'t1':t1,'t2':t2,'t3':t3,'t4':t4}
data_11 = pd.DataFrame(data_1)
data_11.to_csv('done.csv')
data_11.plot(figsize=(12.5,4))
plt.legend(bbox_to_anchor=(0.08, 1.0),prop={'size':10})
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel(u'temperature[\u2103]',fontproperties='SimHei')
plt.figure(dpi=240)
'''
fig3 = plt.figure(figsize=(12.5,4))
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(t1,'b',label='t1',linewidth='1')
ax3.plot(t2,'g',label='t2',linewidth='1')
ax3.plot(t3,'r',label='t3',linewidth='1')
ax3.plot(t4,'m',label='t4',linewidth='1')
ax3.set_xlim([0,2160])
plt.legend(bbox_to_anchor=(0.1, 1),prop={'size':10})
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel(u'temperature[\u2103]',fontproperties='SimHei')
plt.grid(True)
plt.figure(dpi=1200)

#画人流数据
fig4 = plt.figure(figsize=(12.5,4))
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(df1,color='b')
ax4.set_xlim([0,2160])
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel('people counting[number]',fontproperties='SimHei')
plt.grid(True)
plt.figure(dpi=1200)
#地铁进站频率
df_3 = df3.copy()
df_3[(df_3>0)&(df_3<=10)] = 10
df_3[(df_3>10)&(df_3<=17)] = 15
df_3[(df_3>17)&(df_3<=34)] = 20
df_3[(df_3>34)&(df_3<=53)] = 25
df_3[df_3>53] = 30
df_3[2103] = 20
fig5 = plt.figure(figsize=(12.5,4))
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(df3,color='b')
ax5.set_xlim([0,2160])
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel('$P_flow[number]$',fontproperties='SimHei')
plt.legend(loc='best')

plt.grid(True)
plt.figure(dpi=240)
fig6 = plt.figure(figsize=(12.5,4))
ax6 = fig6.add_subplot(1,1,1)
ax6.plot(df_3,color='b')
ax6.set_xlim([0,2160])
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel('$F_train[hour]$',fontproperties='SimHei')
plt.figure(dpi=1200)
'''#温差
delta = np.mean(tem_1+tem_2+tem_3+tem_4)/4
tem_11 = tem_1 - delta
tem_22 = tem_2 - delta
tem_33 = tem_3 - delta
tem_44 = tem_4 - delta
data_2 = {'Δt1':tem_11,'Δt2':tem_22,'Δt3':tem_33,'Δt4':tem_44}
data_22 = pd.DataFrame(data_2)
data_22.plot()
plt.legend(loc='best')
plt.grid(True)
plt.xlabel(u'时间/2分钟',fontproperties=font)
plt.ylabel(u'温度差',fontproperties=font)'''
#归一化
tem_ = np.vstack((t1,t2,t3,t4))
'''min_max_scaler = MinMaxScaler()
x1 = min_max_scaler.(t1)
x2 = min_max_scaler.fit_transform(df3)
x3 = min_max_scaler.fit_transform(t2)
x4 = min_max_scaler.fit_transform(t4)
x5 = min_max_scaler.fit_transform(df_3)
y1 = min_max_scaler.fit_transform(t3)'''
x1 = (t1-np.min(tem_))/(np.max(tem_)-np.min(tem_))
x2 = (t2-np.min(tem_))/(np.max(tem_)-np.min(tem_))

x3 = (t4-np.min(tem_))/(np.max(tem_)-np.min(tem_))

y1 = (t3-np.min(tem_))/(np.max(tem_)-np.min(tem_))
x4 = (df3-np.min(df3))/(np.max(df3)-np.min(df3))
x5 = (df_3-np.min(df_3))/(np.max(df_3)-np.min(df_3))
'''fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)
ax6.plot(x1,color='b',label="x1")
ax6.plot(x2,color='g',label="x2")
ax6.plot(x3,color='y',label="x3")
ax6.plot(x4,color='k',label="x4")
ax6.plot(x5,color='m',label="x5")
ax6.plot(y1,color='r',label="y1")
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel('Normalization',fontproperties='SimHei')
plt.legend(loc='upper left')
plt.grid(True)
plt.figure(dpi=240)'''
#截断数据，三天
x11 = x1[0:720]
x21 = x2[0:720]
x31 = x3[0:720]
x41 = x4[0:720]
x51 = x5[0:720]
y11 = y1[0:720]
x12 = x1[720:1440]
x22 = x2[720:1440]
x32 = x3[720:1440]
x42 = x4[720:1440]
x52 = x5[720:1440]
y12 = y1[720:1440]
x13 = x1[1440:2160]
x23 = x2[1440:2160]
x33 = x3[1440:2160]
x43 = x4[1440:2160]
x53 = x5[1440:2160]
y13 = y1[1440:2160]
x_train1 = np.vstack((x11,x21,x31,x41,x51))
x_train2 = np.vstack((x12,x22,x32,x42,x52))
x_train = np.hstack((x_train1,x_train2))
x_test = np.vstack((x13,x23,x33,x43,x53))
y_train1 = y11
y_train2 = y12
y_train = np.hstack((y_train1,y_train2))
y_test = y13
x_traintest = np.hstack((x_train,x_test))
'''x_test = x_test.T
x_training = x_train.T
y_training = y_train.reshape((1440,1))
clf = svm.SVR(kernel='rbf',gamma=0.2,shrinking=True,tol=0.0001,C=2.0,verbose=False,cache_size=200,max_iter=-1)
clf.fit(x_training,y_training)
'''
A = 5  
B = 200
C = 1
l =1
D = 720
'''
yout = clf.predict(x_test)
fig7 = plt.figure()
ax7 = fig7.add_subplot(1,1,1)
ax7.plot(y_test,color='b',label='real')
ax7.plot(yout,color='g',label='predict')
plt.legend(loc='upper left')
plt.grid(True)

err = np.sum(np.square(y_test - yout))'''
#建立神经网络

net = np.array([A,B,C])
w = np.random.randn(B,A)
b = np.random.randn(C,B)
start = time.time()
n = np.size(x_train,1)
h_out = sigm(np.dot(x_train.T,w.T)+np.tile(b,(n,1)))
beta_out = np.linalg.inv(np.dot(h_out.T,h_out) + l*np.eye(B))
beta_out1 = np.dot(beta_out,h_out.T)
beta_out11 = np.dot(beta_out1,y_train)
stop = time.time()
t = stop - start
#测试
y_out1 = []
y_out2 = []
y_out3 = []
#测试
for i in range(2160):
    x = x_traintest[:,i]
    fai = sigm(np.dot(w,x)+b)
    z1 = np.dot(fai,beta_out11)
    z11 = z1[0]
    y_out1.append(z11)
y_out11 = np.array(y_out1) 
#训练
'''
for i in range(D):
    x = x_train1[:,i]
    fai = sigm(np.dot(w,x)+b)
    z2 = np.dot(fai,beta_out11)
    z22 = z2[0]
    y_out2.append(z22)
y_out22 = np.array(y_out2) 
for i in range(D):
    x = x_train2[:,i]
    fai = sigm(np.dot(w,x)+b)
    z3 = np.dot(fai,beta_out11)
    z33 = z3[0]
    y_out3.append(z33)
y_out33 = np.array(y_out3) 
'''
'''
fig7 = plt.figure()
ax7 = fig7.add_subplot(1,1,1)
ax7.plot(y_test,color='b',label='real')
ax7.plot(y_out11,color='g',label='predict')

plt.xlabel('TIME[2minutes]',fontproperties=font)
plt.ylabel('Normalization',fontproperties=font)
plt.legend(bbox_to_anchor=(0.28, 1.015))
plt.grid(True)

e0 = (y_test - y_out11)/y_out11
err0 = np.abs(e0)
errr0 = np.sum(err0)/720
#训练误差
y_real1 = y11*(np.max(t3)-np.min(t3))+np.min(t3)
y_train11 = y_out22*(np.max(t3)-np.min(t3))+np.min(t3)

y_real2 = y12*(np.max(t3)-np.min(t3))+np.min(t3)
y_train22 = y_out33*(np.max(t3)-np.min(t3))+np.min(t3)
'''
#测试误差
y_test_tem = y1*(np.max(t3)-np.min(t3))+np.min(t3)
y_out2_tem = y_out11*(np.max(t3)-np.min(t3))+np.min(t3)
fig8 = plt.figure(figsize=(12.5,4))
ax8 = fig8.add_subplot(1,1,1)
ax8.plot(y_test_tem,color='b',linestyle='--',label='experiment')
ax8.plot(y_out2_tem,color='g',label='predict')
ax8.set_xlim([0,2160])
ax8.set_ylim()
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel(u'temperature[\u2103]',fontproperties='SimHei')
plt.legend(bbox_to_anchor=(0.15, 1.0),prop={'size':10})
plt.figure(dpi=1200)
'''
fig9 = plt.figure(figsize=(8,4))
ax9 = fig9.add_subplot(1,1,1)
ax9.plot(y_real1,color='b',label='experiment')
ax9.plot(y_train11,color='g',label='predict')
ax9.set_xlim([0,740])
ax9.set_ylim([29.5,33])
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel(u'temperature[\u2103]',fontproperties='SimHei')
plt.legend(bbox_to_anchor=(0.23, 1.0),prop={'size':10})
plt.figure(dpi=480)

fig10 = plt.figure(figsize=(8,4))
ax10 = fig10.add_subplot(1,1,1)
ax10.plot(y_real2,color='b',label='experiment')
ax10.plot(y_train22,color='g',label='predict')
ax10.set_xlim([0,740])
ax10.set_ylim([29.5,33])
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel(u'temperature[\u2103]',fontproperties='SimHei')
plt.legend(bbox_to_anchor=(0.23, 1.0),prop={'size':10})
plt.figure(dpi=480)
'''

e1 = (y_test_tem - y_out2_tem)/y_out2_tem
err1 = np.abs(e1)
etrain = np.sum(err1[0:1440])/1440
evalidation = np.sum(err1[1441:1800])/360
etest = np.sum(err1[1801:2160])/360
evt = (evalidation+etest)/2
fig11 = plt.figure(figsize=(12.5,4))
ax11 = fig11.add_subplot(1,1,1)
ax11.plot(err1*100)

ax11.set_xlim([0,2160])
ax11.set_ylim()
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel('relative error[%]',fontproperties='SimHei')
plt.legend(bbox_to_anchor=(0.1, 1.0),prop={'size':10})
plt.figure(dpi=1200)
'''
e2 = (y_real1 - y_train11)/y_train11
err2 = np.abs(e2)
errr2 = np.sum(err2)/720


e3 = (y_real2 - y_train22)/y_train22
err3 = np.abs(e3)
errr3 = np.sum(err3)/720

errr = (errr2+errr3)/2
'''