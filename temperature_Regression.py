# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:18:06 2018

@author: acer
"""

import SCN as scn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as scio
from sklearn.metrics import confusion_matrix
import scipy.signal as ss

def transfer(X):
    sh= np.shape(X)
    m = sh[0]
    A = np.zeros((m))
    for i in range(m):
         A[i] = np.argmax(X[i,:])
    return A   


#data = scio.loadmat('C:\\SCN_release_v1\\Demo_Data.mat')

data = pd.read_csv('20170810.csv')
df = pd.read_csv('out2m.csv')
 
df1 = df['dx'][0:2160].values#人流数据


tem1 = data['1T'][0:2160].values
tem2 = data['2T'][0:2160].values
tem3 = data['3T'][0:2160].values
tem4 = data['4T'][0:2160].values



#巴特沃斯滤波
b,a = ss.butter(3,0.08,'low')  
t1 = ss.filtfilt(b,a,tem1)-4
t2 = ss.filtfilt(b,a,tem2)-4
t3 = ss.filtfilt(b,a,tem3)-4
t4 = ss.filtfilt(b,a,tem4)-4 
df3 = ss.medfilt(df1,19)
t2 = t2-0.2

tem_ = np.vstack((t1,t2,t3,t4))
x1 = (t1-np.min(tem_))/(np.max(tem_)-np.min(tem_))
x2 = (t2-np.min(tem_))/(np.max(tem_)-np.min(tem_))
x3 = (t4-np.min(tem_))/(np.max(tem_)-np.min(tem_))
y1 = (t3-np.min(tem_))/(np.max(tem_)-np.min(tem_))
x4 = (df3-np.min(df3))/(np.max(df3)-np.min(df3))

x = np.vstack((x1,x2,x3,x4))
y = y1

X = x[:,0:1440].T
T = y[0:1440].reshape((1440,1))

X2 = x[:,1440:2160].T
T2 = y[1440:2160].reshape((720,1))

'''

T = data['T']
T2 = data['T2']
X = data['X']
X2 = data['X2']

'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(T,color='b',linestyle='--',label='experiment')
ax.plot(X)
ax.set_xlim()
ax.set_ylim()
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel(u'temperature[\u2103]',fontproperties='SimHei')
plt.legend()


fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(X2)
ax1.plot(T2,color='g',linestyle='--',label='predict')
ax1.set_xlim()
ax1.set_ylim()
plt.grid(True)
plt.xlabel('TIME[2minutes]',fontproperties='SimHei')
plt.ylabel(u'temperature[\u2103]',fontproperties='SimHei')
plt.legend()

xx = np.arange(2160)

L_max = 250
tol = 0.001
T_max = 350
Lambdas = [0.5,1,5,10,30,50,100,150,200,250]
nB = 1
r = [0.9,0.99,0.999,0.9999,0.99999,0.999999]


M = scn.SCN(L_max,tol,T_max,Lambdas,nB,r)

per_Error = M.Regression(X,T)
print(M)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(per_Error,color='b',linestyle='--',label='experiment')

ax2.set_xlim()
ax2.set_ylim()
plt.grid(True)
plt.xlabel('L',fontproperties='SimHei')
plt.ylabel('RMSE',fontproperties='SimHei')
plt.legend()

O1 = scn.SCN.GetOutput(X)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(T,color='b',linestyle='--',label='Training Target')
ax3.plot(O1,color='r',linestyle='-.',label='Model Output')
ax3.set_xlim()
ax3.set_ylim()
plt.grid(True)

plt.legend()

O2 = scn.SCN.GetOutput(X2)

fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(T2,color='b',linestyle='--',label='Test Target')
ax4.plot(O2,color='r',linestyle='-.',label='Model Output')
ax4.set_xlim()
ax4.set_ylim()
plt.grid(True)

plt.legend()






