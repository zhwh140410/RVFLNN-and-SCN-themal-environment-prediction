# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:44:02 2018

@author: acer
"""
from __future__ import division 
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt 
def transfer(X):
    sh= np.shape(X)
    m = sh[0]
    A = np.zeros((m))
    for i in range(m):
         A[i] = np.argmax(X[i,:])
    return A   
    
def RMSE(E0):
    EN1 = np.shape(E0)
    EN = EN1[0]
    Error = np.sqrt(np.sum(np.sum(np.square(E0),0)/EN))
    return Error
def OneHotMatrix(X):
    N = np.shape(X)
    n = N[0]
    
    p = N[1]
    
    Y = np.zeros((n, p))
    if p > 1:
        for i in range(n):
            ind = np.argmax(X[i, :])#######################
            Y[i, ind] = 1
    else:
        for i in range(n):
            if X[i] > 0.50:
                Y[i] = 1
    return Y
def sigm(z):
    return 1.0/(1.0+np.exp(-z))
class SCN(object):
    
    def __init__(self,L_max=100, T_max=350, tol=0.0001,Lambdas=[0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200], r=[0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999] , nB=1):
        self.L = 1
        self.verbose = 50
        self.cost = 0
        self.L_max=100
        self.tol=0.0001
        self.Lambdas=[0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200]
        self.r=[0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999]
        self.nB=1
        self.T_max=350
        global W
        W = []
        
        
        global b
        b = []
    def InequalityEq( eq, gk, r_L):
        ksi = np.square((np.dot(eq.T,gk)))/(np.dot(gk.T,gk)) - ((1-r_L) * np.dot(eq.T,eq))
        return ksi
    
    def SC_Search(self, X, E0):
        Flag = 0
        d = np.size(X,1)
        m = np.size(E0,1)
        WB = []
        bB = []
        
        # Linear search for better nodes
        C = []
        for i_Lambdas in range(len(self.Lambdas)):
            n = np.size(X,0)
            Lambda = self.Lambdas[i_Lambdas]
            WT = Lambda * (2 * np.random.rand(d, self.T_max))
            bT = Lambda * (2 * np.random.rand(1, self.T_max))
            HT = sigm(np.dot(X,WT)+np.tile(bT,(n,1)))
            for i_r in range(len(self.r)):
        
                r_L = self.r[i_r]   # get the regularization value
                for t in range(self.T_max):
                    H_t = HT[:, t]
                    ksi_m = np.zeros((m))
                    for i_m in range(m):
                        eq = E0[:, i_m]
                        gk = H_t
                       
                        ksi_m[i_m] = SCN.InequalityEq(eq,gk,r_L)
                        #############################
                    Ksi_t = np.sum(ksi_m)
                    if np.min(ksi_m) > 0:
                        C.append(Ksi_t)
                        WB.append(WT[:, t])
                        bB.append(bT[:, t])
                        z = np.shape(WB)
                        print('z',z)
                        zz = np.shape(C)
                        print('C',zz)
                nC = len(C)
                print('C1',nC)
                if nC >= self.nB:
                    break
                else:
                    continue
            if nC >= self.nB:
                break
            else:
                continue
        if nC >= self.nB:
            I = sorted(C,reverse=True)
            print('I',I)
            ##################################
            I_nb = I[0]
            Inb = I_nb.astype(int)
            print('Inb',Inb)
            WB = WB[Inb]
            bB = bB[Inb]
            bB = bB[0]
            print('WB',WB)
            print('bB',bB)
            print('C2',nC)
            
            
        if nC == 0 or nC < self.nB:
            print('End Searching...')
            Flag = 1
        return WB,bB,Flag
    # Ìí¼Ó½Úµã
    def AddNodes(self,WB, bB):
        
        W.append(WB)
        b.append(bB)
        global W1
        W1 = np.array(W)
        
        self.L = len(b)
        print('L',self.L)
        return W1,b,self.L
        
    def UpgradeSCN(X,T):
        H = SCN.GetH(X)
        Beta = SCN.ComputeBeta(H,T)
        O = np.dot(H,Beta)
        E = T - O
        Error = RMSE(E)
        
        return E,Error
        
    def ComputeBeta(H,T):
        global Beta
        Beta = np.dot(np.linalg.pinv(H),T)
        
        return Beta
        
    def Regression(self,X,T):
        per_Error = []
        
        E = T
        Error = RMSE(E)
        
        while self.L < self.L_max and Error > self.tol:
            if np.mod(self.L,self.verbose) == 0:
                print("L:%d RMSE:%.6f "%self.L %Error)
            w_L, b_L,Flag = SCN.SC_Search(self,X,E)
            if Flag == 1:
                break
            W,b,self.L = SCN.AddNodes(self,w_L,b_L)
            E,Error = SCN.UpgradeSCN(X,T)
                
                
            per_Error.append(Error)
        print("L:%d RMSE:%.6f "%(self.L,Error ))
        print(np.tile('*',[1,30]))
        return per_Error
    def Classification(self,X,T):
        per_Error = []
        per_Rate = []
        E = T
        Error = RMSE(E)
        Rate = 0
        while self.L < self.L_max and Error > self.tol:
            if np.mod(self.L,self.verbose) == 0:
                print("L=%d RMSE=%.6f Rate=%.2f" %(self.L,Error,Rate))
            WB, bB,Flag = SCN.SC_Search(self,X,E)
            
            if Flag == 1:
                break
            W,b,self.L = SCN.AddNodes(self,WB,bB)
            E,Error = SCN.UpgradeSCN(X,T)
            O = SCN.GetLabel(X)
            
            
            T_T = transfer(T)
            O_T = transfer(O)
            
            
            
            cm = metrics.confusion_matrix(T_T,O_T)
            
             
            
            
            a = np.trace(cm)
            b = np.sum(cm)
            Rate = a/b
            per_Error.append(Error)
            per_Rate.append(Rate)
            
        print("L=%d RMSE=%.6f Rate=%.2f" %(self.L,Error,Rate))
        print(np.tile('*',[1,30]))
        return per_Error,per_Rate
    def GetH(X):
        H = SCN.ActivationFun(X)
        #print('H:',H)
        return H
    def ActivationFun(X):
        n = np.size(X,0)
        a = np.dot(X,W1.T)
        H = sigm(a+np.tile(b,(n,1)))
        
        return H
    
    def GetOutput(X):
        H = SCN.GetH(X)
        O = np.dot(H,Beta)
        
        return O
    def GetLabel(X):
        O1 = SCN.GetOutput(X)
        O = OneHotMatrix(O1)
        return O
    
    def GetAccuracy(X,T):
        O = SCN.GetLabel(X)
        O = transfer(O)
        T = transfer(T)
        cm = metrics.confusion_matrix(O,T)
        a = np.trace(cm)
        b = np.sum(cm)
        Rate = a/b
        return Rate
    
    def GetResult(X,T):
        H = SCN.GetH(X)
        O = np.dot(H,Beta)
        E = T - O
        Error = RMSE(E)
        return H,O,E,Error
        
if __name__ == '__main__':
    demo = SCN()
          
        
        
        
        
        
        
        
        
        
        
        
        
        
        