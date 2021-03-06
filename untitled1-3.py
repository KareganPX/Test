# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:39:29 2018

@author: admin
"""
#import tensorflow as tf
import numpy as np

def nonlin(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
        
    else:
        return 1/(1+np.exp(-x))
    
x_data = np.random.randn(4,3)
#x_data = np.random.normal(loc=1,scale=1e-1,size=[4, 3])
#y_data = np.random.randn(4,1)
y_data = np.random.normal(loc=4,scale=1e-1,size=[4, 1])

#first = np.random.randn(4,1)
first = np.random.normal(loc=1,scale=1e-1,size=[4, 1])
#second = np.random.randn(1,1)
second = np.random.normal(loc=1,scale=1e-1,size=[1, 1])

print (x_data)
print (y_data)

H1 = np.random.random((3, 4)) 
#H1 = np.random.normal(loc=1,scale=1e-3,size=[3, 4])
H2 = np.random.random((4, 1))
#H2 = np.random.normal(loc=1,scale=1e-3,size=[4, 1])
train_step = 0.5 

B = np.random.random((1, 1))
BB =  np.random.random((4, 3))
BBB = np.random.random((4, 1))

#C = np.random.random((1, 3))
#CC = np.random.random((4, 1))
l2 = np.random.random((4, 1))
#B = np.placeholder(np.float32, [1, None])

#print (first)
#print (second)
#print (H1)
#print (H2)


for j in range(20):#疊代
    if(j%2) == 0:
        
        for f1 in range(0,4):#一筆資料
            if(f1%1) == 0:
                
                for f2 in range (0,1) :#第二層網路
                    if (f2%1) == 0:
                        
                        for f5 in range (0,3) :#第一層網路
                            if (f5%1) == 0:
                                A = x_data[f1,f5]
                                syn0 = H1[f5]
                    
                                l0 = A
                                l1 = np.dot(l0,syn0)
                                
                                BB[:,f5] = l1
                                #print (BB)
                                
                       
                        for f6 in range (0,4) :#第一層網路輸出加總
                            if (f6%1) == 0:
                                #print (np.sum(BB[:,f6]))
                                CC = BB[f6]
                                BBB[f6] = nonlin(np.sum(CC) - first[f6])
                                
                        #print (l1)
                                
                        #print (B)
                        for f7 in range (0,4) :#第一層網路輸出加總
                            if (f7%1) == 0:
                                #E = BBB
                                l2[f7] = np.dot(BBB[f7],H2[f7])
                         
                        B = nonlin(np.sum(l2) - second)
                        
                    
                #計算誤差
                print (f1)
                #print (np.sum([B]))
                ans = 0.5*((y_data[f1] - np.sum([B]))**2)
                l2_error = y_data[f1] - np.sum([B])
                #print (l1_error)
            
                #ans = 0.5*(l2_error**2)
                #print (ans)
        
                if ans < 0.25 :
                    
                    print ("down:"+str(np.mean(l2_error)))
                    print ("---------------------------------------------------------------")
                    
                    continue
            
            
                else :
                    
                    print ("Error:"+str(np.mean(l2_error)))
                    print ("---------------------------------------------------------------")
                    #l2_delta1 = train_step*l2_error*nonlin(B,deriv = True)
                    for f4 in range (0,4) :
                        D = BBB[f4]
                        #print (B)
                        l2_delta = train_step*l2_error*nonlin(np.sum([B]),deriv = True)*D
                        #print (l2_delta)
                        H2[f4] += H2[f4] + l2_delta
                        #print (H2)
                        #C[f4] = l2_delta #有點問題1
                        
                    #l2_delta = np.dot((train_step*l2_error*nonlin(B,deriv = True)),l1)
                    
                    #l2_delta_f = train_step*l2_error*nonlin(np.sum([B]),deriv = True)*-1
                    #print (l2_delta_f)
                    l2_delta_f = train_step*l2_error*nonlin(np.sum([B]),deriv = True)*-1
                    second += second + l2_delta_f
                    
                    l1_error1 = l2_error*nonlin(np.sum([B]),deriv = True)#dj
                    #l1_delta = np.dot((train_step*l1_error*nonlin(H1,deriv = True)),x_data[f1])
                    
               #剩下修改部分     
                    #l1_delta1 = train_step*l1_error*nonlin(H1,deriv = True)
                    #print (x_data[f1])
                    for f3 in range (0,4) :
                        C = x_data[f1]
                        #E = H1[f3]
                        #F = H2[f3] #問題1
                        #print (C)
                        l1_error = l1_error1*H2[f3]
                        l1_delta = train_step*l1_error*nonlin(BBB[f3],deriv = True)*C
                        H1[:,f3] += H1[:,f3] + l1_delta
                        
                        l1_delta_f = train_step*l1_error*nonlin(BBB[f3],deriv = True)*-1
                        first[f3] += first[f3] + l1_delta_f
                        
                    #3*4篇權值修正
                    #E = BBB.T
                    #l1_delta_f = train_step*l1_error1*H2*nonlin(BBB,deriv = True)*-1
                    #first += first + l1_delta_f
                    
            
                    
                    
                    #print (H1)
                    
                
        #print ("out") 
        if ans < 0.25 :
                    
                    #print ("down:"+str(np.mean(l2_error)))
                    #print ("---------------------------------------------------------------")
                    
                    break
        
                    
#l2_error = y_data - l2
    
    
print ("outout after Training:")
print (H1)
print (first)
print (H2)
print (second)