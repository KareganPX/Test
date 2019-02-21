# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:30:30 2019

@author: admin
"""

#導入函數

import numpy as np
import tkinter as tk
import tkinter.filedialog

#----------------------------------------------------------
#設定輸出函數

def nonlin(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
        
    else:
        return 1/(1+np.exp(-x))
#----------------------------------------------------------
#選擇檔案

window = tk.Tk()
window.title('選擇檔案')
window.geometry('200x200')
path = 1
path1 = 1

def C1():
    file_path = tkinter.filedialog.askopenfilename()
    
    if file_path != '':
        lb.config(text = "輸入資料"+file_path);
        global path 
        path = file_path
    else:
        lb.config(text = "沒有選擇文件");

lb = tk.Label(window,text = '沒有選擇文件')
lb.pack()
btn = tk.Button(window,text="選擇輸入data",command=C1)
btn.pack()

def C2():
    file_path1 = tkinter.filedialog.askopenfilename()
    if file_path1 != '':
        lb1.config(text = "目標資料"+file_path1);
        global path1 
        path1 = file_path1
    else:
        lb1.config(text = "沒有選擇文件");

lb1 = tk.Label(window,text = '沒有選擇文件')
lb1.pack()
btn = tk.Button(window,text="選擇目標data",command=C2)
btn.pack()

close = tk.Button(window, text="確認",width=15, height=2, command=window.destroy)
close.pack()

window.mainloop()
   
#----------------------------------------------------------
#讀入X
        
x_data = open(path,'r')

lines = x_data.readlines()
counter = 0

for line in lines:
    
    counter = counter+1

x_data = np.random.randn(counter,3)

i=0

for line in lines:
    
    for j in range (0,3) :
        
        x_data[i,j] = line.split(' ',2)[j]
        
    i=i+1

#print(x_data)

#----------------------------------------------------------    
#前處理 

def pre(x, deriv = False):
    if(deriv == True):
        return (x*(test_x_max-test_x_min))+test_x_min
        
    else:
        return (x-test_x_min)/(test_x_max-test_x_min)

for fp1 in range(0,3):
    
    test_input = x_data[:,fp1].T
    test_input_fix = sorted(test_input)
    test_x_max = test_input_fix[counter-1]
    test_x_min = test_input_fix[0]
    
    

    for fp2 in range(0,counter):
        x_data[fp2,fp1] = pre(x_data[fp2,fp1])

#print (x_data)
 
#----------------------------------------------------------    
#讀入Y

y_data = open(path1,'r')

lines = y_data.readlines()
counter = 0

for line in lines:
    
    counter = counter+1

y_data = np.random.randn(counter,1)

i=0

for line in lines:
    
    y_data[i] = line
        
    i=i+1

#print(y_data)

#----------------------------------------------------------    
#前處理 
    
def pre1(x, deriv = False):
    if(deriv == True):
        return (x*(test_y_max-test_y_min))+test_y_min
        
    else:
        return (x-test_y_min)/(test_y_max-test_y_min)    

for fp1 in range(0,1):
    
    test_output = y_data[:,fp1].T
    test_output_fix = sorted(test_output)
    test_y_max = test_output_fix[counter-1]
    test_y_min = test_output_fix[0]
    
    

    for fp2 in range(0,counter):
       y_data[fp2,fp1] = pre1(y_data[fp2,fp1])

#print (x_data)
    
#----------------------------------------------------------
#隱藏層變數
    
first = np.random.normal(loc=3,scale=1e-1,size=[5, 1]) #same with 隱藏層數目
second = np.random.normal(loc=3,scale=1e-1,size=[1, 1])

H1 = np.random.random((3, 5)) 
H2 = np.random.random((5, 1))
#H2 = np.random.random((4, 1))

train_step = 0.5

#----------------------------------------------------------
#矩陣

B = np.random.random((1, 1))    #same with output
BB =  np.random.random((5, 3))  #same with H1.T
BBB = np.random.random((5, 1))  #same with H2
l2 = np.random.random((5, 1))   #same with H2

#----------------------------------------------------------
#正向傳遞計算

for j in range(30):#疊代
    if(j%1) == 0:
        
        for f1 in range(0,counter):#讀入資料
            if(f1%1) == 0:
                
                for f5 in range (0,3) :#隱藏層網路
                            if (f5%1) == 0:
                                
                                A = x_data[f1,f5]
                                syn0 = H1[f5]
                    
                                l0 = A
                                l1 = np.dot(l0,syn0)
                                BB[:,f5] = l1
   
                for f6 in range (0,5) :#隱藏層輸出加總 same with 隱藏層數目
                            if (f6%1) == 0:
                                
                                CC = BB[f6]
                                #BBB[f6] = nonlin(np.sum(CC) - first[f6]) 
                                BBB[f6] = nonlin(np.sum(CC)) 
                       
                for f7 in range (0,5) :#輸出層輸出加總  same with 隱藏層數目
                            if (f7%1) == 0:
                               
                                l2[f7] = np.dot(BBB[f7],H2[f7])
                         
                #B = nonlin(np.sum(l2) - second)
                B = nonlin(np.sum(l2))
                #Final_test = pre1(np.sum([B]),deriv = True)
                #B = np.sum(l2) - second
                        
#----------------------------------------------------------                 
#計算誤差

                ans = 0.5*((y_data[f1] - np.sum([B]))**2)
                l2_error = y_data[f1] - np.sum([B])
                #ans = 0.5*((y_data[f1] - Final_test)**2)
                #l2_error = y_data[f1] - Final_test

#----------------------------------------------------------
#誤差修正，倒傳遞
   
                if ans < 0.0005 :
                    if f1 == 0:
                        print ("誤差內:"+str(np.mean(ans)))
                        print ("---------------------------------------------------------------")
                                    
                    
                    continue
            
            
                else :
                    if f1 == 0:
                        print ("Error:"+str(np.mean(ans)))
                        print ("---------------------------------------------------------------")
                    
                    #print ("Error:"+str(np.mean(ans)))
                    #print ("---------------------------------------------------------------")
                    
                    for f4 in range (0,5) :  #same with 隱藏層數目
                        
                        D = BBB[f4]
                        #l2_delta = train_step*l2_error*nonlin(np.sum([B]),deriv = True)*D*-1
                        l2_delta = train_step*l2_error*nonlin(np.sum([B]),deriv = True)*D
                        H2[f4] = H2[f4] + l2_delta
              
                    l2_delta_f = train_step*l2_error*nonlin(np.sum([B]),deriv = True)*-1
                    second = second + l2_delta_f
                    
                    l1_error1 = l2_error*nonlin(np.sum([B]),deriv = True)#dj
                    
                    for f3 in range (0,5) :  #same with 隱藏層數目
                        C = x_data[f1]
                        l1_error = l1_error1*H2[f3]
                        #l1_delta = train_step*l1_error*nonlin(BBB[f3],deriv = True)*C*-1
                        l1_delta = train_step*l1_error*nonlin(BBB[f3],deriv = True)*C
                        H1[:,f3] = H1[:,f3] + l1_delta
                        
                        l1_delta_f = train_step*l1_error*nonlin(BBB[f3],deriv = True)*-1
                        first[f3] = first[f3] + l1_delta_f

        
        #if ans < 0.0005 :
            #if j > 10 :
                    
                    #break
                
#----------------------------------------------------------
#印出隱藏層，偏權值參數
                
print ("訓練結束:")
print ("H1:",H1)
print ("first:",first)
print ("H2:",H2)
print ("second:",second)

#----------------------------------------------------------
#選擇測試檔案

window = tk.Tk()
window.title('選擇測試檔案')
window.geometry('200x200')
input_path = 1
input_path1 = 1

def C1():
    file_path = tkinter.filedialog.askopenfilename()
    
    if file_path != '':
        lb.config(text = "輸入資料"+file_path);
        global input_path 
        input_path = file_path
    else:
        lb.config(text = "沒有選擇文件");

lb = tk.Label(window,text = '沒有選擇文件')
lb.pack()
btn = tk.Button(window,text="選擇輸入data",command=C1)
btn.pack()

def C2():
    file_path1 = tkinter.filedialog.askopenfilename()
    if file_path1 != '':
        lb1.config(text = "目標資料"+file_path1);
        global input_path1 
        input_path1 = file_path1
    else:
        lb1.config(text = "沒有選擇文件");

lb1 = tk.Label(window,text = '沒有選擇文件')
lb1.pack()
btn = tk.Button(window,text="選擇目標data",command=C2)
btn.pack()

close = tk.Button(window, text="確認",width=15, height=2, command=window.destroy)
close.pack()

window.mainloop()

#----------------------------------------------------------
#讀入輸入
        
input_data = open(input_path,'r')

lines = input_data.readlines()
counter1 = 0

for line in lines:
    
    counter1 = counter1+1

input_data = np.random.randn(counter1,3)

i=0

for line in lines:
    
    for j in range (0,3) :
        
        input_data[i,j] = line.split(' ',2)[j]
        
    i=i+1

#print(input_data)

#----------------------------------------------------------    
#輸入前處理 

input_data1 = np.random.randn(counter1,3)
 
def pre2(x, deriv = False):
    if(deriv == True):
        return (x*(x_max-x_min))+x_min
        
    else:
        return (x-x_min)/(x_max-x_min)        
    
for fp1 in range(0,3):
    
    input_x = input_data[:,fp1].T
    input_x_fix = sorted(input_x)
    x_max = input_x_fix[counter1-1]
    x_min = input_x_fix[0]
    
    

    for fp2 in range(0,counter1):
        input_data1[fp2,fp1] = pre2(input_data[fp2,fp1])

#print(input_data)    
    
#----------------------------------------------------------    
#讀入輸出

output_data = open(input_path1,'r')

lines = output_data.readlines()
counter1 = 0

for line in lines:
    
    counter1 = counter1+1

output_data = np.random.randn(counter1,1)

i=0

for line in lines:
    
    output_data[i] = line
        
    i=i+1

#print(output_data)


#----------------------------------------------------------    
#輸出前處理 

def pre3(x, deriv = False):
    if(deriv == True):
        return (x*(y_max-y_min))+y_min
        
    else:
        return (x-y_min)/(y_max-y_min)     

for fp1 in range(0,1):
    
    output_y = output_data[:,fp1].T
    output_y_fix = sorted(output_y)
    y_max = output_y_fix[counter1-1]
    y_min = output_y_fix[0]
    
    

    for fp2 in range(0,counter1):
        output_data[fp2,fp1] = pre3(output_data[fp2,fp1])

#print (x_data)
    
   
#----------------------------------------------------------
#執行程式(fixing)
  
print ("正式執行")   
for f1 in range(0,counter1):#讀入資料
            if(f1%1) == 0:
                
                print ("第",(f1+1),"筆資料")
                print ("input:",input_data[f1])
                             
                for f5 in range (0,3) :#隱藏層網路
                            if (f5%1) == 0:
                                                            
                                A = input_data1[f1,f5]
                                syn0 = H1[f5]
                    
                                l0 = A
                                l1 = np.dot(l0,syn0)
                                BB[:,f5] = l1
   
                for f6 in range (0,5) :#隱藏層輸出加總 same with 隱藏層數目
                            if (f6%1) == 0:
                                
                                CC = BB[f6]
                                #BBB[f6] = nonlin(np.sum(CC) - first[f6]) 
                                BBB[f6] = nonlin(np.sum(CC)) 
                       
                for f7 in range (0,5) :#輸出層輸出加總 same with 隱藏層數目
                            if (f7%1) == 0:
                               
                                l2[f7] = np.dot(BBB[f7],H2[f7])
                         
                #B = nonlin(np.sum(l2) - second)
                B = nonlin(np.sum(l2))
                Final = pre3(np.sum([B]),deriv = True) 
                ans = 0.5*((output_data[f1] - np.sum([B]))**2)
                #print ("output:",B)
                print ("output:",Final)
                print ("誤差:",ans,"\n")
