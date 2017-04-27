# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:18:02 2017

@author: T420
"""

f = open('train_Manh.txt', 'r')
g = open('train_Manh_sub.txt','a')
#f.read()
LTrain=50
nTimes=100
t= f.read().splitlines()
c=0;
import random
for x in t:
    ra=random.random()
    if x[0]=='1':
        if ra>0.5 and c<74:    
            g.write(x+'\n');
            c+=1
    if x[0]=='-': 
            g.write(x+'\n');
    
g.close()
f.close()