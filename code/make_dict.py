# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 18:47:47 2017

@author: T420
"""
f = open('train.txt', 'r')
#f.read()
t= f.read().splitlines()
f.close()
nTrain=t.__len__()
inputY=[]
dict=[]
inputWord=[];
for x in t:
    y=x.split(" ")
    inputWord.append(y[1:])
    inputY.append(y[0])
    dict.extend(y[1:])
dict=set(dict)
print dict
probSpam=[]
probHarm=[]
probSpam2=[]
probHarm2=[]
prob=[]
for x in dict:
    cs=0
    ch=0
    c=0
    for xx in inputWord:
        if x in xx:
            c=c+1
            if inputY[inputWord.index(xx)] == '-1' :
                cs=cs+1
            else:
                ch=ch+1
    
            
    probSpam2.append(cs/(c*1.0))
    probHarm2.append(ch/(c*1.0))
    probSpam.append(cs)
    probHarm.append(ch)
print probHarm
print probSpam
listDict=list(dict)
for x in range(0,probSpam.__len__()):
    if probSpam[x]>=4 and probSpam2[x]>0.7 :
        print listDict[x],
print '\n'
for x in range(0,probHarm.__len__()):
    if probHarm[x]>=4 and probHarm2[x]>0.7:
        print listDict[x],
print probSpam2[listDict.index('http://')]