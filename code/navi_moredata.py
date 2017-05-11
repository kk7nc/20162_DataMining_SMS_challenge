# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:23:14 2017

@author: T420
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:27:49 2017
@author: T420
"""
f = open('train.txt', 'r')
#f.read()
LTrain=100
nTimes=100
t= f.read().splitlines()
f.close()
f = open('train_Manh_sub.txt', 'r')
#f.read()
LTrain=50
nTimes=100
t.extend(f.read().splitlines());
f.close()
g=open('spam-harm-dict.txt','r')
e=g.read();
print t.__len__()
dictHarmSpam=e.split(' ');
g.close()
nTrain=t.__len__()
def aRunProcess():
    inputY=[]
    dict=[]
    inputWord=[];
    for x in t:
        y=x.split(" ")
        inputWord.append(y[1:])
        inputY.append(y[0])
        dict.extend(y[1:])
    dict=set(dict)
    inputX=[]

    for x in inputWord:
        v=[]
        for xx in dict:
            if xx in x:
                v.append(1)
            else:
                v.append(0)
        for xx in dictHarmSpam:
            if xx in x:
                v.append(1)
            else:
                v.append(0)
        inputX.append(v)
    trainX=inputX[:LTrain]
    trainY=inputY[:LTrain]
    testX=inputX[LTrain:]
    testY=inputY[LTrain:]

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(trainX, trainY)
    res=gnb.predict(testX)
    sol=(res==testY)
    return sol.sum()
acc=0;
for i in range(0,nTimes):
    import random
    random.shuffle(t)
    acc=acc+aRunProcess()*1.0/(nTrain-LTrain)
print "accuracy of "+str(nTimes)+" is : "+ str(acc/(nTimes))