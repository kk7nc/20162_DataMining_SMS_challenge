# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:27:49 2017

@author: T420
"""
f = open('train.txt', 'r')
#f.read()
LTrain=50
nTimes=100
t= f.read().splitlines()
f.close()
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
        inputX.append(v)

    trainX=inputX[:LTrain]
    trainY=inputY[:LTrain]
    testX=inputX[LTrain:]
    testY=inputY[LTrain:]

    from sklearn.svm import SVC
    clf = SVC(C=30, kernel='rbf',decision_function_shape='ovr', gamma=0.048)
    #clf = SVC(C=4.7, kernel='sigmoid',decision_function_shape='ovr', gamma=0.048,coef0=0.525)
    #clf = SVC(C=5, kernel='poly',decision_function_shape='ovr', gamma=0.048,coef0=200,degree=2)
    clf.fit(trainX, trainY)
    res=clf.predict(testX)
    sol=(res==testY)
    return sol.sum()
acc=0;
for i in range(0,nTimes):
    import random
    random.shuffle(t)
    acc=acc+aRunProcess()*1.0/(nTrain-LTrain)
print "accuracy of "+str(nTimes)+" is : "+ str(acc/(nTimes))
