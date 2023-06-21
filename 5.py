import numpy as np

import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_digits

def Softmax(y):
    e = np.exp(y )
    a=[]
    for i in range(y.shape[0]):
        e[i] /= np.sum(e[i])
        a.append(np.sum(e[i]))
    return e

def accuracy(y_pred,y):
    tp=0
    tn=0

    for i in range(len(y)):
        if y_pred[i]==y[i]:
            tp+=1
        else:
            tn+=1
    return tp/(tp+tn)

def one_hot(y,c):
    ey=np.zeros((y.shape[0],c))
    for i in range(y.shape[0]):
        num=y[i]
        ey[i][num]=1
    return ey


def Loss(y_pred,y):
    E=0
    E=np.sum(np.log(y_pred[np.arange(len(y)), y]))
    return -E/len(y)


def Grad(x,t,y):
    E=0
    E=x.T@(y-t)
    return E

def Predict(x,w,b):
    y=x@w+b
    return y


def Preds(x,w,b):
    y=x@w+b
    y=Softmax(y)
    idx=np.zeros(y.shape[0],dtype=np.int32)
    for i in range(y.shape[0]):
        ind,=np.where(y[i]==max(y[i]))
        idx[i]=ind
    return idx




def Learning(x,t,prev_y,w,b,lr=0.001,steps=100):
    loss=[]
    acc=[]
    y=Predict(x,w,b)
    y=Softmax(y)
    ws = 0
    for i in range(w.shape[0]):
        ws += w[i]@w[i]
    ls=Loss(y,prev_y) + 0.00001*ws
    print(Loss(y,prev_y), ls, "losses")
    pred_ns=Preds(x,w,b)
    cur_acc=accuracy(pred_ns,prev_y)
    gr=Grad(x,t,y)
    y=Predict(x,w,b)
    y=Softmax(y)
    bgr=np.sum(y-t)
    for ep in range(steps):
        gr=Grad(x,t,y)
        y=Predict(x,w,b)
        y=Softmax(y)
        bgr=np.sum(y-t)
        gr=gr + 2*0.0001*w
        bgr=bgr + 2*0.0001*b
        w=w-lr*gr
        b=b-lr*bgr
        ws = 0
        for i in range(w.shape[0]):
            ws += w[i]@w[i]
        ls=Loss(y,prev_y) + 0.00001*ws
        pred_ns=Preds(x,w,b)
        cur_acc=accuracy(pred_ns,prev_y)

        if(ep%10==0):
            print(ep,". Loss:",ls,"| Accuracy:",cur_acc)
        if(np.sum(abs(gr))<0.1):
            print("Grad stop epoch -",ep)
            return w,b,loss,acc
        if(cur_acc>=0.99):
            print("Accuracy stop epoch -",ep)
            return w,b,loss,acc
#        if ep != 0 and abs(loss[-1]-ls)<=0.00002 :
#            print("Loss stop epoch -",ep)
#            return w,b,loss,acc
        if(ep == 99):
            print("End: ", "Loss:",ls,"| Accuracy:",cur_acc)
        acc.append(cur_acc)
        loss.append(ls)
    return w,b,loss,acc



def Conf_M(pred_y,y,K):
    m=np.zeros((K,K),dtype=np.int8)
    for i in range(len(y)):
        m[y[i]][pred_y[i]]+=1
    return m


K=10
data=load_digits()
x=data.data
y=data.target
"""
z = data.images
plt.imshow(z[1])
plt.figure()
"""

means = np.mean(x, axis=0)
stds=np.std(x, axis=0)
for i in range(len(means)):
    if(stds[i]!=0):
        x[:,i]=(x[:,i]-means[i])/stds[i]


ey=one_hot(y,K)

xt=yt=xv=yv=eyt=eyv=[]



indexesr=np.random.permutation(x.shape[0])
tstop=int(x.shape[0]*0.8)
xt=np.zeros((tstop,x.shape[1]))
yt=np.zeros(tstop,dtype=np.int32)
eyt=np.zeros((tstop,K))

for i in range(tstop):

    xt[i]=x[indexesr[i],:]
    yt[i]=y[indexesr[i]]
    eyt[i]=ey[indexesr[i]]

vl_diff=x.shape[0]-tstop
xv=np.zeros((vl_diff,x.shape[1]))
yv=np.zeros(vl_diff,dtype=np.int32)
eyv=np.zeros((vl_diff,K))
for i in range(tstop,x.shape[0]):
    xv[i-tstop]=x[indexesr[i],:]
    yv[i-tstop]=y[indexesr[i]]
    eyv[i-tstop]=ey[indexesr[i]]


w=np.random.random((x.shape[1], K))
b=np.random.random(K)

print("Validation")
yvc=Predict(xv,w,b)
yvc=Softmax(yvc)
ls=Loss(yvc,yv)
pred_ns=Preds(xv,w,b)
cur_acc=accuracy(pred_ns,yv)
print("Loss:",ls,"| Accuracy:",cur_acc)
w,b,l,acc=Learning(xt,eyt,yt,w,b)
cm=Conf_M(pred_ns,yv,10)
print(cm)
ls=Loss(yvc,yv)
pred_ns=Preds(xv,w,b)
cur_acc=accuracy(pred_ns,yv)
cm=Conf_M(pred_ns,yv,10)
print(cm)

plt.plot(l)
plt.show()

plt.plot(acc)
plt.show()