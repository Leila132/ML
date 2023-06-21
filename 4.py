import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def sigmoid(x):
    return 1. / (1 + np.exp(-1 * x))

#футболисты
X1 = np.random.randn(1000) * 2.5 + 190
X = []
for i in range(1000):
    X.append([X1[i], 0])
#баскетболисты
X2 = np.random.randn(1000) * 2.6 + 170
for i in range(1000, 2000):
    X.append([X2[i-1000], 1])

random.shuffle(X)

#делим выборку
x_train = []
y_train = []
for i in range(1000):
    x_train.append(X[i][0])
    y_train.append(X[i][1])
x_test = []
y_test = []
for i in range(1000):
    x_test.append(X[i][0])
    y_test.append(X[i][1])

x_train = np.array(x_train)
x_test = np.array(x_test)

scaler = StandardScaler(with_mean=True, with_std=True)

x_train = scaler.fit_transform(x_train.reshape(-1, 1))
scaler.fit(x_test.reshape(-1, 1))
x_test = scaler.transform(x_test.reshape(-1, 1))

lr = 0.01
w = random.randint(0, 10) / 10.
b = random.randint(0, 10) / 10.
n_epochs = 300
tr = 0.5

print(w, b)

for epoch in range(n_epochs):
    grad_w = 0
    grad_b = 0
    for i in range(len(x_train)):
        sigma = sigmoid(w * x_train[i] + b) #1 / (1 + pow(math.e, -1 * w * x_train[i]))
        grad_w += (sigma - y_train[i]) * (x_train[i])
        grad_b += (sigma - y_train[i])

    y_pred = np.array([1 if sigmoid(w * x + b) > tr else 0 for x in x_test])

    w = w - lr * grad_w
    b = b - lr * grad_b
  
print(w, b)

y_pred = np.array([1 if sigmoid(w[0] * x + b) > tr else 0 for x in x_test])

print(confusion_matrix(y_test, y_pred))

N = 1000

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(1000):
    if y_test[i] == 0 and y_pred[i] == 0:
        TN += 1
    if y_test[i] == 1 and y_pred[i] == 1:
        TP += 1
    if y_test[i] == 0 and y_pred[i] == 1:
        FP += 1
    if y_test[i] == 1 and y_pred[i] == 0:
        FN += 1

accuracy = (TP + TN) / N
print("Accuracy: ", accuracy)

if TP + FP == 0:
    print("Precision: ", 0)
else:
    precision = TP / (TP + FP)
    print("Precision: ", precision)

if TP + FN == 0:
    print("Recale: ", 1)
else:
    recale = TP / (TP + FN)
    print("Recale: ", recale)

F1 = 2 * precision * recale / (precision + recale)
print("F1_score: ", F1)

if FP + TN == 0:
    print("Alpha: ", 0)
else:
    print("Alpha: ", FP / (FP + TN))

if TP + FN == 0:
    print("Beta: ", 0)
else:
    beta = FN / (TP + FN)
    print("Beta: ", beta)

#ROC

trs = np.linspace(0, 1, 100)
alphas = []
recales = []
accuracies = []
for i in range(100):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    y_pred = np.array([1 if sigmoid(w[0] * x + b) > trs[i] else 0 for x in x_test])
    for i in range(1000):
        if y_test[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_test[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_test[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_test[i] == 1 and y_pred[i] == 0:
            FN += 1

    if TP + FN == 0:
        recales.append(1)
    else:
        recales.append(TP / (TP + FN))
    
    if FP + TN == 0:
        alphas.append(0)
    else:
        alphas.append(FP / (FP + TN))
    
    accuracies.append((TP + TN) / N)

AUC = 0
#AUC
for i in range(1, 100):
    AUC += (recales[i-1] + recales[i]) * (alphas[i-1] - alphas[i]) / 2

print("AUC: ", AUC)
print("*" * 100)


best_index = accuracies.index(max(accuracies))
print("Best tr: ", trs[best_index])
print("Accuracy maximum: ", max(accuracies))
TP = 0
TN = 0
FP = 0
FN = 0
y_pred = np.array([1 if sigmoid(w[0] * x + b) > trs[best_index] else 0 for x in x_test])
for i in range(1000):
    if y_test[i] == 0 and y_pred[i] == 0:
        TN += 1
    if y_test[i] == 1 and y_pred[i] == 1:
        TP += 1
    if y_test[i] == 0 and y_pred[i] == 1:
        FP += 1
    if y_test[i] == 1 and y_pred[i] == 0:
        FN += 1

print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)

if TP + FN == 0:
    print("Recale: ", 1)
    print("Beta: ", 0)
else:
    print("Recale: ", TP / (TP + FN))
    beta = FN / (TP + FN)
    print("Beta: ", beta)

if FP + TN == 0:
    print("Alpha: ", 0)
else:
    print("Alpha: ", FP / (FP + TN))

if TP + FP == 0:
    print("Precision: ", 0)
else:
    precision = TP / (TP + FP)
    print("Precision: ", precision)

F1 = 2 * precision * recale / (precision + recale)
print("F1_score: ", F1)


plt.figure()
plt.plot(alphas, recales, c="black", label = "х0")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
