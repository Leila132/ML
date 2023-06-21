from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import hstack
import numpy as np
import math
from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

x, t = fetch_california_housing(return_X_y=True)
df = pd.DataFrame(data={str(i): f for i, f in enumerate(np.transpose(x))})
df['t'] = t
df = shuffle(df)
len(df)

train = df[:16000]
test = df[16000:]

scaler = MinMaxScaler()

train_x = train.iloc[:, :8]
train_y = np.array(train.iloc[:, 8]).reshape(-1, 1)

scaler.fit(train_x)
train_x = scaler.transform(train_x)

scaler.fit(train_y)
train_y = scaler.transform(train_y)


test_x = test.iloc[:, :8]
test_y = np.array(test.iloc[:, 8]).reshape(-1, 1)

scaler.fit(test_x)
test_x = scaler.transform(test_x)

scaler.fit(test_y)
test_y = scaler.transform(test_y)

print("Train:\n")
print(train_x.shape, train_y.shape)

print("Test:\n")
print(test_x.shape, test_y.shape)

lr = 0.01
w_old = np.random.randn(8) * 0.1
n_epochs = 100

errors1 = []
errors2 = []
for epoch in range(n_epochs):
  
    # рассчитываем результирующий массив с текущими коэффициентами w
    # на основе обучающей выборки 
    yhat = np.array([x.dot(w_old) for x in train_x]).reshape(train_x.shape[0], 1)


    # 1. определяем лосс
    # считаем отклонение нового результата от обучающего:
    error = (train_y - yhat)

    #check_error = (error ** 2).mean()
    w = np.zeros(8)
    # 2. считаем градиенты (вспоминая формулу производной) и обновляем параметры
    # для коэффициентов w
    for i, e in enumerate(w):
        x_var = train_x[:, i].reshape(train_x.shape[0], 1)
        grad_i = -2 * (x_var * error).mean()

        w[i] = w_old[i] - lr * grad_i
    
    if LA.norm(w) < 0.001:
        break

    if LA.norm(w - w_old) < 0.001:
        break
    
    w_old = w
    #считаем ошибку на тренировочной выборке
    N = len(train_y)
    y_pred = [x.dot(w) for x in train_x]
    error = 0
    for y, pred in zip(train_y, y_pred):
        error = error + (y - pred) ** 2
    
    errors1.append(error/N)

    #считаем ошибку на тестовой выборке
    N = len(train_y)
    y_pred = [x.dot(w) for x in test_x]
    error = 0
    for y, pred in zip(test_y, y_pred):
        error = error + (y - pred) ** 2
    errors2.append(error/N)

    print(epoch + 1)



print(w)   

N = len(train_y)
y_pred = [x.dot(w) for x in train_x]

error = 0
for y, pred in zip(train_y, y_pred):
  error = error + (y - pred) ** 2

print("Ошибка в тренировочной части: ", error / N)

y_pred = [x.dot(w) for x in test_x]
error = 0
for y, pred in zip(test_y, y_pred):
  error = error + (y - pred) ** 2

print("Ошибка в тестовой части: ", error / N)

row = np.linspace(1, len(errors1), len(errors1))

# график изменения ошибки в тренировочной выборке
plt.figure()
plt.plot(row, errors1, c="black", label = "тренировочная выборка")
plt.xlabel("iterations")
plt.ylabel("error")
plt.title('Зависимость ошибки от количества иттераций')

# график изменения ошибки в тестовой выборке
plt.plot(row, errors2, c="red", label = "тестовая выборка")
plt.legend(loc='right')
plt.show()

"""
x = df.iloc[:, 7:8]
y = df.iloc[:, 8:]
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

scaler.fit(y)
y = scaler.transform(y)
plt.figure()
plt.scatter(x, y, c="black", label = "х0")
plt.xlabel("x7")
plt.ylabel("y")
plt.show()
"""