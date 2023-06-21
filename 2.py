import numpy as np
import matplotlib.pyplot as plt
import math

def Plan1(X):
    F = np.ones([X.shape[0], 5])
    for i in range(X.shape[0]):
        F[i][1] = math.sin(X[i])
        F[i][2] = math.cos(X[i])
        F[i][3] = math.exp(X[i])
        F[i][4] = math.sqrt(X[i])
    return F

def Plan2(X):
    F = np.ones([X.shape[0], 5])
    for i in range(X.shape[0]):
        F[i][1] = math.cos(X[i])
        F[i][2] = math.exp(X[i])
        F[i][3] = math.sin(X[i])
        F[i][4] = math.sqrt(X[i])
    return F

def Plan3(X):
    F = np.ones([X.shape[0], 5])
    for i in range(X.shape[0]):
        F[i][1] = math.sqrt(X[i])
        F[i][2] = math.cos(X[i])
        F[i][3] = math.sin(X[i])
        F[i][4] = math.exp(X[i])
    return F

def Plan4(X):
    F = np.ones([X.shape[0], 5])
    for i in range(X.shape[0]):
        F[i][1] = math.exp(X[i])
        F[i][2] = math.sqrt(X[i])
        F[i][3] = math.cos(X[i])
        F[i][4] = math.sin(X[i])
    return F

def Plan5(X):
    F = np.ones([X.shape[0], 8])
    for i in range(X.shape[0]):
        F[i][1] = math.cos(X[i])
        F[i][2] = math.exp(X[i])
        F[i][3] = math.sin(X[i])
        F[i][4] = (math.sqrt(X[i]))**2
        F[i][5] = (math.cos(X[i]))**2
        F[i][6] = (math.exp(X[i]))**2
        F[i][7] = (math.sin(X[i]))**2
    return F

def Plan6(X):
    M = 8
    F = np.ones([X.shape[0], M+1])
    for i in range(X.shape[0]):
        for j in range(1, M + 1):
            F[i][j] = X[i] ** j
    return F

def Plan7(X):
    M = 100
    F = np.ones([X.shape[0], M+1])
    for i in range(X.shape[0]):
        for j in range(1, M + 1):
            F[i][j] = X[i] ** j
    return F

N = 1000
x = np.linspace(0, 1, N)
np.random.shuffle(x)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error # mean-std norm, minmax 

train = x[:800]
valid = x[801:901]
test = x[901:]

t_train = t[:800]
t_val = t[801:901]
t_test = t[901:]

laambda = [0, 1e-15, 1e-14,1e-13, 1e-12, 1e-11, 1e-10, 1e-09, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
E_1 = []
for i in range(20):
    F = Plan1(train)

    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + laambda[i]*l) @ F.T @ t_train
    F = Plan1(valid)
    Y = np.dot(w, F.T)
    print(t_val.shape)
    E = (1 / 2) * sum((t_val - Y) ** 2) + laambda[i]*np.dot(w, w)
    E_1.append(E)

E_2 = []
for i in range(20):
    F = Plan2(train)

    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + laambda[i]*l) @ F.T @ t_train
    F = Plan2(valid)
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_val - Y) ** 2) + laambda[i]*np.dot(w, w)
    E_2.append(E)

E_3 = []
for i in range(20):
    F = Plan3(train)

    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + laambda[i]*l) @ F.T @ t_train
    F = Plan3(valid)
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_val - Y) ** 2) + laambda[i]*np.dot(w, w)
    E_3.append(E)

E_4 = []
for i in range(20):
    F = Plan4(train)

    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + laambda[i]*l) @ F.T @ t_train
    F = Plan4(valid)
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_val - Y) ** 2) + laambda[i]*np.dot(w, w)
    E_4.append(E)

E_5 = []
for i in range(20):
    F = Plan5(train)
    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + laambda[i]*l) @ F.T @ t_train
    F = Plan5(valid)
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_val - Y) ** 2) + laambda[i]*np.dot(w, w)
    E_5.append(E)

E_6 = []
for i in range(20):
    F = Plan6(train)

    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + laambda[i]*l) @ F.T @ t_train
    F = Plan6(valid)
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_val - Y) ** 2) + laambda[i]*np.dot(w, w)
    print(E)
    E_6.append(E)

E_7 = []
for i in range(20):
    F = Plan7(train)

    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + laambda[i]*l) @ F.T @ t_train
    print("w", w)
    F = Plan7(valid)
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_val - Y) ** 2) + laambda[i]*np.dot(w, w)
    E_7.append(E)

E = []
E.append(min(E_1))
E.append(min(E_2))
E.append(min(E_3))
E.append(min(E_4))
E.append(min(E_5))
E.append(min(E_6))
E.append(min(E_7))
a = min(E)
funk_num = E.index(a)
best_koef = 0
best_funk = ""

if funk_num == 0:
    best_koef = laambda[E_1.index(a)]
    best_funk = "1"
    print("Лучший коэффициент регуляризации: ", best_koef)
    print("Лучшие базовые функции: ", best_funk)
    test.sort()
    t_test.sort()
    F = Plan1(test)
    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + best_koef*l) @ F.T @ t_test
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_test - Y) ** 2) + best_koef*np.dot(w, w)
    test.sort()
    z = 20 * np.sin(2 * np.pi * 3 * test) + 100 * np.exp(test)
    print("Минимальная ошибка в валидационной части:", a)
    print("Ошибка в тестовой части:", E)
    plt.plot(test, z, 'r--', label='z')
    plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
    plt.plot(test, Y, 'b--', label='Y')
    plt.legend(loc='right')
    plt.title('Регрессия с регуляризацией')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
if funk_num == 1:
    best_koef = laambda[E_2.index(a)]
    best_funk = "2"
    print("Лучший коэффициент регуляризации: ", best_koef)
    print("Лучшие базовые функции: ", best_funk)
    test.sort()
    t_test.sort()
    F = Plan2(test)
    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + best_koef*l) @ F.T @ t_test
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_test - Y) ** 2) + best_koef*np.dot(w, w)
    test.sort()
    z = 20 * np.sin(2 * np.pi * 3 * test) + 100 * np.exp(test)
    print("Минимальная ошибка в валидационной части:", a)
    print("Ошибка в тестовой части:", E)
    plt.plot(test, z, 'r--', label='z')
    plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
    plt.plot(test, Y, 'b--', label='Y')
    plt.legend(loc='right')
    plt.title('Регрессия с регуляризацией')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
if funk_num == 2:
    best_koef = laambda[E_3.index(a)]
    best_funk = "3"
    print("Лучший коэффициент регуляризации: ", best_koef)
    print("Лучшие базовые функции: ", best_funk)
    test.sort()
    t_test.sort()
    F = Plan3(test)
    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + best_koef*l) @ F.T @ t_test
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_test - Y) ** 2) + best_koef*np.dot(w, w)
    test.sort()
    z = 20 * np.sin(2 * np.pi * 3 * test) + 100 * np.exp(test)
    print("Минимальная ошибка в валидационной части:", a)
    print("Ошибка в тестовой части:", E)
    plt.plot(test, z, 'r--', label='z')
    plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
    plt.plot(test, Y, 'b--', label='Y')
    plt.legend(loc='right')
    plt.title('Регрессия с регуляризацией')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
if funk_num == 3:
    best_koef = laambda[E_4.index(a)]
    best_funk = "4"
    print("Лучший коэффициент регуляризации: ", best_koef)
    print("Лучшие базовые функции: ", best_funk)
    test.sort()
    t_test.sort()
    F = Plan4(test)
    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + best_koef*l) @ F.T @ t_test
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_test - Y) ** 2) + best_koef*np.dot(w, w)
    test.sort()
    z = 20 * np.sin(2 * np.pi * 3 * test) + 100 * np.exp(test)
    print("Минимальная ошибка в валидационной части:", a)
    print("Ошибка в тестовой части:", E)
    plt.plot(test, z, 'r--', label='z')
    plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
    plt.plot(test, Y, 'b--', label='Y')
    plt.legend(loc='right')
    plt.title('Регрессия с регуляризацией')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
if funk_num == 4:
    best_koef = laambda[E_5.index(a)]
    best_funk = "5"
    print("Лучший коэффициент регуляризации: ", best_koef)
    print("Лучшие базовые функции: ", best_funk)
    test.sort()
    t_test.sort()
    F = Plan5(test)
    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + best_koef*l) @ F.T @ t_test
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_test - Y) ** 2) + best_koef*np.dot(w, w)
    test.sort()
    z = 20 * np.sin(2 * np.pi * 3 * test) + 100 * np.exp(test)
    print("Минимальная ошибка в валидационной части:", a)
    print("Ошибка в тестовой части:", E)
    plt.plot(test, z, 'r--', label='z')
    plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
    plt.plot(test, Y, 'b--', label='Y')
    plt.legend(loc='right')
    plt.title('Регрессия с регуляризацией')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
if funk_num == 5:
    best_koef = laambda[E_6.index(a)]
    best_funk = "6"
    print("Лучший коэффициент регуляризации: ", best_koef)
    print("Лучшие базовые функции: ", best_funk)
    test.sort()
    t_test.sort()
    F = Plan6(test)
    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + best_koef*l) @ F.T @ t_test
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_test - Y) ** 2) + best_koef*np.dot(w, w)
    test.sort()
    z = 20 * np.sin(2 * np.pi * 3 * test) + 100 * np.exp(test)
    print("Минимальная ошибка в валидационной части:", a)
    print("Ошибка в тестовой части:", E)
    plt.plot(test, z, 'r--', label='z')
    plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
    plt.plot(test, Y, 'b--', label='Y')
    plt.legend(loc='right')
    plt.title('Регрессия с регуляризацией')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
if funk_num == 6:
    best_koef = laambda[E_7.index(a)]
    best_funk = "7"
    print("Лучший коэффициент регуляризации: ", best_koef)
    print("Лучшие базовые функции: ", best_funk)
    test.sort()
    t_test.sort()
    F = Plan7(test)
    l = np.eye(F.shape[1])
    w = np.linalg.inv(F.T @ F + best_koef*l) @ F.T @ t_test
    Y = np.dot(w, F.T)
    E = (1 / 2) * sum((t_test - Y) ** 2) + best_koef*np.dot(w, w)
    test.sort()
    z = 20 * np.sin(2 * np.pi * 3 * test) + 100 * np.exp(test)
    print("Минимальная ошибка в валидационной части:", a)
    print("Ошибка в тестовой части:", E)
    plt.plot(test, z, 'r--', label='z')
    plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
    plt.plot(test, Y, 'b--', label='Y')
    plt.legend(loc='right')
    plt.title('Регрессия с регуляризацией')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

