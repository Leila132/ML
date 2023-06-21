import numpy as np
import matplotlib.pyplot as plt

# матрицa Плана
def Plan_matrix(M, N, X):
    F = np.ones([N, M + 1])
    for i in range(N):
        for j in range(1, M + 1):
            F[i][j] = X[i] ** j
    return F

N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

# 1 случай
M = 1
F = Plan_matrix(M, N, x)
w = np.linalg.inv(np.transpose(F) @ F) @ np.transpose(F) @ t
Y = np.dot(w, F.T)
plt.plot(x, z, 'r--', label='z')
plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
plt.plot(x, Y, 'black', label='Y')
plt.legend(loc='right')
plt.title('M = 1')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 2 случай
M = 8
F = Plan_matrix(M, N, x)
w = np.linalg.inv(np.transpose(F) @ F) @ np.transpose(F) @ t
Y = np.dot(w, F.T)
plt.plot(x, z, 'r--', label='z')
plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
plt.plot(x, Y, 'black', label='Y')
plt.legend(loc='right')
plt.title('M = 8')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 3 случай
M = 100
F = Plan_matrix(M, N, x)
a = 1
I = np.eye(F.shape[1])
w = np.linalg.inv(np.transpose(F) @ F + a*I) @ np.transpose(F) @ t
Y = np.dot(w, F.T)
plt.plot(x, z, 'r--', label='z')
plt.scatter(x, t, 10, '#9999ff', 'o', label='t')
plt.plot(x, Y, 'black', label='Y')
plt.legend(loc='right')
plt.title('M = 100')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

E = np.ones(100)
for M in range(1, 101):
    F = Plan_matrix(M, N, x)
   # w = np.linalg.inv(np.transpose(F) @ F) @ np.transpose(F) @ t
    w= np.linalg.pinv(F) @ t
    # F * w = t
    # 
    Y = np.dot(w, F.T)
    E[M - 1] = (1 / 2) * sum((t - Y) ** 2)
M = np.array(range(1, 101))
plt.plot(M, E, alpha=0.7)
plt.title('Зависимости ошибки E(w) от степени полинома M')
plt.xlabel('M')
plt.ylabel('E')
plt.show()
