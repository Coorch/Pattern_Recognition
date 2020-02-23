from sklearn import svm
import numpy as np
import math
import matplotlib.pyplot as plt


def sk_svr(X, Y):
    '''
        sklearn自带的SVR求解
        X = [x, y]
        Y = -x^2 - y^2
        (a,b)是圆心的坐标
        r是圆的半径
    '''
    clf = svm.SVR(kernel='linear')
    clf.fit(X, Y)
    #rint(clf.coef_, clf.intercept_)
    a = - 1 * clf.coef_[0][0] / 2
    b = - 1 * clf.coef_[0][1] / 2
    R = a * a + b * b - clf.intercept_[0]
    r = math.sqrt(R)
    return [a, b, r]


def my_svr(x, y, epoches, rho, rho_1, rho_2, rho_3):
    '''
        对朗格朗日对偶函数求梯度下降，得到对偶变量的最优解
        通过KKT条件，求出原变量a，b和R
    '''
    alpha_1 = np.zeros(len(x))
    alpha_2 = np.zeros(len(x))
    x_1 = np.zeros(len(x))
    y_1 = np.zeros(len(x))
    for i in range(epoches):
        for j in range(len(x)):
            x_1[j] = (alpha_1[j] - alpha_2[j]) * x[j]
            y_1[j] = (alpha_1[j] - alpha_2[j]) * y[j]
        for j in range(len(x)):
            alpha_1[j] = alpha_1[j] - rho * ((np.sum(x_1) * x[j] + np.sum(y_1) * y[j]) * 1 - (x[j] * x[j] + y[j] * y[j]) + (rho_1 + rho_2) * alpha_1[j] - c * rho_1 + rho_3 * (np.sum(alpha_1)-np.sum(alpha_2)))
            alpha_2[j] = alpha_2[j] - rho * ((np.sum(x_1) * x[j] + np.sum(y_1) * y[j]) * (-1) + (x[j] * x[j] + y[j] * y[j]) + (rho_1 + rho_2) * alpha_2[j] - c * rho_1 + rho_3 * (np.sum(alpha_2)-np.sum(alpha_1)))
    #print(alpha_1)
    #print(alpha_2)
    omega_1 = -1 * np.sum(x_1)
    omega_2 = -1 * np.sum(y_1)
    #print(omega_1)
    #print(omega_2)
    a = -1 * omega_1 / 2
    b = -1 * omega_2 / 2
    R = x[0] * x[0] + omega_1 * x[0] + a * a + y[0] * y[0] + omega_2 * y[0] + b * b
    r = math.sqrt(R)
    return [a, b, r]


def plot_circle(x, y, a, b, r):
    
    theta = np.arange(0, 2 * np.pi, 0.01)
    m = a + r * np.cos(theta)
    n = b + r * np.sin(theta)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(x, y, 'ro')
    axes.plot(m, n)
    axes.axis('equal')
    axes.set_title('data0')

    plt.xlabel('x')
    plt.ylabel('y')
    #plt.xlim((0, 1.5))
    plt.show()


def cal_error(x, y, a, b, r):


    error = 0
    for i in range(len(x)):
        error = error + abs(x[i] * x[i] - 2 * a * x[i] + y[i] * y[i] - 2 * b * y[i] + a * a + b * b - r * r)
    print(error)
    return error


x = []
y = []
with open('data/data2.txt') as f:
    line = f.readline()
    while line:
        s = line.split(' ')
        x.append(float(s[0]))
        y.append(float(s[1].replace('\n', '')))
        line = f.readline()
x = np.array(x)
y = np.array(y)
X = np.zeros((len(x), 2))
X[:,0] = x
X[:,1] = y
Y = np.zeros((len(x)))
for i in range(len(x)):
    Y[i] = - x[i] * x[i] - y[i] * y[i]

[a, b, r] = sk_svr(X, Y)
print([a, b, r])

error = cal_error(x, y, a, b, r)

plot_circle(x, y, a, b, r)


epoches = 100
rho = 0.1
c = 1
rho_1 = 0.1
rho_2 = 0.5
rho_3 = 10
[a, b, r] = my_svr(x, y, epoches, rho, rho_1, rho_2, rho_3)
print([a, b, r])

error = cal_error(x, y, a, b, r)

plot_circle(x, y, a, b, r)