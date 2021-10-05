import numpy as np
import matplotlib.pyplot as plt
import os
import math

def Readfile(train_num, val_num, test_num):
    train_x = []
    train_y = []
    f = open("train.txt")
    for i in range(train_num*2):
        line = f.readline().split()
        if i < train_num:
            train_x.append(float(line[0]))
        else:
            train_y.append(float(line[0]))
    # plt.scatter(train_x, train_y)
    # plt.show()
    f.close()

    val_x = []
    val_y = []
    f = open("val.txt")
    for i in range(val_num * 2):
        line = f.readline().split()
        if i < val_num:
            val_x.append(float(line[0]))
        else:
            val_y.append(float(line[0]))
    # plt.scatter(val_x, val_y)
    # plt.show()
    f.close()

    test_x = []
    test_y = []
    f = open("test.txt")
    for i in range(test_num * 2):
        line = f.readline().split()
        if i < test_num:
            test_x.append(float(line[0]))
        else:
            test_y.append(float(line[0]))
    # plt.scatter(test_x, test_y)
    # plt.show()
    f.close()
    return train_x, train_y, val_x, val_y, test_x, test_y

def func(W, x, N):    # 拟合出的多项式函数
    list = []
    for i in range(N+1):
        tmpv = x**i
        list.append(tmpv)
    X = np.mat(list)
    pred = np.dot(W.T, X.T)
    return pred[0, 0]

def vande(N, x):      # 生成范德蒙德矩阵
    X = np.zeros((N + 1, len(x)))
    # 转为范德蒙德
    for i in range(N + 1):
        tmp = []
        for j in range(len(x)):
            tmpv = x[j] ** i
            tmp.append(tmpv)
        X[i] = tmp
    return X

def draw(x, y, W, N):   #画图像
    x_norm = np.linspace(0, 1.0, 100)
    y_norm = []
    for i in range(100):
        y_norm.append(math.sin(2 * math.pi * x_norm[i]))
    # 绘制回归曲线
    pred = []
    k = np.linspace(0, 0.9, 60)
    for i in range(len(k)):
        pre = func(W, k[i], N)
        pred.append(pre)

    plt.scatter(x, y)
    plt.plot(x_norm, y_norm, label = "$sin$", color = "red")
    plt.plot(k, pred, label = "$fit$", color = "blue")
    plt.legend()
    title = 'N=' + '%d' % N
    plt.title(title)
    # plt.savefig(title + '.jpg')
    plt.show()

def LSM(x, y, N, param=0.0):     # 最小二乘法
    X = vande(N, x)
    Y = np.mat(y)

    # 计算W
    W = np.linalg.inv(np.dot(X, X.T) + param * np.eye(N+1)).dot(X).dot(Y.T)
    print("W:")
    print(W.T)

    # 计算loss
    print("loss: ", calc_loss(W, X, Y, param))

    #画图
    draw(x, y, W, N)
    return W

def calc_loss(W, X, Y, param):     # 计算loss
    loss = 0.5 * np.dot((np.dot(W.T, X) - Y), (np.dot(W.T, X) - Y).T) + 0.5 * param * np.dot(W.T, W)
    return loss[0, 0]

def calc_param(val_x, val_y, train_x, train_y, N, num):     # 确定参数param
    loss = []
    list = np.linspace(0.1, 8, 88)
    for i in range(len(list)):
        param = 10 ** (-list[i])
        W = LSM(train_x, train_y, N, param)
        certain_loss = 0
        for i in range(num):
            certain_loss += (val_y[i] - func(W, val_x[i], N))**2
        loss.append((certain_loss/num)**0.5)
    print(loss)
    plt.plot(list, loss)
    title = 'N=' + '%d' % N
    plt.title(title)
    plt.show()

def calc_grad(W, X, Y, param):
    return np.dot(X, X.T).dot(W) - np.dot(X, Y.T) + param * W

def reshape(W, x, y):
    X = vande(W.shape[0] - 1, x)
    Y = np.mat(y)
    return X, Y

def GD(W, x, y, lr, epoch, param):    #梯度下降法
    loss = []
    Epoch = np.linspace(0, epoch-1, epoch)
    X, Y = reshape(W, x, y)
    loss_1 = 100
    for i in range(epoch):
        gradient = calc_grad(W, X, Y, param)
        #print(gradient)
        W = W - lr * gradient
        #print(W)
        loss_0 = loss_1
        loss_1 = calc_loss(W, X, Y, param)
        loss.append(loss_1)
        if abs(loss_1 - loss_0) < 1e-20:
            print(i, loss_0, loss_1)
            break
    Epoch = Epoch[1000:]
    loss = loss[1000:]
    #print(loss[epoch-1])
    draw(x, y, W, W.shape[0] - 1)
    # 保存模型
    filename = 'weight.txt'
    file = open(filename, 'w')
    W = W.T
    W = W.tolist()
    W = W[0]
    for i in range(len(W)):
        s = str(W[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()

    # 绘图 loss-epoch
    plt.plot(Epoch, loss)
    title = 'loss-epoch'
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def CD(W, x, y, epoch, param):    #共轭梯度法
    loss = []
    Epoch = np.linspace(0, epoch - 1, epoch)
    X, Y = reshape(W, x, y)
    Alpha = np.dot(X, X.T) + param * np.eye(X.shape[0])
    beta = np.dot(X, Y.T)
    r = beta
    p = beta
    loss_1 = 100
    for i in range(epoch):
        norm_2 = np.dot(r.T, r)
        a = norm_2 / np.dot(p.T, Alpha).dot(p)
        W = W + a[0, 0] * p
        r = r - (a[0, 0] * Alpha).dot(p)
        beta = np.dot(r.T, r) / norm_2
        p = r + beta[0, 0] * p
        loss_0 = loss_1
        loss_1 = calc_loss(W, X, Y, param)
        loss.append(loss_1)
        """if abs(loss_1 - loss_0) < 1e-7:
            print(i)
            break"""
    draw(x, y, W, W.shape[0] - 1)

    # 保存模型
    filename = 'weight_CD.txt'
    file = open(filename, 'w')
    W = W.T
    W = W.tolist()
    W = W[0]
    for i in range(len(W)):
        s = str(W[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()

    # 绘图 loss-epoch
    plt.plot(Epoch, loss)
    title = 'loss-epoch'
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def rand_weight(N):
    W = np.zeros((N + 1, 1))
    #W = np.random.rand(N + 1, 1)
    return W

def load_weight(N, filename):
    W = []
    if os.path.exists(filename):
        f = open(filename)
        for i in range(N + 1):
            line = f.readline().split()
            W.append(float(line[0]))
        # plt.scatter(val_x, val_y)
        # plt.show()
        f.close()
        W = np.mat(W)
        W = W.T
    else:
        W = rand_weight(N=9)
    return W

if __name__ == '__main__':
    train_x, train_y , val_x, val_y, test_x, test_y = Readfile(train_num=10, val_num=10, test_num=10)
    #W = LSM(train_x, train_y, N=9)
    #print(W)

    # 确定param
    """for i in range(5, 10):
        calc_param(val_x, val_y, train_x, train_y, N=i, num=len(val_x))"""

    param = 1e-4    # train.txt

    # 进行LSM
    """for i in range(3, 15):
        print("N = ", i)
        LSM(train_x, train_y, N=i, param=param)"""

    # 进行GD
    """W = load_weight(N=9, filename='weight.txt')
    lr = 0.14    # train.txt

    GD(W, train_x, train_y, lr=lr, epoch=20000, param=param)"""

    # 进行CD
    # lr =1e-5
    W_CD = load_weight(N=9, filename='weight_CD.txt')
    CD(W_CD, train_x, train_y, epoch=40, param=param)