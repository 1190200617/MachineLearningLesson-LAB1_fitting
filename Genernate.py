import math
import random
import numpy as np
import matplotlib.pyplot as plt

def sort(x, y, num):
    for i in range(num):
        j=i+1
        while(j<num):
            if x[j] < x[i]:
                tmp = x[j]
                x[j] = x[i]
                x[i] = tmp
                tmp = y[j]
                y[j] = y[i]
                y[i] = tmp
            j += 1
    return x, y

def Gen(begin, end, train_num, val_num, test_num, mid=0, sigma=0.1):     # 生成数据集，sin(2*pi*x), x:[0, 1], 带高斯噪声（均值为0，方差为0.15）
    total = train_num + val_num + test_num
    sample_x = list(np.linspace(begin, end, total))
    sample_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []

    for i in range(total):
        sample_y.append(math.sin(2 * math.pi * sample_x[i]))
        sample_y[i] += random.gauss(mid, sigma)

    # 生成验证集
    for i in range(val_num):
        k = random.randint(0, len(sample_x)-1)
        val_x.append(sample_x[k])
        val_y.append(sample_y[k])
        del sample_x[k]
        del sample_y[k]
    val_x, val_y = sort(val_x, val_y, val_num)

    # 生成测试集
    for i in range(test_num):
        k = random.randint(0, len(sample_x)-1)
        test_x.append(sample_x[k])
        test_y.append(sample_y[k])
        del sample_x[k]
        del sample_y[k]
    test_x, test_y = sort(test_x, test_y, test_num)
    filename = 'train1.txt'
    file = open(filename, 'w')
    for i in range(len(sample_x)):
        s = str(sample_x[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    for i in range(len(sample_y)):
        s = str(sample_y[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()

    filename = 'val1.txt'
    file = open(filename, 'w')
    for i in range(len(val_x)):
        s = str(val_x[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    for i in range(len(val_y)):
        s = str(val_y[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()

    filename = 'test1.txt'
    file = open(filename, 'w')
    for i in range(len(test_x)):
        s = str(test_x[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    for i in range(len(test_y)):
        s = str(test_y[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()

    plt.scatter(sample_x, sample_y)
    plt.show()

    return sample_x, sample_y, val_x, val_y, test_x, test_y

if __name__ == '__main__':
    Gen(begin=0, end=1, train_num=20, val_num=10, test_num=10)
