import numpy as np
import matplotlib.pyplot as plt
import h5py


def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

X, Y = load_planar_dataset()

def load_dataset():
    train_dataset = h5py.File('training_data/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('training_data/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    return train_set_x / 255, train_set_y_orig, \
           test_set_x / 255, test_set_y_orig, classes


# X_train, Y_train, X_test, Y_test, classes = load_dataset()


def sigma(z):
    s = 1 / (1 + np.exp(-z))
    return s


def layer_size(x, y):
    """
    定义神经网络结构
    :param
        x - 输入数据集，维度为(输入的数量，训练/测试的数量）
        y - 标签，维度为（输入的数量，训练/测试的数量）
    :return
        n_x - 输入层数量
        n_h - 隐藏层节点数
        n_y - 输出层数量
    """
    n_x = x.shape[0]  # 输入层
    n_h = 4  # 隐藏层，默认为4层
    n_y = y.shape[0]  # 输出层
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    初始化神经网络中的参数
    :param
        n_x - 输入层数量
        n_h - 隐藏层节点数
        n_y - 输出层数量
    :return
        parameters - 包含参数的字典：
            W1 - 权重矩阵,维度为（n_h，n_x）
            b1 - 偏向量，维度为（n_h，1）
            W2 - 权重矩阵，维度为（n_y，n_h）
            b2 - 偏向量，维度为（n_y，1）
    """
    W1 = np.random.randn(n_h, n_x) * 0.001
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.001
    b2 = np.zeros(shape=(n_y, 1))

    #使用断言确保我的数据格式是正确的
    assert(W1.shape == ( n_h , n_x ))
    assert(b1.shape == ( n_h , 1 ))
    assert(W2.shape == ( n_y , n_h ))
    assert(b2.shape == ( n_y , 1 ))

    parameters = {
        "W1": W1,
        "b1": b1,
        'W2': W2,
        'b2': b2}

    return parameters


def forward_propagation(X, parameters):
    """"
    正向传播函数
    :param
        X - 测试的数据集，维度（单个样本数据量（即神经网络输入层n_x),样本数量)
        parameter - 单隐层神经网络参数
    :return
        A2 - 使用sigma函数计算后的激活值
        cache - 包含Z1，Z2，A1，A2，
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = W1.dot(X) + b1
    A1 = np.tanh(Z1)  # 隐藏层使用tanh函数
    # A1 = np.maxium(0,Z1) * Z1  # 隐藏层使用ReLU函数
    Z2 = W2.dot(A1) + b2
    A2 = sigma(Z2)
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    计算成本
    :param
        A2 - 预测输出
        Y - 样本实际输出
        parameters - 神经网络参数
    :return
        cost - 成本
    """
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    cost = float(np.squeeze(cost))
    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    向后传播函数
    :param
        parameters - 神经网络参数字典
        cache - 包含A1，A2，Z1，Z2的字典
        X - 输入数据，维度（n_x,样本数）
        Y - 输入数据对应的标签，维度（1，样本数）
    :return
        grads - 包含梯度的字典
    """
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1, 2))  # 使用tanh的dZ1
    # dZ1 = np.dot(W2.T,dZ2) * 1 * (Z1 > 0)  # 使用ReLU的dZ1
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters,grads, learning_rate):
    """
    更新神经网络参数
    :param
        parameters - 神经网络参数
        grads - 梯度字典
        learning_rate - 学习速率
    :return:
        parameters - 包含新参数的字典
    """
    dW1, dW2, db1, db2 = grads['dW1'],grads['dW2'],grads['db1'],grads['db2']
    W1, W2, b1, b2 = parameters['W1'],parameters['W2'],parameters['b1'],parameters['b2']

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def model(X,Y,n_h,num_iterations,learning_rate = 1.2,print_cost=False):
    """
    神经网络模型
    :param
        X - 数据集
        Y - 数据集标签
        n_h - 隐藏层数量
        num_iterations - 迭代次数
        print_cost - 是否每1000次迭代打印一次成本。
    :return:
        parameters - 用于预测的神经网络参数
        costs - 迭代过程中的所有损失函数
    """
    n_x = layer_size(X,Y)[0]
    n_y = layer_size(X,Y)[2]
    costs = []
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']


    for i in range(num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        costs.append(cost)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost and i%1000 ==0:
            print('第{}次循环，成本为{}'.format(i,cost))
    return parameters,costs


def predict(parameters, X):
    """
    使用模型参数预测数据
    :param
        parameters - 神经网络训练得到的参数
        X - 待预测数据
    :return
        prediction - 预测结果
    """
    A2,cache = forward_propagation(X, parameters)
    return np.round(A2)



accurs = []
for i in range(1,101):
    rate = i/100
    parameters, costs = model(X, Y, n_h=5, num_iterations=5000, learning_rate=rate, print_cost=False)
    predictions = predict(parameters, X)
    accur = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    accurs.append(accur)
    if i%10 == 0:
        print (f'第{i}次循环，准确率: {accur:.2f}%')

plt.plot(accurs)
plt.show()



