# coding = utf-8
# Dragon's Python3.8 code
# Created at 2020/12/29 20:09
# Edit with PyCharm
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import math

import myNetwork


plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def load_dataset(is_plot=True):
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    return train_X, train_Y


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数
    Args:
        parameters: 需要更新的参数
        grads: 每个参数对应的梯度值
        learning_rate: 学习率

    Returns:
        parameters: 更新后的参数
    """
    L = len(parameters) // 2  # 神经网络层数
    # 更新参数
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 打乱顺序
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]  # X[行，列] 将X的列随机排序
    shuffled_Y = Y[:, permutation].reshape(Y.shape[0], m)

    # 分割
    num_complete_minibatch = math.floor(m / mini_batch_size)  # 计算一共分割多少份
    for k in range(0, num_complete_minibatch):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batches.append((mini_batch_X, mini_batch_Y))
    # 处理剩余数据
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatch:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatch:]

        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


def initialize_velocity(parameters):
    """
    初始化速度，velocity是一个字典。key：“dW1”，“db1”，……“dWL”，“dbL”
                                value：对应的值
    Args:
        parameters: 包含各层W，b的字典

    Returns:
        v：字典。v["dW"+str(l)] = dW1的速度，v["db"+str(l)] = dbl的速度
    """
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        v['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    使用动量更新参数
    Args:
        parameters: 参数字典。 parameters['W'+str(l)] = Wl
        grads: 梯度字典。grads['dW'+str(l)] = dWl
        v: 保存当前速度的字典。v['dW'+str(l)] = ……
        beta: 参数beta
        learning_rate: 学习率
    Returns:
        parameters: 更新后的参数
        v: 更新后的速度
    """
    L = len(parameters) // 2
    for l in range(L):
        v['dW' + str(l + 1)] = beta * v['dW' + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]

        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]

    return parameters, v


def update_parameters_with_RMSProp(parameters, grads, S, beta, learning_rate):
    """
    使用RMSProp(平方根)更新参数
    Args:
        parameters: 参数字典。 parameters['W'+str(l)] = Wl
        grads: 梯度字典。grads['dW'+str(l)] = dWl
        S: 保存当前平方的字典。v['dW'+str(l)] = ……
        beta: 参数beta
        learning_rate: 学习率
    Returns:
        parameters: 更新后的参数
        v: 更新后的速度
    """
    L = len(parameters) // 2
    epsilon = 1e-8
    for l in range(L):
        S['dW' + str(l + 1)] = beta * S['dW' + str(l + 1)] + (1 - beta) * np.square(grads['dW' + str(l + 1)])
        S['db' + str(l + 1)] = beta * S['db' + str(l + 1)] + (1 - beta) * np.square(grads['db' + str(l + 1)])

        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)] / (
                    np.sqrt(S['dW' + str(l + 1)]) + epsilon)
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)] / (
                    np.sqrt(S['db' + str(l + 1)]) + epsilon)

    return parameters, S


def initialize_adam(parameters):
    """
    初始化v和s字典，key：dWl ， value: 与对应dW、db维度相同的零矩阵
    Args:
        parameters: 参数字典
    Returns:
        v，s字典
    """

    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        v['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])

        s['dW' + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        s['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用adam更新参数
    Args:
        parameters: 参数字典
        grads: 梯度字典
        v: 速度字典
        s: 平方梯度字典
        t: 当前迭代次数
        learning_rate:学习率
        beta1: momentum超参数，同时用与保证v起始不从0开始
        beta2: ROSProp超参数
        epsilon: 防止分母为零

    Returns:
        parameters: 更新后参数
        v: 更新后速度字典
        s: 更新后的平方和字典
    """
    L = len(parameters) // 2
    v_corrected = {}  # 偏差修正后的值
    s_corrected = {}  # 偏差修正后的值

    for l in range(L):
        # 计算momentum字典
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
        # 修正偏差
        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - np.power(beta1, t))
        # 计算平方梯度
        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * np.square(grads['dW' + str(l + 1)])
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * np.square(grads['db' + str(l + 1)])
        # 修正偏差
        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - np.power(beta2, t))
        # 更新参数
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v_corrected['dW' + str(l + 1)] \
                                       / (np.sqrt(s_corrected['dW' + str(l + 1)]) + epsilon)
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v_corrected['db' + str(l + 1)] \
                                       / (np.sqrt(s_corrected['db' + str(l + 1)]) + epsilon)

    return parameters, v, s


def model(X, Y, layers_dims, optimizer, learning_rate=0.0007,
          mini_batch_size=64, beta=0.9, beta1=0.0, beta2=0.999,
          epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):
    """
    可以运行在不同优化器下的三层神经网络
    Args:
        X: 输入数据，维度为（2，样本数量）
        Y: 数据对应的标签
        layers_dims: 层数和和节点数的列表
        optimizer: 优化器，['gd','momentum','adam']
        learning_rate: 学习率
        mini_batch_size: 每个小数据的大小
        beta: momentum优化的超参数
        beta1: adam的超参数1
        beta2: adam的超参数2
        epsilon: adam中防止除零的一个很小的数
        num_epochs: 训练集遍历次数，代数
        print_cost:是否打印误差值，每1000代打印一次，100代记录一次误差
        is_plot:是否绘制曲线图

    Returns:
        parameters: 学习后的参数
    """
    L = len(layers_dims)
    costs = []  # 记录成本
    t = 0  # 每学习一个mini-batch就加一
    seed = 10  # 随机种子
    parameters = myNetwork.initialize_parameters(layers_dims)

    # 选择优化器
    if optimizer == 'gd':
        pass
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        v, s = initialize_adam(parameters)
    elif optimizer == 'RMS':
        s = initialize_velocity(parameters)
    else:
        print("optimizer参数错误，程序退出")
        exit(1)

    # 开始学习
    for i in range(num_epochs):
        seed += 1  # 每个epoch都用不同的mini-batch排列
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            A3, cache = myNetwork.model_forward(minibatch_X, parameters, 'relu')
            cost = myNetwork.compute_cost(A3, minibatch_Y)
            grads = myNetwork.model_backward(A3, minibatch_Y, cache, 'relu')

            # 更新参数
            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, v, grads, beta, learning_rate)
            elif optimizer == 'adam':
                t += 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,
                                                               epsilon)
            elif optimizer == 'RMS':
                parameters, s = update_parameters_with_RMSProp(parameters, grads, s, beta2, learning_rate)
        if i % 100 == 0:
            costs.append(cost)
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
    # 是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

if __name__ == '__main__':

    train_X, train_Y = load_dataset(is_plot=False)
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="RMS", is_plot=True)
