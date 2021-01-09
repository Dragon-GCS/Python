import numpy as np
import h5py
import matplotlib.pyplot as plt
import time


def load_dataset():
    train_dataset = h5py.File('training_data/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('training_data/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classess = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    return train_set_x / 255, train_set_y_orig, \
           test_set_x / 255, test_set_y_orig, classess


def sigma(z):

    return 1 / (1 + np.exp(-z))


def sigma_primer(Z):
    a = sigma(Z)
    return a * (1 - a)


def relu_primer(dA, Z):
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ


def tanh_primer(Z):
    return 1 - np.power(np.tanh(Z), 2)


def initialize_parameters(layer_dims):  # 输入层和输出层？
    """
    初始化多层神经网络的参数
    :param
        layer_dims - 网络中每层节点数的列表,包含输入层和输入层
    :return
        parameters - 包含参数W[l]和b[l]的字典
    """
    np.random.seed(3)
    parameter = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameter['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameter['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert parameter['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameter['b' + str(l)].shape == (layer_dims[l], 1)

    return parameter


def forward_propagation(A_prev, W, b, activation):
    """
    实现激活函数
    :param
        A_prev - 前一层的输出，维度（上一层节点数，样本数）
        W - 权重数组，维度（本层节点数，上一层节点数）
        b - 偏向量，维度（本层节点数，1）
        activation - 选择本层使用的激活函数【“sigma”|“relu”】
    :return
        A - 激活函数输出
        cache - 包含"linear_cache"和"activation_cache"的字典
    """
    Z = W.dot(A_prev) + b
    assert Z.shape == (W.shape[0], A_prev.shape[1])

    if activation == 'sigma':
        A = sigma(Z)
    elif activation == 'relu':
        A = np.maximum(0, Z)
    elif activation == 'tanh':
        A = np.tanh(Z)

    assert (W.shape[0], A_prev.shape[1]) == A.shape
    cache = ((A_prev, W, b), Z)
    return A, cache


def model_forward(X, parameter, hidden_activations):
    """
    实现多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION

    :param
        X - 数据，numpy数组，维度为（输入节点数量，示例数）
        parameters - initialize_parameters_deep（）的输出
    :return
        AL - 最后的激活值
        caches - 包含以下内容的缓存列表：
                 linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）
                 linear_sigmoid_forward（）的cache（只有一个，索引为L-1）
    """
    caches = []
    A = X
    L = len(parameter) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = forward_propagation(A_prev, parameter['W' + str(l)], parameter['b' + str(l)], hidden_activations)
        caches.append(cache)

    AL, cache = forward_propagation(A, parameter['W' + str(L)], parameter['b' + str(L)], 'sigma')
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])
    return AL, caches


def compute_cost(AL, Y):
    """
    计算成本
    :param
        AL - 神经网络的计算值
        Y - 实际值
    :return
        cost - 计算得到的成本
    """
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    assert cost.shape == ()
    return cost


def backward_propagation(dA, cache, activation):
    """
    计算当前层的dW、db与前一层的dA（dA_prev）

    :param
        dA - 当前层的梯度值
        cache - 元组，（linear_cache,activation_cache）
        activation - 当前层激活函数
    :return
        dA_prev - 前一层的梯度值
        dW - 当前层的dW
        db - 当前层的db
    """
    linear_cache, Z_l = cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    if activation == 'relu':
        dZ = relu_primer(dA, Z_l)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
    elif activation == 'sigma':
        dZ = dA * sigma_primer(Z_l)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
    elif activation == 'tanh':
        dZ = dA * tanh_primer(Z_l)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def model_backward(AL, Y, caches, hidden_activations):
    """
    :param
        AL - 正向传播计算得到
        Y - 样本实际值矩阵
        caches - 包含以下内容的cache列表：
                linear_activation_forward（"relu"）的cache，不包含输出层
                linear_activation_forward（"sigmoid"）的cache
    :return
        grads - 含梯度值的字典
                grads [“dA”+ str（l）] = ...
                grads [“dW”+ str（l）] = ...
                grads [“db”+ str（l）] = ...
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = backward_propagation(dAL, caches[L - 1],
                                                                                                'sigma')
    # 反向逐层计算该层的dW，db和dA_prev
    while L - 1:
        L -= 1  # 当前层数，计算完第一层后停止。
        dA_prev_temp, dW_temp, db_temp = backward_propagation(grads['dA' + str(L)], caches[L - 1], hidden_activations)
        grads['dA' + str(L - 1)] = dA_prev_temp
        grads['dW' + str(L)] = dW_temp
        grads['db' + str(L)] = db_temp

    return grads


def update_parameters(parameteres, grads, learning_rate):
    L = len(parameteres) // 2
    for i in range(L):
        parameteres['W' + str(i + 1)] = parameteres['W' + str(i + 1)] - learning_rate * grads['dW' + str(i + 1)]
        parameteres['b' + str(i + 1)] = parameteres['b' + str(i + 1)] - learning_rate * grads['db' + str(i + 1)]
    return parameteres


def multi_layers_model(X, Y, layer_dims, hidden_activations, learning_rate=0.0075, iteration=3000, print_cost=True,
                       isPlot=True):
    costs = []
    Parameters = initialize_parameters(layer_dims)

    for i in range(0, iteration):
        AL, caches = model_forward(X, Parameters, hidden_activations)  # 正向计算得到AL和每一层的dA_prev、W、b和Z缓存，dA_prev和Z用于计算梯度
        cost = compute_cost(AL, Y)
        grads = model_backward(AL, Y, caches, hidden_activations)  # 根据输出层的AL和每一层对应的缓存计算每层的梯度，返回梯度字典
        Parameters = update_parameters(Parameters, grads, learning_rate)  # 根据梯度字典更新W，b

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第", i, "次学习，成本值为：", np.squeeze(cost))

    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return Parameters


def predict(X, Y, parameter, hidden_activations):
    m = Y.shape[1]
    # L = len(parameter) // 2

    prediction, caches = model_forward(X, parameter, hidden_activations)
    p = np.round(prediction)

    accuracy = float((np.sum(p * Y + (1 - p) * (1 - Y))) / m) * 100
    print("准确度为:{}% ".format(accuracy))
    return prediction


def print_mislabeled_images(classess, X, y, p):
    """
    绘制预测和实际不同的图像。
        X - 数据集
        y - 实际的标签
        p - 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classess[int(p[0, index])].decode("utf-8") + " \n Class: " + classess[y[0, index]].decode(
                "utf-8"))
        plt.show()


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, classes = load_dataset()
    hidden_dim = [20, 7, 5]
    layers_dims = [X_train.shape[0]] + hidden_dim + [Y_train.shape[0]]

    start = time.time()
    parameters = multi_layers_model(X_train, Y_train, layers_dims, 'relu', learning_rate=0.0075, iteration=2500)

    import cv2
    label = np.array([1]).reshape(-1,1)
    fname = "training_data/cat.jpg"
    img = cv2.imread(fname)
    image = cv2.resize(img, (64, 64))
    s = predict(image.reshape(64 * 64 * 3, 1), label, parameters, 'relu')
    print('测试图片为猫的概率为'+str(s[0,0]*100)+'%')
    plt.imshow(image)
    plt.show()

    pred_train = predict(X_train, Y_train, parameters, hidden_activations='relu')  # 训练集
    pred_test = predict(X_test, Y_test, parameters, hidden_activations='relu')  # 测试集
    # print_mislabeled_images(classes, X_test, Y_test, pred_test)
    end = time.time()
    print('用时为：'+str(end-start))