# Dragon's code
# encoding = utf-8
# Created at 2020/12/19

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import myNetwork  # 自己的神经网络模型

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def load_2D_dataset(is_plot=True):
    data = sio.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    if is_plot:
        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);

    return train_X, train_Y, test_X, test_Y


def regulation_model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0.0,
                     keep_prob=1):
    costs = []
    layer_dims = [X.shape[0], 20, 3, 1]
    Parameters = myNetwork.initialize_parameters(layer_dims)

    for i in range(num_iterations):
        # 正向传播是否使用dropout
        if keep_prob == 1:
            a3, caches = myNetwork.model_forward(X, Parameters, 'relu')
        elif keep_prob < 1:
            a3, caches = model_forward_with_dropout(X, Parameters, keep_prob)
        else:
            print("keep_prob参数错误！程序退出。")
            exit()

        # 是否使用L2正则化
        if lambd == 0:
            cost = myNetwork.compute_cost(a3, Y)
        else:
            cost = compute_cost_with_lambd(a3, Y, Parameters, lambd)

        # 反向传播
        if lambd == 0 and keep_prob == 1:
            # 不使用dorpout和L2
            grads = myNetwork.model_backward(a3, Y, caches, 'relu')
        elif lambd != 0 and keep_prob < 1:
            grads = model_backward_with_lambd_and_dropout(X, Y, caches, lambd, keep_prob)
        elif lambd != 0:
            # 使用L2正则化不使用dropout
            grads = model_backward_with_lambd(X, a3, Y, caches, lambd)
        elif keep_prob < 1:
            # 使用dropout不使用L2正则化
            grads = model_backward_with_dropout(X, Y, caches, keep_prob)

        Parameters = myNetwork.update_parameters(Parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("第", i, "次学习，成本值为：", np.squeeze(cost))

    if is_plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return Parameters


def compute_cost_with_lambd(A, Y, parameters, lambd):
    m = Y.shape[1]
    W1, W2, W3 = parameters['W1'], parameters['W2'], parameters['W3']

    regulation_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)
    cost = myNetwork.compute_cost(A, Y) + regulation_cost

    return cost


def model_forward_with_dropout(X, Parameters, keep_prob=0.5):
    # 使用dropout进行向前传播
    np.random.seed(1)

    W1 = Parameters["W1"]
    b1 = Parameters["b1"]
    W2 = Parameters["W2"]
    b2 = Parameters["b2"]
    W3 = Parameters["W3"]
    b3 = Parameters["b3"]

    Z1 = W1.dot(X) + b1
    A1 = np.maximum(0, Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    # A1 = np.multiply(A1, np.int64(D1 < keep_prob)) / keep_prob
    # D1需要向后传递，上面的写法没有把D1进行dropout
    D1 = D1 < keep_prob
    A1 = np.multiply(A1, D1) / keep_prob

    Z2 = W2.dot(A1) + b2
    A2 = np.maximum(0, Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = np.multiply(A2, D2) / keep_prob

    Z3 = W3.dot(A2) + b3
    A3 = myNetwork.sigma(Z3)

    caches = ((X, W1, b1), (Z1, D1)), ((A1, W2, b2), (Z2, D2)), ((A2, W3, b3), (Z3, A3))

    return A3, caches


def model_backward_with_lambd(X, a3, Y, cache, lambd):
    # 计算使用L2正则化时各层的dW（增加了(lambd * W3) /m)
    m = X.shape[1]
    ((X, W1, b1), p1), ((A1, W2, b2), p2), ((A2, W3, b3), p3) = cache

    dZ3 = a3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def model_backward_with_dropout(X, Y, caches, keep_prob):
    m = X.shape[1]
    ((X, W1, b1), (Z1, D1)), ((A1, W2, b2), (Z2, D2)), ((A2, W3, b3), (Z3, A3)) = caches

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = np.multiply(dA2, D2) / keep_prob
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dA1 = np.multiply(dA1, D1) / keep_prob
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients


def model_backward_with_lambd_and_dropout(X, Y, caches, lambd, keep_prob):
    m = X.shape[1]
    ((X, W1, b1), (Z1, D1)), ((A1, W2, b2), (Z2, D2)), ((A2, W3, b3), (Z3, A3)) = caches
    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = np.multiply(dA2, D2) / keep_prob
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dA1 = np.multiply(dA1, D1) / keep_prob
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = myNetwork.model_forward(X, parameters, 'relu')
    predictions = (a3 > 0.5)
    return predictions


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y = load_2D_dataset(is_plot=False)
    Parameters = regulation_model(train_X, train_Y, learning_rate=0.007, num_iterations=500000,lambd=0, keep_prob=1, is_plot=True)
    print("训练集:")
    predictions_train = myNetwork.predict(train_X, train_Y, Parameters, 'relu')
    print("测试集:")
    predictions_test = myNetwork.predict(test_X, test_Y, Parameters, 'relu')

    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(Parameters, x.T), train_X, train_Y)