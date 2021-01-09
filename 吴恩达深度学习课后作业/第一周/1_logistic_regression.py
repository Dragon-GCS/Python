import numpy as np
import matplotlib.pyplot as plt
import h5py


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


def sigma(z):
    s = 1 / (1 + np.exp(-z))

    return s


def initialize_with_zero(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    # 确保数据正确
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]  # 获取样本数量
    A = sigma(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    # 写法2：
    # cost = (-1 / m) * (Y.dot(np.log(A.T)) + 1-Y.dot(np.log(1-A.T)))

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    # 确保数据正确
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return dw, db, cost


def logistic_regression(w, b, X, Y, num_iteration, alpha, print_cost=False):
    costs = []

    for i in range(num_iteration):
        dw, db, cost = propagate(w, b, X, Y)

        w = w - alpha * dw
        b = b - alpha * db
        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print('迭代次数：{},误差值：{:.2f}'.format(i, cost))

        params = {
            'w': w,
            'b': b,
            'dw': dw,
            'db': db,
        }
    return params, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_predict = np.zeros((1, m))
    # X情况下Y为1的概率
    A = sigma(np.dot(w.T, X) + b)

    for i in range(m):
        Y_predict[0, i] = 1 if A[0, i] > 0.5 else 0
    assert (Y_predict.shape == (1, m))

    return Y_predict


def model(X_train, Y_train, X_test, Y_test, alpha, iteration, print_cost=False):
    # 初始化w, b
    w, b = initialize_with_zero(X_train.shape[0])
    # 获取训练后的参数以及过程中的成本
    params, costs = logistic_regression(w, b, X_train, Y_train, iteration, alpha, print_cost)
    # 对测试集进行预测
    Y_predict_train = predict(params['w'], params['b'], X_train)
    Y_predict_test = predict(params['w'], params['b'], X_test)
    # 计算预测率Y_predict_train - Y_train正确为0，错误为±1
    accuracy_train = 100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100
    accuracy_test = 100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100
    print("训练集的准确性：{:.2f}%".format(accuracy_train))
    print("测试集的准确性：{:.2f}%".format(accuracy_test))

    d = {
        'w': params['w'],
        'b': params['b'],
        'costs': costs,
        'Y_predict_test': Y_predict_test,
        'Y_predict_train:': Y_predict_train
    }

    return d


X_train, Y_train, X_test, Y_test, classes = load_dataset()

d = model(X_train, Y_train, X_test, Y_test, 0.009, 10000, 1)

costs = d['costs']
plt.plot(costs)
plt.title('Learning Rate')
plt.ylabel('costs')
plt.xlabel('iterations (per hundreds)')
plt.show()
