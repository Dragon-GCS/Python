# coding = utf-8
# Dragon's Python3.8 code
# Created at 2020/12/31 21:19
# Edit with PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import time,math


tf.compat.v1.disable_eager_execution()

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) / 255
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) / 255
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

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

def create_placeholders(n_x, n_y):
    x = tf.compat.v1.placeholder(tf.float32, [n_x, None], name="X")
    y = tf.compat.v1.placeholder(tf.float32, [n_y, None], name="Y")

    return x, y

def initialize_parameters():
    """
    初始化神经网络，参数维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]
    Returns:
        parameters: 包含了W，b的字典
    """
    tf.random.set_seed(1)
    parameters = {"W1": tf.compat.v1.get_variable("W1", [25,12288], initializer = tf.initializers.GlorotUniform(seed = 1)),
                  "b1": tf.compat.v1.get_variable("b1", [25,1], initializer = tf.zeros_initializer()),
                  "W2": tf.compat.v1.get_variable("W2", [12,25], initializer = tf.initializers.GlorotUniform(seed = 1)),
                  "b2": tf.compat.v1.get_variable("b2", [12,1], initializer = tf.zeros_initializer()),
                  "W3": tf.compat.v1.get_variable("W3", [6,12], initializer = tf.initializers.GlorotUniform(seed = 1)),
                  "b3": tf.compat.v1.get_variable("b3", [6,1], initializer = tf.zeros_initializer())}

    return parameters

def initialize_parameters_for_iris():
    """
    初始化神经网络，参数维度如下：
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]
    Returns:
        parameters: 包含了W，b的字典
    """
    tf.random.set_seed(1)
    parameters = {"W1": tf.compat.v1.get_variable("W1", [4,4], initializer = tf.initializers.GlorotUniform(seed = 1)),
                  "b1": tf.compat.v1.get_variable("b1", [4,1], initializer = tf.zeros_initializer()),
                  "W2": tf.compat.v1.get_variable("W2", [4,4], initializer = tf.initializers.GlorotUniform(seed = 1)),
                  "b2": tf.compat.v1.get_variable("b2", [4,1], initializer = tf.zeros_initializer()),
                  "W3": tf.compat.v1.get_variable("W3", [3,4], initializer = tf.initializers.GlorotUniform(seed = 1)),
                  "b3": tf.compat.v1.get_variable("b3", [3,1], initializer = tf.zeros_initializer())}

    return parameters

def forward_prop(X, parameters):
    """
    实现一个模型的向前传播：LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Args:
        X: 输入数据的占位符
        parameters: 参数字典
    Returns:
        Z3: 最后的输出
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)

    return Z3

def compute_cost(Z3,Y):
    """
    计算成本
    Args:
        Z3:前向传播结果
        Y: 标签占位符
    Returns:
        cost: 成本值
    """
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))

    return cost

def model(train_x,train_y,test_x,test_y,learning_rate=0.0001,num_epochs=1500,
          minibatch_size=256,print_plot=True,is_plot=True):
    """
    实现一个三层的TF神经网络LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Args:
        train_x: 训练集，维度为（12288，1080）
        train_y: 训练集分类数量（6，1080）
        test_x:  测试集，维度为（12288.120）
        test_y:   测试集分类数量（6，120）
        learn_rate: 学习速率
        num_epochs: 训练集遍历次数
        minibatch_size: 每个小数据集大小
        print_plot: 是否打印成本
        is_plot: 是否绘制曲线图
    Returns:
        parameters: 学习后参数
    """
    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.random.set_seed(1)
    seed = 3
    (n_x, m) = train_x.shape  # 输入节点数量和样本数
    n_y = train_y.shape[0]  # 输出节点数量
    costs = []

    x, y = create_placeholders(n_x,n_y)
    parameters = initialize_parameters()
    # 正向传播
    Z3 = forward_prop(x, parameters)
    # 计算成本
    cost = compute_cost(Z3,y)
    # 反向传播，adam优化
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # 初始化所有变量
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    # 开始计算
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost=0
            num_minibatches = int( m / minibatch_size)
            seed += 1
            minibatches = random_mini_batches(train_x, train_y,minibatch_size,seed)

            for minibatch in minibatches:
                (minibatch_x,minibatch_y) = minibatch
                _, minibatch_cost = sess.run([optimizer,cost],feed_dict={x:minibatch_x, y:minibatch_y})
                epoch_cost += minibatch_cost / num_minibatches

            if epoch % 10 == 0:
                costs.append(epoch_cost)
                if print_plot and epoch % 100==0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))
        # 是否绘制散点图
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('Cost')
            plt.xlabel('Epoch(per tens)')
            plt.title("Learn Rate = " + str(learning_rate))
            plt.show()
        # 保存参数
        parameters = sess.run(parameters)
        print("参数已保存至Session")
        # 计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(y))
        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({x: train_x, y: train_y}))
        print("测试集的准确率:", accuracy.eval({x: test_x, y: test_y}))
        saver.save(sess, "TF_Model/")
    return parameters

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, classes = load_dataset()
    x_train = x_train.reshape(x_train.shape[0],-1).T
    x_test = x_test.reshape(x_test.shape[0],-1).T

    y_train = np.eye(6)[y_train.reshape(-1)].T      # one-hot
    y_test = np.eye(6)[y_test.reshape(-1)].T        # one-hot

    print(x_test.shape)
    print(y_test.shape)

    #开始时间
    start_time = time.perf_counter()
    #开始训练
    parameters = model(x_train, y_train, x_test, y_test)
    #结束时间
    end_time = time.perf_counter()
    #计算时差
    print("GPU的执行时间 = " + str(end_time - start_time) + " 秒" )