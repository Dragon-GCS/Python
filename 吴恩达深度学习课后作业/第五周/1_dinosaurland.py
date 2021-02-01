# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/19 22:12
# Edit with PyCharm
import time
import numpy as np


def softmax(x):
    "计算softmax"
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def clip(gradients, max_value):
    """
    实现梯度修建，将梯度限制在[-max，max]之间
    Args:
        gradients: 梯度字典
        max_value: 阈值
    Returns:
        gradients: 梯度字典
    """
    for key, value in gradients.items():
        np.clip(value, -max_value, max_value, out=value)
        gradients[key] = value

    return gradients


def sample(parameters, char_to_ix, seed):
    """
    根据RNN输出的概率分布序列对字符序列进行采样
    Args:
        parameters: 参数字典
        char_to_ix: 字符到索引的映射字典
        seed: 随机种子
    Returns:
        indices: 包含采样字符索引的长度为n的列表
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    # 使用零向量初始化输入x0和a0
    x = np.zeros((vocab_size,1))
    a_prev = np.zeros((n_a,1))
    # 创建索引的空列表，包含要生成的字符的索引
    indices = []
    # idx是检测换行符“\n”的标志，初始化为-1
    idx = -1
    # 遍历时间步，每个时间步中从概率分布中抽取一个字符并将其索引增加到indices中
    count = 0
    newline_character = char_to_ix['\n']
    while idx != newline_character and count < 50:
        a = np.tanh(Wax.dot(x) + Waa.dot(a_prev) + b)
        z = Wya.dot(a) + by
        y = softmax(z)
        # 设定随机种子
        np.random.seed(count+seed)
        # 按照输出y的概率分布抽取索引值
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())    # 按y的概率分布从vocab中选择索引值
        indices.append(idx)
        # 将索引对应至x中
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        a_prev = a  # 更新a_prev

        seed += 1
        count += 1
    if count == 50:
        indices.append(char_to_ix["\n"])

    return indices


def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values

    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x) * 0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a) * 0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a) * 0.01  # hidden to output
    b = np.zeros((n_a, 1))  # hidden bias
    by = np.zeros((n_y, 1))  # output bias

    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

    return parameters


def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)  # hidden state
    p_t = softmax(
        np.dot(Wya, a_next) + by)  # unnormalized log probabilities for next chars # probabilities for next chars

    return a_next, p_t


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']  # backprop into h
    daraw = (1 - a * a) * da  # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b'] += -lr * gradients['db']
    parameters['by'] += -lr * gradients['dby']
    return parameters


def rnn_forward(X, Y, a0, parameters, vocab_size=27):
    # Initialize x, a and y_hat as empty dictionaries
    x, a, y_hat = {}, {}, {}

    a[-1] = np.copy(a0)

    # initialize your loss to 0
    loss = 0

    for t in range(len(X)):

        # Set x[t] to be the one-hot vector representation of the t'th character in X.
        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector.
        x[t] = np.zeros((vocab_size, 1))
        if (X[t] != None):
            x[t][X[t]] = 1

        # Run one step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t - 1], x[t])

        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= np.log(y_hat[t][Y[t], 0])

    cache = (y_hat, a, x)

    return loss, cache


def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}

    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])

    return gradients, a


def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    执行模型的单步优化
    Args:
        X: 整数列表，每个整数映射到词汇表中的一个单词或字符
        Y: 整数列表，与X相同但向左移动了一个索引
        a_prev: 上一个隐藏状态
        parameters: 字典，包含了以下参数：
                        Wax： 权重矩阵乘以输入，维度为(n_a, n_x)
                        Waa： 权重矩阵乘以隐藏状态，维度为(n_a, n_a)
                        Wya： 隐藏状态与输出相关的权重矩阵，维度为(n_y, n_a)
                        b: 偏置，维度为(n_a, 1)
                        by: 隐藏状态与输出相关的权重偏置，维度为(n_y, 1)
        learning_rate: 学习率
    Returns:
        loss: 损失函数的值
        gradients: 更新后的字典
        a[len(x)-1]: 最后的隐藏状态
    """
    # 前向传播
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    # 反向传播
    gradients, a = rnn_backward(X, Y, parameters, cache)
    # 梯度修剪
    gradients = clip(gradients, 5)
    # 更新参数
    parameters = update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X)-1]


def model(data, ix_to_char, char_to_ix, num_iteration=3500,lr=0.01, n_a=50, dino_names=7, vocab_size=27):
    """
    训练模型并生成恐龙的名字
    Args:
        data: 语料库
        ix_to_char: 索引到字符的字典
        char_to_ix: 字符到索引的字典
        num_iteration: 迭代次数
        n_a: RNN单元数量
        dino_names: 每次迭代中采样的数量
        vocab_size: 文本中唯一字符的数量
    Returns:
        parameters: 学习后参数
    """
    # 初始化
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a,n_x,n_y)
    loss = -np.log(1.0/vocab_size)*dino_names
    a_prev = np.zeros((n_a,1))
    # 构建恐龙名称列表
    with open('data/dinos.txt') as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    # 打乱恐龙名称
    np.random.seed(0)
    np.random.shuffle(examples)
    # 训练
    for j in range(num_iteration):
        # 定义一个训练样本
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix['\n']]
        # 执行单步optimize
        curr_loss, gradients, a_prev = optimize(X,Y,a_prev,parameters,lr)
        loss = loss * 0.999 + curr_loss * 0.001 # 保持损失平滑，可以加速训练
        if j%2000 == 0:
            print("第" + str(j+1) + "次迭代，损失值为：" + str(loss))
            seed = 0
            for name in range(dino_names):
                # 采样
                sampled_indices = sample(parameters, char_to_ix, seed)
                txt = ''.join(ix_to_char[ix] for ix in sampled_indices)
                txt = txt[0].upper() + txt[1:]  # 首字母大写
                print(txt, end='')
                seed += 1
    return parameters


if __name__ == '__main__':
    # 读取恐龙名字
    with open('data/dinos.txt', 'r') as f:
        data = f.read().lower()
    # 转换为无序且不重复的字符列表
    chars = list(set(data))
    # 获取数据大小信息
    data_size, vocab_size = len(data), len(chars)

    # 创建字符映射的字典，包含“\n”
    char_to_ix = {ch:i for i,ch in enumerate(sorted(chars))}    # 字符：索引
    ix_to_char = {i:ch for i,ch in enumerate(sorted(chars))}    # 索引: 字符

    start_time = time.time()
    parameters = model(data, ix_to_char, char_to_ix, num_iteration=3500,lr=1e-3)
    end_time = time.time()
    # 计算时差
    minium = end_time - start_time
    print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium % 60)) + "秒")