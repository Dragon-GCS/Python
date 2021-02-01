# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/19 20:07
# Edit with PyCharm

import numpy as np

def softmax(x):
    "计算softmax"
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rnn_cell_forward(xt, a_prev, parameters):
    """
    实现向前传播的单个RNN神经元
    Args:
        xt: 时间步‘t’输入的数据（n_x, m）
        a_prev: 时间步‘t-1’的隐藏状态（n_a， m）
        parameters:  参数字典，包含：
                    Wax - 矩阵，输入权重，维度为（n_a, n_x）
                    Waa - 矩阵，隐藏层权重，维度为（n_a, n_a）
                    Wya - 矩阵，输出权重， 维度为（n_y, n_a）
                    ba - 输入偏置（n_a, 1）
                    by - 输出偏置（n_y, 1）
    Returns:
        a_next: 下一个隐层状态（n_a, m）
        yt_pred: 时间步‘t’的预测输出（n_y, m）
        cache: 反向传播需要的信息，包含（a_next, a_prev, xt, parameters)
    """
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    # 计算下一个时间步的激活值
    a_next = np.tanh(Waa.dot(a_prev) + Wax.dot(xt) + ba)
    # 预测输出y_pred
    yt_pred = softmax(Wya.dot(a_next) + by)
    # 缓存反向传播数据
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    实现rnn的前向传播
    Args:
        x: 输入的全部数据（n_x, m, T_x)
        a0: rnn神经元的初始化状态（n_a, m）
        parameters:  参数字典，包含：
                    Wax - 矩阵，输入权重，维度为（n_a, n_x）
                    Waa - 矩阵，隐藏层权重，维度为（n_a, n_a）
                    Wya - 矩阵，输出权重， 维度为（n_y, n_a）
                    ba - 输入偏置（n_a, 1）
                    by - 输出偏置（n_y, 1）
    Returns:
        a: 所有时间步的隐藏状态（n_a, m, T_x）
        y_pred: 所有时间步的预测（n_y, m, T_x）
        caches: 反向传播所需要的数据（cache列表，x）
    """
    # 初始化缓存列表
    caches = []
    # 获取x与Wya维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape
    # 初始化
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])
    a_next = a0
    # 遍历时间步
    for t in range(T_x):
        # 使用神经元更新next状态和cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # 使用a保存next状态（第t个位置)
        a[:,:,t] = a_next
        # 使用y保存预测值
        y_pred[:, :, t] = yt_pred
        # 保存cache
        caches.append(cache)
    caches = (caches, x)

    return a, y_pred, caches


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    前向传播的长短时记忆神经元
    Args:
        xt: 在时间步“t”输入的数据，维度为(n_x, m)
        a_prev: 上一个时间步“t-1”的隐藏状态，维度为(n_a, m)
        c_prev: 上一个时间步“t-1”的记忆状态，维度为(n_a, m)
        parameters: 字典类型的变量，包含了：
                        Wf: 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf: 遗忘门的偏置，维度为(n_a, 1)
                        Wi: 更新门的权值，维度为(n_a, n_a + n_x)
                        bi: 更新门的偏置，维度为(n_a, 1)
                        Wc: 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc: 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo: 输出门的权值，维度为(n_a, n_a + n_x)
                        bo: 输出门的偏置，维度为(n_a, 1)
                        Wy: 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by: 隐藏状态与输出相关的偏置，维度为(n_y, 1)

    Returns:
        a_next: 下一个隐藏状态，维度为(n_a, m)
        c_next: 下一个记忆状态，维度为(n_a, m)
        yt_pred: 在时间步“t”的预测，维度为(n_y, m)
        cache: 包含了反向传播所需要的参数(a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    """
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    # 连接a_prev与xt
    contact = np.zeros([n_a+n_x, m])
    contact[:n_a,:] = a_prev
    contact[n_a:,:] = xt
    # 计算ft,it,cct(c tilda),c_next, ot, a_next
    ft = sigmoid(Wf.dot(contact) + bf)      # 计算遗忘门
    it = sigmoid(Wi.dot(contact) + bf)      # 计算更新门
    cct = np.tanh(Wc.dot(contact) + bc)     # 计算临时c
    c_next = it*cct + ft*c_prev             # 新c = 旧c * 遗忘门 + 新临时c * 更新门

    ot = sigmoid(Wo.dot(contact) + bo)      # 计算输出门
    a_next = ot * np.tanh(c_next)           # 输出 = 输出门 * 新c

    yt_pred = softmax(Wy.dot(a_next) + by)  # 使用softmax输出y的预测值
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    Args:
        x: 输入的全部数据（n_x, m, T_x)
        a0: rnn神经元的初始化状态（n_a, m）
        parameters:  参数字典，包含：
                    Wf: 遗忘门的权值，维度为(n_a, n_a + n_x)
                    bf: 遗忘门的偏置，维度为(n_a, 1)
                    Wi: 更新门的权值，维度为(n_a, n_a + n_x)
                    bi: 更新门的偏置，维度为(n_a, 1)
                    Wc: 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                    bc: 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                    Wo: 输出门的权值，维度为(n_a, n_a + n_x)
                    bo: 输出门的偏置，维度为(n_a, 1)
                    Wy: 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                    by: 隐藏状态与输出相关的偏置，维度为(n_y, 1)
    Returns:
        a: 所有时间步的隐藏状态（n_a, m, T_x）
        y_pred: 所有时间步的预测（n_y, m, T_x）
        caches: 反向传播所需要的数据（cache列表，x）
    """
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    # 初始化
    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])
    a_next = a0
    c_next = np.zeros([n_a, m])
    # 遍历时间步
    for t in range(T_x):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:,:,t],a_next,c_next,parameters)
        # 保存输出值
        a[:,:,t] = a_next
        y[:,:,t] = yt_pred
        c[:,:,t] = c_next
        caches.append(cache)
    # 保存反向传播数据
    caches = (caches, x)

    return a, y, c, caches


def rnn_cell_backward(da_next, cache):
    """
    基本RNN神经元的反向传播
    Args:
        da_next: 下一个时间步的梯度
        cache: 缓存数据（a_next, a_prev, xt, parameters)
    Returns:
        gradients: 字典，包含了以下参数：
            dx: 输入数据的梯度，维度为(n_x, m)
            da_prev: 上一隐藏层的隐藏状态，维度为(n_a, m)
            dWax: 输入到隐藏状态的权重的梯度，维度为(n_a, n_x)
            dWaa: 隐藏状态到隐藏状态的权重的梯度，维度为(n_a, n_a)
            dba: 偏置向量的梯度，维度为(n_a, 1)
    """
    a_next, a_prev, xt, parameters = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 正向公式: a_next = np.tanh(Waa.dot(a_prev) + Wax.dot(xt) + ba)
    dtanh = (1 - np.square(a_next)) * da_next

    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    dba = np.sum(dtanh, keepdims=True, axis=-1)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


def rnn_backward(da, caches):
    """
    RNN的反向传播
    Args:
        da: 所有隐藏状态的梯度（n_a, m, T_x）
        caches: 前向传播缓存的数据（caches，x）
    Returns:
        gradients: 字典，包含了以下参数：
            dx: 输入数据的梯度，维度为(n_x, m)
            da_prev: 上一隐藏层的隐藏状态，维度为(n_a, m)
            dWax: 输入到隐藏状态的权重的梯度，维度为(n_a, n_x)
            dWaa: 隐藏状态到隐藏状态的权重的梯度，维度为(n_a, n_a)
            dba: 偏置向量的梯度，维度为(n_a, 1)
    """
    caches, x = caches
    a1, a0, x1, parameters = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    # 处理所有时间步
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t]+da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]

        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
    da0 = da_prevt
    # 保存这些梯度到字典内
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


def lstm_cell_backward(da_next, dc_next, cache):
    """
    长短时记忆神经元的反向传播
    Args:
        da_next: 下一个隐藏状态的梯度（n_a, m）
        dc_next: 下一个记忆细胞的梯度（n_a, m）
        cache: 前向传播的缓存(a_next, c_next, a_prev, c_prev, xt, parameters)
    Returns:
        gradients: 包含了梯度信息的字典：
                dxt: 输入数据的梯度，维度为(n_x, m)
                da_prev: 先前的隐藏状态的梯度，维度为(n_a, m)
                dc_prev: 前的记忆状态的梯度，维度为(n_a, m, T_x)
                dWf: 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                dbf: 遗忘门的偏置的梯度，维度为(n_a, 1)
                dWi: 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                dbi: 更新门的偏置的梯度，维度为(n_a, 1)
                dWc: 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                dbc: 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                dWo: 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                dbo: 输出门的偏置的梯度，维度为(n_a, 1)

    """
    a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters = cache
    n_x, m = xt.shape
    n_a, m = a_next.shape

    dot = da_next * np.tanh(c_next) * ot * 1-(ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh((c_next))) * it * da_next) * (1 - np.square(cct)))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    concat = np.concatenate((a_prev, xt), axis=0).T
    dWf = np.dot(dft, concat)
    dWi = np.dot(dit, concat)
    dWc = np.dot(dcct, concat)
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) + np.dot(
        parameters["Wi"][:, :n_a].T, dit) + np.dot(parameters["Wo"][:, :n_a].T, dot)

    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next

    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) + np.dot(
        parameters["Wi"][:, n_a:].T, dit) + np.dot(parameters["Wo"][:, n_a:].T, dot)

    # 保存梯度信息到字典
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


def lstm_backward(da, caches):
    """
    实现LSTM的反向传播
    Args:
        da: 所有隐藏状态的梯度（n_a, m, T_x）
        caches: 前向传播的缓存(caches, x)
    Returns:
        gradients: 包含了梯度信息的字典：
                dxt: 输入数据的梯度，维度为(n_x, m)
                da_prev: 先前的隐藏状态的梯度，维度为(n_a, m)
                dc_prev: 前的记忆状态的梯度，维度为(n_a, m, T_x)
                dWf: 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                dbf: 遗忘门的偏置的梯度，维度为(n_a, 1)
                dWi: 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                dbi: 更新门的偏置的梯度，维度为(n_a, 1)
                dWc: 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                dbc: 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                dWo: 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                dbo: 输出门的偏置的梯度，维度为(n_a, 1)
    """
    caches, x =caches
    a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters = caches[0]
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    #初始化梯度
    dx = np.zeros([n_x, m, T_x])
    da0 = np.zeros([n_a, m])
    da_prev = np.zeros([n_a, m])
    dc_prev = np.zeros([n_a, m])
    dWf = np.zeros([n_a, n_a + n_x])
    dWi = np.zeros([n_a, n_a + n_x])
    dWc = np.zeros([n_a, n_a + n_x])
    dWo = np.zeros([n_a, n_a + n_x])
    dbf = np.zeros([n_a, 1])
    dbi = np.zeros([n_a, 1])
    dbc = np.zeros([n_a, 1])
    dbo = np.zeros([n_a, 1])
    # 处理所有时间步
    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[:,:,t]+da_prev,dc_prev, caches[t])
        # 保存参数
        dx[:, :, t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWi = dWi + gradients['dWi']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbi = dbi + gradients['dbi']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
        da_prev = da_prev + gradients['da_prev']
        dc_prev + dc_prev + gradients['dc_prev']
    # 将第一个激活的梯度设置为反向传播的梯度da_prev。
    da0 = gradients['da_prev']

    # 保存所有梯度到字典变量内
    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients
