# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/8 21:22
# Edit with PyCharm

import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def zero_pad(X,pad):
    """
    将数据集X的边界全部用0来扩充pad个宽度和高度
    Args:
        X: 待填充的数据集，维度（样本数，图像高度，图像宽度，通道数）
        pad: 整数，填充的数量
    Returns:
        X_paded： 填充后的数据集，维度（样本数，高度+2*pad，宽度+2*pad，通道数）
    """
    X_paded = np.pad(X,(
                        (0,0),       # 样本数，不填充
                        (pad,pad),   # 图像高度，上面填充pad，下面填充pad
                        (pad,pad),   # 图像宽度，左面填充pad，右面填充pad
                        (0,0)),      # 通道数，不听啊冲
                        'constant', constant_values=0)   # 填充值，使用0填充。（x,y）表示前面用x填，后面用y填。
    return X_paded

def conv_single_step(a_slice_prev,W,b):
    """
    在前一层的激活输出的一个片段上应用一个由W定义的滤波器（卷积核）
    切片大小与过滤器大小相同
    Args:
        a_slice_prev: 输入数据的一个片段，维度为（f_h,f_w,n_c)
        W: 权重参数，也可以认为是卷积核，维度为（f_h,f_w,n_c)
        b: 编制参数，包含在矩阵中，维度为（1，1，1）
    Returns:
        Z：在输入片段上卷积滑动的结果
    """
    s = np.multiply(a_slice_prev,W) + b

    return np.sum(s)

def conv_forward(A_prev,W,b,hyperparams):
    """
    实现卷积网络的前向传播，使用for循环，未使用向量化
    Args:
        A_prev: 前一层的激活输出矩阵，维度为(m,n_h_prev,m_w_prev,n_c_prev)
        W: 权重矩阵，也可以认为是卷积核，维度为（f，f，n_c_prev，n_c）(卷积核大小，卷积核大小，上一层过滤器数，本层过滤器数)
        b: 偏执矩阵，维度为（1，1，1，n_c）（1，1，1，这一层的过滤器数量）
        hyperparams:包含了"stride"和"pad"的超参数字典
    Returns:
        Z: 卷积输出，维度为（m，n_h，n_W，n_c）（样本数，图像高度，图像宽度，过滤器数量）
        cache: 缓存了一些反向传播函数conv_backward()需要的数据
    """
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    (f, f, n_c_prev, n_c) = W.shape
    stride = hyperparams['stride']
    pad = hyperparams['pad']
    # 计算卷积后的凸显宽高，使用int向下取整
    n_h = int((n_h_prev + 2*pad - f) / stride) + 1
    n_w = int((n_w_prev + 2*pad - f) / stride) + 1
    # 初始化卷积输出Z
    Z = np.zeros((m, n_h, n_w, n_c))
    # 将输入数据A_prev使用0填充每个通道的高和宽
    A_prev_paded = zero_pad(A_prev,pad)

    for i in range(m):                      # 遍历样本（第一维度）
        a_prev_paded = A_prev_paded[i]      # 选择第i个样本的扩充后的激活矩阵
        for h in range(n_h):                # 在输出的垂直轴上循环（第二维度）
            for w in range(n_w):            # 在输出的水平轴上循环（第三维度）
                for c in range(n_c):        # 在输出的通道轴上遍历（第四维度）
                    # 定位切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horizon_start = w * stride
                    horizon_end = horizon_start + f
                    # 取出当前通道的切片
                    a_slice_prev = a_prev_paded[vert_start:vert_end,
                                                horizon_start:horizon_end,
                                                :]  # shape(f,f,n_c_prev)
                    # 使用第c个卷积核（f,f,n_c_prev)卷积该片段（f,f,n_c_prev
                    Z[i,w,h,c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[0,0,0,c])
    assert Z.shape == (m, n_h, n_w, n_c)
    cache = (A_prev,W,b,hyperparams)

    return (Z,cache)

def pool_forward(A_prev,hyperparams,mode='max'):
    """
    实现池化层的前向传播
    Args:
        A_prev: 输入数据，维度为（m，n_h_prev，n_w_prev，n_c_prev）
        hyperparams: 包含了"f"和"stride"的超参数字典
        mode: 池化模式【"max"|"average"】
    Returns:
        A: 池化层的输出，维度为（m，n_h，n_w，n_c）
        cache: 存储一些反向传播用到的值，包含了输入和超参数的字典
    """
    (m, n_h_prev, n_w_prev, n_c_prev) = A_prev.shape
    f = hyperparams['f']
    stride = hyperparams['stride']
    # 输出维度
    n_h = int((n_h_prev - f) / stride) + 1   # 池化时默认没有pad
    n_w = int((n_w_prev - f) / stride) + 1   # 池化时默认没有pad
    # 初始化输出矩阵
    A = np.zeros((m, n_h, n_w, n_c_prev))

    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c_prev):
                    # 定位切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # 获取片段
                    a_slice_prev = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]

                    # 池化
                    if mode == 'max':
                        A[i,h,w,c] = np.max(a_slice_prev)
                    elif mode == 'average':
                        A[i,h,w,c] = np.mean(a_slice_prev)
    assert A.shape == (m,n_h,n_w,n_c_prev)
    cache = (A_prev,hyperparams)
    return (A,cache)

def conv_backward(dZ,cache):
    """
    实现卷积网络的反向传播
    Args:
        dZ: 卷积层的输出Z的梯度
        cache: 反向传播需要的参数（A_prev，W，b，hyperparam）
    Returns:
        dA_prev: 卷积层的输入（A_prev）的梯度，维度（m，n_h_prev，n_w_prev，n_c_prev）
        dW: 卷积层W的梯度值，维度（f，f，n_c_prev，n_c）
        db: 卷积层b的梯度值，维度（1，1，1，n_c）
    """
    (A_prev,W,b,hyperparams) = cache
    (m,n_h_prev,n_w_prev,n_c_prev) = A_prev.shape
    (m,n_h,n_w,n_c) = dZ.shape
    (f,f,n_c_prev,n_c) = W.shape
    pad = hyperparams['pad']
    stride = hyperparams['stride']
    # 初始化各个梯度
    dA_prev = np.zeros((m,n_h_prev,n_w_prev,n_c_prev))
    dW = np.zeros((f,f,n_c_prev,n_c))
    db = np.zeros((1,1,1,n_c))

    # 使用pad与前向传播一致
    A_prev_paded = zero_pad(A_prev,pad)
    dA_prev_paded = zero_pad(dA_prev,pad)

    for i in range(m):
        A_prev_paded = A_prev_paded[i]
        dA_prev_paded = dA_prev_paded[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # 定位切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # 获取片段
                    a_slice = A_prev_paded[vert_start:vert_end, horiz_start:horiz_end, :]
                    # 计算梯度
                    dA_prev_paded[vert_start:vert_end,horiz_start:horiz_end] += W[:,:,:,c]*dZ[i,w,h,c]
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
        dA_prev[i,:,:,:] = dA_prev[i,pad:-pad,pad:-pad,:]

    assert dA_prev.shape == (m,n_h_prev,n_w_prev,n_c_prev)
    return dA_prev,dW,db

def create_mask_from_window(X):
    """
    从输入矩阵创建掩码，用于保存最大值在矩阵中的位置，用于max池化反向传播
    Args:
        X: 一个维度为（f，f）的矩阵
    Returns:
        mask: 包含X最大值的位置的矩阵
    """
    return X == np.max(X)

def distribute_value(dz,shape):
    """
    给定一个值，为矩阵大小平均分配到每一个矩阵位置中，用于average池化反向传播
    Args:
        dz: 输入的实数
        shape: （n_h,n_w）
    Returns:
        a: 分配好的矩阵，里面值都一样
    """
    n_h, n_w = shape
    average = dz/(n_h * n_w)
    return  np.ones(shape) * average

def pool_backward(dA,cache,mode='max'):
    """
    池化层的反向传播函数
    Args:
        dA: 池化层输出的梯度，与A维度相同
        cache: 池化层的缓存，（A_prev，hyperparams）
        mode: 池化模式【"max"|"average"】
    Returns:
       dA_prev: 池化层的输入梯度，与A_prev维度相同
    """
    A_prev,hyperparams = cache
    f = hyperparams['f']
    stride = hyperparams['stride']
    (m,n_h_prev,n_w_prev,n_c_prev) = A_prev.shape
    (m, n_h, n_w, n_c) = dA.shape
    # 初始化dA_prev
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # 定位切片位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # 选择反向传播的计算方式
                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += np.multiply(mask,dA[i,h,w,c])
                    elif mode == 'average':
                        da = dA[i,h,w,c]
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += distribute_value(da,(f,f))
    assert dA_prev.shape == A_prev.shape
    return dA_prev
