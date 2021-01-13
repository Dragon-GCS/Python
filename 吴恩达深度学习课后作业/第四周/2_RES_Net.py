# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/10 22:43
# Edit with PyCharm

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
from tensorflow import keras


def load_dataset():
    train_dataset = h5py.File('./data/datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('./data/datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def identity_block(x,f,filters,stage,block):
    """
    实现恒等块，跳跃连接，幅度为三层，第一层和第三层为(1,1)卷积核，第二层卷积核维度为函数输入
    -->conv2D->BatchNorm->Relu-->conv2D->BatchNorm->Relu-->conv2D->BatchNorm-+->Relu
     |-----------------------------------------------------------------------↑
    Args:
        x: 输入的Tensor数据，维度（m, n_h_prev, n_w_prev, n_c_prev)
        f: 整数，主路径中间的卷积层的维度
        filters: 整数列表，主路经中每个卷积层的卷积核数目
        stage: 整数，根据每层位置命名每层，与block一起使用
        block: 字符串，根据每层位置命名每层，与stage一起使用
    Returns:
        x: 恒等块的输出，维度为(n_h, n_w, n_c)
    """
    # 定义块内名称
    conv_name_base = "res" + str(stage) + block + "_brach"
    bn_name_base = "bn" + str(stage) + block + "_brach"
    # 获取过滤器
    F1, F2, F3, = filters
    # 保存输入数据，用于添加捷径
    x_shortcut = x
    # 第一个 ->conv2D->BatchNorm->Relu 部分
    x = keras.layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid',
                            name=conv_name_base+'2a', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = keras.layers.BatchNormalization(axis=3,name=bn_name_base+'2a')(x)
    x = keras.layers.Activation('relu')(x)
    # 第二个 ->conv2D->BatchNorm->Relu  部分
    x = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                            name=conv_name_base + '2b', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)
    # 第三个 ->conv2D->BatchNorm 部分
    x = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                            name=conv_name_base + '2c', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # 合并捷径
    x = keras.layers.Add()([x, x_shortcut])
    # 最后的激活函数
    x = keras.layers.Activation('relu')(x)

    return x

def convolutional_block(x,f,filters,stage,block,s=2):
    """
    实现卷积块，适用于输入输出维度不一致的情况
    -->conv2D->BatchNorm->Relu-->conv2D->BatchNorm->Relu-->conv2D->BatchNorm-+->Relu
     |-------------------------->conv2D->BatchNorm--------------------------↑
    Args:
        x: 输入的tensor类型的变量，维度（m, n_h_prev, n_w_prev, n_c_prev)
        f: 整数，主路径中间的卷积层的维度
        filters: 整数列表，主路经中每个卷积层的卷积核数目
        stage: 整数，根据每层位置命名每层，与block一起使用
        block: 字符串，根据每层位置命名每层，与stage一起使用
        s: 整数，要使用的stride（步长）
    Returns:
        x: 恒等块的输出，维度为(n_h, n_w, n_c)
    """
    # 定义块内名称
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    # 获取过滤器
    # 获取过滤器
    F1, F2, F3, = filters
    # 保存输入数据，用于添加捷径
    x_shortcut = x
    # 第一个 ->conv2D->BatchNorm->Relu 部分
    x = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                            name=conv_name_base + '2a', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)
    # 第二个 ->conv2D->BatchNorm->Relu  部分
    x = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                            name=conv_name_base + '2b', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)
    # 第三个 ->conv2D->BatchNorm 部分
    x = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                            name=conv_name_base + '2c', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    # 捷径
    x_shortcut = keras.layers.Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s), padding='valid',
                                     name=conv_name_base + '1',
                                     kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x_shortcut)
    x_shortcut = keras.layers.BatchNormalization(axis=3,name=bn_name_base+'1')(x_shortcut)
    # 合并捷径
    x = keras.layers.Add()([x, x_shortcut])
    # 最后的激活函数
    x = keras.layers.Activation('relu')(x)

    return x

def ResNet50(input_shape=(64,64,3),classes=6):
    """
    实现50层残差网络，结构如下
    conv2D -> BatchNorm -> RELU -> MaxPool -> ConvBlock -> IdBlock*2 -> ConvBlock -> IdBlock*3
    -> ConvBlock -> IdBlock*5 -> ConvBlock -> IdBlock*2 -> AvgPool -> TopLayer
    Args:
        input_shape: 图形数据的维度
        classes: 整数，分类数
    Returns:
        model: 使用keras构建的ResNet50模型
    """
    x_input = keras.Input(input_shape)
    # 0填充
    x = keras.layers.ZeroPadding2D((3,3))(x_input)
    # stage1    conv2D -> BatchNorm -> RELU -> MaxPool
    x = keras.layers.Conv2D(64,(7,7),(2,2),name='conv1',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = keras.layers.BatchNormalization(axis=3,name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    # stage2    ConvBlock -> IdBlock*2
    filter = [64,64,256]
    x = convolutional_block(x, 3, filter, stage=2, block='a', s=1)
    x = identity_block(x, 3, filter, stage=2, block='b')
    x = identity_block(x, 3, filter, stage=2, block='c')
    # stage3    ConvBlock -> IdBlock*3
    filter = [128,128,512]
    x = convolutional_block(x, 3, filter, stage=3, block='a', s=2)
    x = identity_block(x, 3, filter, stage=3, block='b')
    x = identity_block(x, 3, filter, stage=3, block='c')
    x = identity_block(x, 3, filter, stage=3, block='d')
    # stage4    ConvBlock -> IdBlock*5
    filter = [256, 256, 1024]
    x = convolutional_block(x, 3, filter, stage=4, block='a', s=2)
    x = identity_block(x, 3, filter, stage=4, block='b')
    x = identity_block(x, 3, filter, stage=4, block='c')
    x = identity_block(x, 3, filter, stage=4, block='d')
    x = identity_block(x, 3, filter, stage=4, block='e')
    x = identity_block(x, 3, filter, stage=4, block='f')
    # stage5    ConvBlock -> IdBlock*2
    filter = [512, 512, 2048]
    x = convolutional_block(x, 3, filter, stage=5, block='a', s=2)
    x = identity_block(x, 3, filter, stage=5, block='b')
    x = identity_block(x, 3, filter, stage=5, block='c')
    # AvgPool
    x = keras.layers.AveragePooling2D(pool_size=(2,2), padding='same')(x)
    # 输出层
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(classes, activation='softmax', name='fc'+str(classes),
                           kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)

    return keras.Model(x_input, x, name='ResNet50')

if __name__ == '__main__':
    model = ResNet50(input_shape=(64,64,3), classes=6)
    # model.summary()

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = load_dataset()
    x_train = x_train_orig / 255.
    x_test = x_test_orig / 255.
    y_train = np.eye(6)[y_train_orig.reshape(-1)]
    y_test = np.eye(6)[y_test_orig.reshape(-1)]

    history = model.fit(x_train, y_train, epochs=50,batch_size=32)
    model.evaluate(x_test, y_test)

    plt.plot(history.history['loss'])
    plt.show()

    model.save('data/model/ResNet50.h5')
    # 预测图片
    from PIL import Image
    model = keras.models.load_model('data/model/ResNet50.h5')
    classes = [1, 2, 3, 4, 5, 6]
    for i in range(6):
        img = np.array(Image.open(f'data/sign{i+1}.jpg')) / 255.
        img = tf.image.resize(img, (64, 64))
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        print(f'图中手势为{i+1}，预测图中的手势为：{classes[np.argmax(pred)]}')
