# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/10 10:37
# Edit with PyCharm

from tensorflow import keras
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    train_dataset = h5py.File('data/datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('data/datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()

    train_imgs = train_set_x_orig / 255.0
    train_labs = train_set_y_orig.reshape(-1, 1)

    test_imgs = test_set_x_orig / 255.0
    test_labs = test_set_y_orig.reshape(-1, 1)
    # 使用tf.keras构筑模型
    model = keras.Sequential([
        keras.layers.ZeroPadding2D((3, 3), input_shape=(train_imgs.shape[1:])),  # 使用0填充周围三层
        keras.layers.Conv2D(32, (7, 7), name='conv0'),  # 32个7x7卷积核
        keras.layers.BatchNormalization(axis=3, name='bn0'),  # 归一化
        keras.layers.Activation('relu'),  # 激活层
        keras.layers.MaxPooling2D((2, 2), name='max_pool'),  # 最大值池化
        keras.layers.Flatten(),  # 展平
        keras.layers.Dense(1, activation='sigmoid', name='fc')])  # 全连接层

    # 使用构建的模型进行训练
    model.compile(optimizer ='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    history = model.fit(train_imgs, train_labs, epochs=40, batch_size=50)
    preds = model.evaluate(test_imgs, test_labs, batch_size=32, verbose=2)

    model.save('./data/model/Smile_recognize.h5')
    print(f"准确率：{preds[1]}\n 误差值：{preds[0]}")
    plt.plot(history.history['loss'])
    plt.plot(history.history['accuracy'])
    plt.show()