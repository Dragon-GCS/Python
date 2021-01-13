# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/8 23:29
# Edit with PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import h5py
from tensorflow import keras

def load_dataset():
    train_dataset = h5py.File('data/datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) / 255
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File('data/datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) / 255
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    classes = np.array(test_dataset["list_classes"][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, classes = load_dataset()
    print(x_train.shape)
    model = keras.models.Sequential([keras.layers.Conv2D(8,(3,3),activation='relu',input_shape=(64,64,3)),
                              keras.layers.MaxPooling2D((2,2)),
                              keras.layers.Conv2D(16,(3,3),activation='relu'),
                              keras.layers.MaxPooling2D((2,2)),
                              keras.layers.Flatten(),
                              keras.layers.Dense(6,kernel_regularizer=keras.regularizers.l2(0.0001))])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    shuffle = np.random.permutation(1080)
    x_train = x_train[shuffle,:,:,:]
    y_train = y_train[shuffle]
    history = model.fit(x_train[:1000],y_train[:1000],epochs=50,validation_data=(x_train[1000:],y_train[1000:]))
    model.evaluate(x_test,y_test)