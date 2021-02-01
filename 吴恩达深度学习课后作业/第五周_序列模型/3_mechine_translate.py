# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/30 23:06
# Edit with PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import random

import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from faker import Faker
from tqdm import tqdm
from babel.dates import format_date

#-----------------------------------utils start-----------------------------------#
fake = Faker()
Faker.seed(12345)
random.seed(12345)
FORMATS = ['short', 'medium', 'long', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full',
           'd MMM YYY', 'd MMMM YYY', 'dd MMM YYY', 'd MMM, YYY', 'd MMMM, YYY', 'dd, MMM YYY', 'd MM YY',
           'd MMMM YYY', 'MMMM d YYY', 'MMMM d, YYY', 'dd.MM.YY']
# change this if you want it to work with another language
LOCALES = ['en_US']


def load_date():
    dt = fake.date_object()
    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),
                                     locale='en_US')  # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()

    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m):
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30

    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'],
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v: k for k, v in inv_machine.items()}

    return dataset, human, machine, inv_machine


def string_to_int(string, length, vocab):
    string = string.lower()
    string = string.replace(',', '')

    if len(string) > length:
        string = string[:length]

    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))

    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    return rep


def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    X, Y = zip(*dataset)

    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]

    Xoh = np.array(list(map(lambda x: tf.keras.utils.to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: tf.keras.utils.to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh
#------------------------------------utils end------------------------------------#

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
# dataset: [(ori-data, format-data), ......]
Tx, Ty = 30, 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
# X.shape: (10000, 30)
# Y.shape: (10000, 10)
# Xoh.shape: (10000, 30, 37)
# Yoh.shape: (10000, 10, 11)
repeator = layers.RepeatVector(Tx)
concatenator = layers.Concatenate(axis=-1)
densor1 = layers.Dense(10, activation='tanh')
densor2 = layers.Dense(1, activation='relu')
activator = layers.Activation('softmax')
dotor = layers.Dot(axes=1)


def one_step_attention(a, s_prev):
    """
    执行一步attention，输出一个context向量
    Args:
        a: attention前的BiLSTM的输出隐藏状态（m, Tx, 2*n_a)
        s_prev: attention的LSTM层的前一个隐藏状态（m, n_s）
    Returns:
        context: 上下文向量，下一个attention-LSTM层的输入
    """
    # 使用RepeatVector层重复s_prev向量，将其维度从（m, n_s）变为（m, Tx, n_s）用于连接所有隐藏状态a
    s_prev = repeator(s_prev)
    # 将a和s_prev的最后一个维度进行连接，连接后为（m, Tx, n_s + 2*n_a）
    concat = concatenator([a, s_prev])
    # 使用一个小的全连接网络进行计算“中间能量”变量 e
    e = densor1(concat)
    # 使用一个小的全连接神经网络计算能量变量energies
    energies = densor2(e)
    # 使用激活函数softmax传入参数energies，计算注意力权重alphas
    alphas = activator(energies)
    # 使用dot层计算context
    context = dotor([alphas, a])

    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = layers.LSTM(64, return_state = True)
output_layer = layers.Dense(len(machine_vocab), activation='softmax')


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Args:
        Tx: 输入序列
        Ty: 输出序列长度
        n_a: attention前的BiLSTM的隐藏层数
        n_s: attention后的LSTM的隐藏层数
        human_vocab_size: human_vocab的大小
        machine_vocab_size: machine_vocab的大小
    Returns:
        model: 构建好的模型
    """
    X = layers.Input(shape=(Tx, human_vocab_size))
    s0 = layers.Input(shape=(n_s,), name='s0')
    c0 = layers.Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    outputs = []
    # 定义attention前的LSTM层
    a = layers.Bidirectional(layers.LSTM(n_a, return_sequences=True), input_shape=(m,Tx,n_a*2))(X)
    # 迭代Ty步
    for t in range(Ty):
        # 得到第t步的上下文向量
        context = one_step_attention(a, s)
        # 使用attention后的LSTM层得到新的context
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)

    return tf.keras.Model(inputs=[X,s0,c0], outputs=outputs)


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
# model.summary()
opt = tf.keras.optimizers.Adam(5e-3)
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

s0 = tf.zeros((m, n_s))
c0 = tf.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))
#model.fit([Xoh, s0, c0], outputs, epochs=5, batch_size=128)
#model.save('./models/MT_model.h5')
model.load_weights('./models/MT_model.h5')
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
            'March 3rd 2001', '1 March 2001']

for example in EXAMPLES:
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: tf.keras.utils.to_categorical(x, num_classes=len(human_vocab)), source)))
    source = np.expand_dims(source, axis=0)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", example)
    print("output:", ''.join(output))

