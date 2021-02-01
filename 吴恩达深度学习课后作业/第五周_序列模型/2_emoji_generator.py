# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/25 22:18
# Edit with PyCharm

import numpy as np
import csv
from tensorflow import keras


def read_csv(filename):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


X_train, Y_train = read_csv('./data/train_emoji.csv')
X_test, Y_test = read_csv('./data/test.csv')
# 根据len找到最长的句子，分割为单词列表后返回其长度
maxLen = len(max(X_train, key=len).split())

Y_oh_train = np.eye(5)[Y_train.reshape(-1)]
Y_oh_test = np.eye(5)[Y_test.reshape(-1)]

word2index, index2word, word2vec_map = read_glove_vecs('./data/glove.6B.50d.txt')


def sentence2avg(sentence,word2vec_map):
    """
    将句子转换为单词列表，提取GloVe向量，并取平均
    Args:
        sentence: 句子
        word2vec_map: 单词到GloVe向量的映射
    Returns:
        avg: 取平均后的矩阵
    """
    words = sentence.lower().split()
    avg = np.zeros(50,)
    for word in words:
        avg += word2vec_map[word]

    return avg / len(words)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict(X, Y, w, b, word2vec_map):
    """
    使用model的参数进行预测
    Args:
        X: 输入数据
        Y: 数据对应标签
        w: 权重矩阵
        b: 偏置矩阵
        word2vec_map: 单词到向量的映射
    Returns:
        pred: 预测结果
    """
    m = X.shape[0]
    pred = np.zeros((m, 1))

    for j in range(m):  # 遍历样本

        avg = sentence2avg(X[j], word2vec_map)
        Z = np.dot(w, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)

    print("Accuracy: " + str(np.mean((pred[:] == Y.reshape(Y.shape[0], 1)[:]))))

    return pred


def model(X, Y, word2vec_map, lr=1e-2, iteration=400):
    """
    使用numpy训练模型
    Args:
        X: 输入的字符串数据，维度（m，1）
        Y: 对应的标签，范围0-4，维度*（m，1）
        word2vec_map: 单词到50维词向量GloVe的映射
        lr: 学习率
        iteration: 迭代次数
    Returns:
        pred: 预测的向量，维度（m，1）
        w: 权重参数（n_y, n_h）
        b: 偏置参数（n_y, ）
    """
    np.random.seed(1)
    m = Y.shape[0]
    n_y = 5
    n_h = 50
    w = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y))
    # 将标签转换为onehot向量
    Y_oh = np.eye(n_y)[Y.reshape(-1)]

    for t in range(iteration):
        # 遍历样本
        for i in range(m):
            avg = sentence2avg(X[i], word2vec_map)
            # 前向传播
            z = w.dot(avg) + b
            a = softmax(z)
            # 计算成本
            cost = -np.sum(Y_oh[i] * np.log(a))
            # 梯度计算
            dz = a - Y_oh[i]
            dw = dz.reshape(n_y,1).dot(avg.reshape(1,n_h))
            db = dz
            # 更新参数
            w -= dw * lr
            b -= db * lr

        if t % 100 == 0:
            print(f'第{t}轮，损失为{cost}')
            prediction = predict(X, Y, w, b, word2vec_map)

    return prediction, w, b

#------------------------------keras构建模型------------------------------#
def sentence2indices(X, word2index, max_len):
    """
    将输入转换为embeding层能够接受的列表或矩阵
    Args:
        X: 样本列表，维度（m，1）
        word2index: 单词到索引的映射
        max_len: 最大句子的长度
    Returns:
        X_indices: X中每个单词对应的索引的数据（m, max_len）
    """
    m = X.shape[0]
    X_indices = np.zeros((m,max_len))
    for i in range(m):
        words = X[i].lower().split()
        j = 0
        for word in words:
            X_indices[i, j] = word2index[word]
            j += 1

    return X_indices


def pretrained_embedding_layer(word2vec_map, word2index):
    """
    将训练好的50维GloVe向量加载到keras的embedding层中
    Args:
        word2vec_map: 单词到GloVe向量的映射
        word2index: 单词到索引的映射，共400 000个单词
    Returns:
        embedding_layer: 训练好的embedding层
    """
    vocab_len = len(word2index) + 1             # 400,000个单词+1（防止out of index）
    emb_dim = word2vec_map['cucumber'].shape[0] # GloVe词向量的维度，50

    emb_matrix = np.zeros((vocab_len, emb_dim)) # (400,001, 50）
    # 将嵌入矩阵的每行index转换为对应的词向量表示
    for word, index in word2index.items():      # index从1开始，所以不更新emb_matrix[0,:] = [0,....0,]
        emb_matrix[index, :] = word2vec_map[word]
    # 定义keras的embedding层
    layer = keras.layers.Embedding(vocab_len, emb_dim, trainable=False)
    layer.build((None,))
    layer.set_weights([emb_matrix])

    return layer


def emoji_V2(input_shape, word2vec_map, word2index):
    """
    实现emojiV2模型
    Args:
        input_shape: 数据输入维度，通常为（maxlen，）
        word2vec_map: 单词到向量的映射
        word2index: 单词到索引的映射
    Returns:
        model: 构建的模型
    """
    sentence_index = keras.Input(input_shape, dtype='int32')
    embedding = pretrained_embedding_layer(word2vec_map, word2index)(sentence_index)

    x = keras.layers.LSTM(128, return_sequences=True)(embedding)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.LSTM(128, return_sequences=False)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(5, activation='softmax')(x)

    return keras.models.Model(inputs=sentence_index, outputs=x)


# 创建模型并进行预测
model = emoji_V2((maxLen,), word2vec_map, word2index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indice = sentence2indices(X_train, word2index, maxLen)
model.fit(X_train_indice, Y_oh_train, epochs=50, batch_size=32, shuffle=True)

# 评估准确率
X_test_indice = sentence2indices(X_test, word2index, maxLen)
loss, acc = model.evaluate(X_test_indice, Y_oh_test)
print("Test accuracy = ", acc)

#-----------------------------检查预测错误的元素-----------------------------#
import emoji
emoji_dictionary = {"0": "\u2764\uFE0F",
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}
def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

C = 5
pred = model.predict(X_test_indice)
for i in range(len(X_test)):
    x = X_test_indice
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print(f'正确表情：{label_to_emoji(Y_test[i])}   预测结果： {X_test[i] + label_to_emoji(num).strip()}')
