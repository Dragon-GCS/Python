# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/21 19:26
# Edit with PyCharm


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow.keras import layers, Model, backend, utils, optimizers
import numpy as np
import tensorflow as tf
import qa
from music21 import stream, note, tempo, midi



def load_music_utils():
    chords, abstract_grammars = qa.get_musical_data('data/original_metheny.mid')
    corpus, tones, tones_indices, indices_tones = qa.get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = qa.data_processing(corpus, tones_indices, 60, 30)
    return (X, Y, N_tones, indices_tones)


def djmodel(Tx, n_a, n_values):
    """
    音乐生成模型
    Args:
        T_x: 语料库的长度
        n_a: 激活值的数量
        n_values: 音乐数据中唯一数据的数量
    Returns:
        model: 构筑后的模型
    """
    X = layers.Input(shape=(Tx, n_values))
    # 定义a0，初始化隐藏状态
    a0 = layers.Input(shape=(n_a,), name='a0')
    c0 = layers.Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    # 创建空列表保存LSTM所有时间步的输出
    output = []
    for t in range(Tx):
        # 从X中选择第t个时间步
        x = layers.Lambda(lambda x:X[:,t,:])(X)
        # 使用reshapor将x重构为（1，n_values)
        x = reshapor(x)
        # 单步传播
        a, _, c = LSTM_cell(x, initial_state=[a,c])
        out = densor(a)

        output.append(out)

    return Model(inputs=[X, a0, c0], outputs=output)


def one_hot(x):
    x = backend.argmax(x)
    x = tf.one_hot(x, 90)
    x = layers.RepeatVector(1)(x)
    return x


def music_inference_model(LSTM_cell, densor, n_values=90, n_a=64, Ty=100):
    """
    Args:
        LSTM_cell: model训练后的LSTM层
        densor: model训练后的dense层
        n_values: 唯一值的数量
        n_a: LSTM层的节点数
        Ty: 时间鞥的时间步的数量
    Returns:
        inference_model: 生成的模型
    """
    x0 = layers.Input((1,n_values))
    a0 = layers.Input((n_a,))
    c0 = layers.Input((n_a,))
    a = a0
    c = c0
    x = x0

    output = []
    for t in range(Ty):
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        output.append(out)
        x = layers.Lambda(one_hot)(out)

    return Model(inputs=[x0,a0,c0], outputs = output)


def predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer):
    """
    使用模型预测当前值的下一个值
    Args:
        inference_model: 模型
        x_initializer: 初始值x
        a_initializer: 初始值隐藏状态
        c_initializer: 初始值cell状态
    Returns:
        results: 生成值的one-hot向量，维度（Ty, n_values)
        indices: 生成值的索引矩阵（Ty, 1）
    """
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis=-1)
    results = utils.to_categorical(indices, num_classes=n_values)

    return results, indices


chords, abstract_grammars = qa.get_musical_data('data/original_metheny.mid')
corpus, tones, tones_indices, indices_tones = qa.get_corpus_data(abstract_grammars)
N_tones = len(set(corpus))
def generate_music(inference_model, corpus=corpus, abstract_grammars=abstract_grammars, tones=tones,
                   tones_indices=tones_indices, indices_tones=indices_tones, T_y=10, max_tries=1000, diversity=0.5):
    """
    使用训练的模型生成音乐
    Arguments:
    model -- 训练的模型
    corpus -- 音乐语料库, 193个音调作为字符串的列表(ex: 'C,0.333,<P1,d-5>')
    abstract_grammars -- grammars列表: 'S,0.250,<m2,P-4> C,0.250,<P4,m-2> A,0.250,<P4,m-2>'
    tones -- set of unique tones, ex: 'A,0.250,<M2,d-4>' is one element of the set.
    tones_indices -- a python dictionary mapping unique tone (ex: A,0.250,< m2,P-4 >) into their corresponding indices (0-77)
    indices_tones -- a python dictionary mapping indices (0-77) into their corresponding unique tone (ex: A,0.250,< m2,P-4 >)
    Tx -- integer, number of time-steps used at training time
    temperature -- scalar value, defines how conservative/creative the model is when generating music
    Returns:
    predicted_tones -- python list containing predicted tones
    """

    # set up audio stream
    out_stream = stream.Stream()

    # Initialize chord variables
    curr_offset = 0.0  # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)  # number of different set of chords

    print("Predicting new values for different set of chords.")
    # Loop over all 18 set of chords. At each iteration generate a sequence of tones
    # and use the current chords to convert it into actual sounds
    for i in range(1, num_chords):

        # Retrieve current chord from stream
        curr_chords = stream.Voice()

        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate offset, no need to understand this
            curr_chords.insert((j.offset % 4), j)

        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]

        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' '

        predicted_tones += pred[-1]

        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A', ' C').replace(' X', ' C')

        # Pruning #1: smoothing measure
        predicted_tones = qa.prune_grammar(predicted_tones)

        # Use predicted tones and current chords to generate sounds
        sounds = qa.unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = qa.prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = qa.clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (
        len([k for k in sounds if isinstance(k, note.Note)]), i))

        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0

    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()

    return out_stream

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    X, Y, n_values, indices_values = load_music_utils()

    n_a = 64
    reshapor = layers.Reshape((1, n_values))
    LSTM_cell = layers.LSTM(n_a, return_state=True)
    densor = layers.Dense(n_values, activation='softmax')

    model = djmodel(Tx=30, n_a=n_a, n_values=n_values)
    opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    # 初始化a0和c0，使LSTM的初始状态为零。
    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))

    model.fit([X, a0, c0], list(Y), epochs=100)
    # 获取模型实体，模型被硬编码以产生50个值
    inference_model = music_inference_model(LSTM_cell, densor, n_values=n_values, n_a=n_a, Ty=50)
    x_initializer = np.zeros((1, 1, n_values))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))

    # results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

    out_stream = generate_music(inference_model)