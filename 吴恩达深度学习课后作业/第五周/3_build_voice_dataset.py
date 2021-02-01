# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/2/1 21:17
# Edit with PyCharm

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


from pydub import AudioSegment
from scipy.io import wavfile
import random
import sys
import io
import os
import glob
import IPython
#------------------------------utils start------------------------------#
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("data/Trigger word detection/raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./data/Trigger word detection/raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("data/Trigger word detection/raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./data/Trigger word detection/raw_data/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir("data/Trigger word detection/raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./data/Trigger word detection/raw_data/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds


def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data


def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx
#-------------------------------utils end-------------------------------#
X = np.load('./data/XY_train/X.npy')
Y = np.load('./data/XY_train/Y.npy')
X_dev = np.load('./data/XY_dev/X_dev.npy')
Y_dev = np.load('./data/XY_dev/Y_dev.npy')


def model(input_shape):
    input = layers.Input(input_shape)
    x = layers.Conv1D(196, 15, strides=4)(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.8)(x)

    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.Dropout(0.8)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.Dropout(0.8)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.8)(x)
    # 时间分布全连接层
    x = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(x)

    return models.Model(input, x)


Tx, n_freq, Ty = 5511, 101, 1375
model = model((Tx, n_freq))
# model.summary()
# model.compile(Adam(1e-4), 'binary_crossentropy', ["accuracy"])
# model.fit(X, Y, epochs=800, validation_data=(X_dev,Y_dev))
# model.save('./models/tr_model.h5')
model = models.load_model('./models/tr_model.h5')
#model.evaluate(X_dev, Y_dev)


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # 频谱图输出（freqs，Tx），我们想要（Tx，freqs）输入到模型中
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions


chime_file = "./data/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # 第一步：将连续输出步初始化为0
    consecutive_timesteps = 0
    # 第二步： 循环y中的输出步
    for i in range(Ty):
        # 第三步： 增加连续输出步
        consecutive_timesteps += 1
        # 第四步： 如果预测高于阈值并且已经过了超过75个连续输出步
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            # 第五步：使用pydub叠加音频和背景
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            # 第六步： 将连续输出步重置为0
            consecutive_timesteps = 0

    audio_clip.export("./data/chime_output.wav", format='wav')


filename = "./data/Trigger word detection/raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)















