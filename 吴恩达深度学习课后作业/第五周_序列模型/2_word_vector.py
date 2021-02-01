# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/25 21:24
# Edit with PyCharm

import numpy as np


with open('./data/glove.6B.50d.txt', 'r', encoding='utf8') as f:
    words = set()
    word2vec_map = {}

    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        words.add(curr_word)
        word2vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

def cosine_similarity(u, v):
    """
    计算余弦相似度 点积/长度积
    Args:
        u: 词向量
        v: 词向量
    Returns:
        cosine_similarity: 余弦相似度，两个向量约相似，相似度约接近1
    """
    distance = 0
    # 点积
    dot = u.dot(v)
    # u的长度
    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    # v的长度
    norm_v = np.sqrt(np.sum(np.power(v, 2)))

    return np.divide(dot, (norm_u * norm_v))


def complete_analogy(worda, wordb, wordc, word2vec_map):
    """
    词类类比，a与b相比和c与?相比一样
    Args:
        worda: 单词a
        wordb: 单词b
        wordc: 单词c
        word2vec_map: 单词到GloVe向量的映射字典
    Returns:
        best_word: 最符合条件的词
    """
    worda, wordb, wordc = worda.lower(), wordb.lower(), wordc.lower()
    # 获取对应的词向量
    ea, eb, ec = word2vec_map[worda], word2vec_map[wordb], word2vec_map[wordc]
    # 获取全部单词
    words = word2vec_map.keys()

    max_cos_simi = -100
    best_word = None

    for word in word2vec_map.keys():
        if word in [worda, wordb, wordc]:
            continue
        cos_simi = cosine_similarity((ea-eb), (ec-word2vec_map[word]))

        if cos_simi > max_cos_simi:
            max_cos_simi = cos_simi
            best_word = word

    return best_word


def neutralize(word, g, word2vec_map):
    """
    将word投影到与bias轴正交（垂直）的空间上，消除word偏差
    Args:
        word: 带消除偏差的word
        g: 维度为（50，），对应于bias轴，如性别轴：e_man-e_woman
        word2vec_map: 单词到GloVe向量的映射
    Returns:
        e_disbiased: 消除bias后的向量
    """
    e = word2vec_map[word]
    # e在g上的投影 e_biscomponent即e在g方向的bias
    #   e·g           /|g|^2 x g
    # = |e|x|g|x sinθ/|g|^2 x g
    # = |e|xsinθ x (g/|g|) :  |e|xsinθ是e在g方向上投影的长度，g/|g|表示g的方向
    e_biscomponent = np.divide(np.dot(e, g), np.square(np.linalg.norm(g))) * g
    # e减去其在g方向的投影即为消除其在g方向的bias
    return e - e_biscomponent


def equalize(pair,bias_axis, word2vec_map):
    """
    使用均衡方法消除bias
    Args:
        pair: 需要消除bias的单词对
        bias_axis: bias轴
        word2vec_map: 单词到GloVe向量的映射
    Returns:
        e1: 第一个词消除bias后的向量
        e2: 第二个词消除bias后的向量
    """
    w1, w2 = pair
    e_w1, e_w2 = word2vec_map[w1], word2vec_map[w2]
    # 计算两个单词的均值
    mu = (e_w1 + e_w2) / 2.
    # 计算mu在bias轴上的投影与消除bias后的值
    mu_bia = np.divide(mu.dot(bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    mu_dbia = mu - mu_bia
    # 计算w1, w2在bias上的投影
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    # 计算w1, w2的偏置部分
    corrected_e_w1B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_dbia))))\
                      * np.divide(e_w1B - mu_bia, np.abs(e_w1 - mu_dbia - mu_bia))

    corrected_e_w2B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_dbia))))\
                      * np.divide(e_w2B - mu_bia, np.abs(e_w2 - mu_dbia - mu_bia))

    return corrected_e_w1B+mu_dbia, corrected_e_w2B+mu_dbia


#------------------------------测试部分1------------------------------#
# father = word2vec_map["father"]
# mother = word2vec_map["mother"]
# ball = word2vec_map["ball"]
# crocodile = word2vec_map["crocodile"]
# france = word2vec_map["france"]
# italy = word2vec_map["italy"]
# paris = word2vec_map["paris"]
# rome = word2vec_map["rome"]
#
# print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
# print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
# print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
#------------------------------测试部分2------------------------------#
# triads_to_try = [('italy', 'italian', 'spain'),
#                  ('india', 'delhi', 'japan'),
#                  ('man', 'woman', 'boy'),
#                  ('small', 'smaller', 'large')]
# for triad in triads_to_try:
#     print ('{} -> {} <====> {} -> {}'.format( *triad, complete_analogy(*triad,word2vec_map)))
#------------------------------测试部分3------------------------------#
# g = word2vec_map['woman'] - word2vec_map['man']
#
# e = "receptionist"
# print("去偏差前{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(word2vec_map["receptionist"], g)))
#
# e_debiased = neutralize("receptionist", g, word2vec_map)
# print("去偏差后{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(e_debiased, g)))
#------------------------------测试部分4------------------------------#
# print("==========均衡校正前==========")
# print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word2vec_map["man"], g))
# print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word2vec_map["woman"], g))
# e1, e2 = equalize(("man", "woman"), g, word2vec_map)
# print("\n==========均衡校正后==========")
# print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
# print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
