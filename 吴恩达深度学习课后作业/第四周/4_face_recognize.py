# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/15 20:44
# Edit with PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)

import cv2
import fr_utils as u

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = tf.transpose(img, (2,0,1))/255    # 此处img类型为tf.int8，如果除255.0会因为类型不同报错
    x_train = tf.convert_to_tensor(img)[tf.newaxis,...]     # 使用tf.newaxis为数组增加一个维度
    embedding = model.predict_on_batch(x_train)
    return embedding


def creat_database():
    """
    创建面部识别的数据库
    Returns:
        database: 包含姓名和对应面部编码的字典
    """
    database = {}
    database["danielle"] = img_to_encoding("./data/face_recognize/images/danielle.png", FRmodel)
    database["younes"] = img_to_encoding("./data/face_recognize/images/younes.jpg", FRmodel)
    database["tian"] = img_to_encoding("./data/face_recognize/images/tian.jpg", FRmodel)
    database["andrew"] = img_to_encoding("./data/face_recognize/images/andrew.jpg", FRmodel)
    database["kian"] = img_to_encoding("./data/face_recognize/images/kian.jpg", FRmodel)
    database["dan"] = img_to_encoding("./data/face_recognize/images/dan.jpg", FRmodel)
    database["sebastiano"] = img_to_encoding("./data/face_recognize/images/sebastiano.jpg", FRmodel)
    database["bertrand"] = img_to_encoding("./data/face_recognize/images/bertrand.jpg", FRmodel)
    database["kevin"] = img_to_encoding("./data/face_recognize/images/kevin.jpg", FRmodel)
    database["felix"] = img_to_encoding("./data/face_recognize/images/felix.jpg", FRmodel)
    database["benoit"] = img_to_encoding("./data/face_recognize/images/benoit.jpg", FRmodel)
    database["arnaud"] = img_to_encoding("./data/face_recognize/images/arnaud.jpg", FRmodel)

    return database


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    实现课程中讲到的三元组损失函数
    Args:
        y_true: 训练模型的true标签, 这里不需要
        y_pred: 列表类型，包括：anchor、positive、negative三个图形在最后一层的激活数值，维度为（ ，128）
        alpha: 超参数，阈值
    Returns:
        loss: 实数，损失值
    """
    anchor, pos, neg = y_pred[0], y_pred[1], y_pred[2]
    # 计算anchor和positive之间的差距, 注意这里需要计算最后一个维度（全连接层激活值）
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1)
    # 计算anchor和negative之间的差距, 注意这里需要计算最后一个维度（全连接层激活值）
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1)
    # 两个距离相减后加上阈值
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # 如果导入了tf，上述所有加减函数可以之间使用+、-代替。tf对运算符号进行底层编译
    return tf.reduce_sum(tf.maximum(basic_loss, 0))


def verify(image_path, identity, database, model):
    """
    人脸认证，比较摄像头图像与id信息是否符合，即比较库中identity的编码和image_path的编码（即全连接层的输出）
    Args:
        image_path: 摄像头的图片
        identity: 字符串，想要验证的人的名字
        database: 字典， 包含了成员姓名和对应编码
        model: 训练好的模型
    Returns:
        dist: 摄像头中图片与数据库中图片的编码差距
        is_open_door： True/False 是否开门
    """
    # 计算图像的编码
    # 计算图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = img_to_encoding(image_path, model)
    # 计算与数据库中保存的编码的差距，这里使用tf的范数计算函数替换原文的np.linalg.norm
    dist = tf.norm(encoding - database[identity])
    # 判断是否打开门
    if dist < 0.7:
        print("欢迎 " + str(identity) + "回家！")
        is_door_open = True
    else:
        print("经验证，您与" + str(identity) + "不符！")
        is_door_open = False

    return dist, is_door_open


def who_is_it(image_path, database, model):
    """
    根据指定的图片来进行人脸识别
    Args:
        image_path: 图片地址
        database: 包含了名字与比那吗的字典
        model: 训练好的图形
    Returns:
        min_dist: 字典中与输入图像最相近的编码
        identity: 与min_dist对应的名字
    """
    # 计算指定图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = img_to_encoding(image_path, model)
    # 找到最相近的编码
    min_dist = 100                  # 初始化min_dist变量为足够大的数字，这里设置为100
    # 遍历数据库，找到min_dist
    for name, db_enc in database.items():
        dist = tf.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    # 判断输入图像是否在数据库中
    if min_dist > 0.7:
        print("抱歉，您的信息不在数据库中。")
    else:
        print("姓名" + identity + "  差距：" + str(min_dist))

    return min_dist, name


if __name__ == '__main__':
    # 运行前需要将.keras/keras.json的"image_data_format"改为: "channels_first"
    FRmodel = u.faceRecoModel(input_shape=(3, 96, 96))
    # 编译模型
    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    # 加载权值
    u.load_weights_from_FaceNet(FRmodel)
    # 加载数据库
    database = creat_database()

    who_is_it("./data/face_recognize/images/camera_0.jpg", database, FRmodel)