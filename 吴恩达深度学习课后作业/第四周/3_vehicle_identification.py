# coding = utf-8
# Dragon's Python3.8 code
# Created at 2021/1/12 20:57
# Edit with PyCharm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
# 防止显存不足
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)

import matplotlib.pyplot as plt
import tensorflow as tf
import yolo_utils
from tensorflow import keras


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
    """
    通过阈值来过滤对象和分类的置信度
    此例中模型的output维度为（19，19，5，85）19x19为图片分割的数量，5表示锚框数量，85为每个锚框中的信息
    Args:
        box_confidence: tensor类型，维度(19,19,5,1)，包含了19x19个格中每个格预测的5个锚框的5个P_c(概率）
                        P_c为锚框中包含某个物体的概率，即第一个数据[pc,bx,by,bw,bh,c1,c2....c80]
        boxes: tensor类型，维度(19,19,5,4)，包含了所有锚框的(p_x, p_y, p_h, p_w)
        box_class_probs: tensor类型，维度(19,19,5,80)，包含了19x19个格中5个锚框检测的对象(c1,c1...c80)的类别概率
        threshold: 阈值，预测概率高于阈值的会被保留
    Returns:
        scores: tensor类型，维度（None，），包含了所有格中保留的锚框的分类概率
        boxes: tensor类型，维度（None，4），包含了所有格中保留的锚框的(p_x, p_y, p_h, p_w)
        classes: tensor类型，维度（None，），包含了所有格中保留的锚框的索引
        注：None是由于无法确定数量，取决于阈值之上有多少个铆框
    """
    # 计算锚框的得分   锚框中存在某类的概率 * 锚框各个分类的概率
    boxes_scores = box_confidence * box_class_probs         # 输出维度(19,19,5,80)
    # 找到最大值的铆框的索引以及对应的最大的分数
    box_classes = tf.argmax(boxes_scores,axis=-1)           # 最后一个维度80个分类中最大值的索引(19,19,5)
    box_class_score = tf.reduce_max(boxes_scores,axis=-1)   # 最后一个维度80个分类中的最大值(19,19,5)
    # 根据阈值创建掩码
    mask = (box_class_score >= threshold)                   # 为了选择五个框中最大值高于阈值的框
    # 对score， boxes， classes使用掩码
    scores = tf.boolean_mask(box_class_score, mask)         # 仅保留的高于阈值的分数
    boxes = tf.boolean_mask(boxes, mask)                    # 分数高于阈值的框的位置
    classes = tf.boolean_mask(box_classes, mask)            # 分数高于阈值的框的分类的索引

    return scores, boxes, classes

def iou(box1, box2):
    """
    计算两个盒子的交并比，定义区域内左上为（0，0），右下为（1，1）
    Args:
        box1: 第一个盒子的位置信息，元组（x1, y1, x2, y2）左上角坐标和右下角坐标
        box2: 第二个盒子的位置信息，元组（x1, y1, x2, y2）左上角坐标和右下角坐标
    Returns:
        iou：交并比
    """
    # 计算交集面积
    xi1 = tf.maximum(box1[0],box2[0])
    yi1 = tf.maximum(box1[1],box2[1]) # (xi1,yi1) 交集的左上角坐标
    xi2 = tf.minimum(box1[2],box2[2])
    yi2 = tf.minimum(box1[3],box2[3]) # (xi2,yi2) 交集的右下角坐标
    # 判定两个box是否存在交集
    inter_width = max((xi2-xi1), 0)
    inter_height = max((yi2-yi1), 0)
    inter_area = inter_width * inter_height
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_are = box1_area + box2_area - inter_area

    return inter_area / union_are

def yolo_non_max_suppression(scores,boxes,classes,max_boxes=10,iou_threshold=0.5):
    """
    使用tf中tf.image.non_max_suppression()实现非最大值抑制
    Args:
        scores: tensor类型，维度为(None,)，yolo_filter_boxes()的输出
        boxes: tensor类型，维度为(None,4)，yolo_filter_boxes()的输出，已缩放到图像大小
        classes: tensor类型，维度为(None,)，yolo_filter_boxes()的输出
        max_boxes: 整数，预测的锚框数量的最大值
        iou_threshold: 实数，交并比阈值。
    Returns:
        scores: tensor类型，维度为(,None)，每个锚框的预测的可能值
        boxes: tensor类型，维度为(4,None)，预测的锚框的坐标
        classes: tensor类型，维度为(,None)，每个锚框的预测的分类
    """
    # 原文中此处初始化了max_boxes_tensor，tf2版本好像不需要这一步骤了，故删去
    # 使用使用tf.image.non_max_suppression()来获取与保留的框相对应的索引列表
    num_indeces = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold)    # 返回的是是个分数最高的十个边框的索引值
    # 使用keras.backend.gather()根据索引选择对应的分数、位置和分类
    scores = keras.backend.gather(scores, num_indeces)
    boxes = keras.backend.gather(boxes, num_indeces)
    classes = keras.backend.gather(classes, num_indeces)

    return scores, boxes, classes

def yolo_eval(yolo_output, image_shape=(720.,1280.),
              max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    将yolo的输出（多个锚框）转换为预测框以及他们的分数，框坐标和类
    Args:
        yolo_output: 模型的输出。对（608，608，3）的图片包含四个Tensor向量
                        box_confidence：tensor，维度（None，19，19，5，1）
                        box_xy：tensor，维度（None，19，19，5，2）
                        box_hw：tensor，维度（None，19，19，5，2）
                        box_class_probs：tensor，维度（None，19，19，5，80）
        image_shape: tensor，表示图像的维度，这里是（608，608）
        max_boxes: 预测的锚框数量的最大值
        score_threshold: 可能性阈值
        iou_threshold: 交并比阈值
    Returns:
        scores: tensor，维度（，None）每个锚框预测的可能值
        boxes: tensor，维度（4，None）每个预测锚框的坐标
        classes: tensor，维度（，None）每个锚框预测的分类
    """
    box_confidence, box_xy, box_wh, box_class_prob = yolo_output
    # 中心点坐标转换为边角坐标
    boxes = yolo_utils.yolo_boxes_to_corners(box_xy, box_wh)
    # 过滤掉低于score_threshold的框
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_prob, score_threshold)
    # 缩放锚框，以适应原始图像
    boxes = yolo_utils.scale_boxes(boxes, image_shape)
    # 使用非最大值抑制
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes)

    return scores, boxes, classes

def predict(model, image_file, is_show_info=True, is_plot=True):
    """
    运行model并输出预测的图和信息
    由于练习的使用的时tf2.3版本，代码和博客中的tf1有很大区别，所以和部分想了半天，也算是从头梳理了一遍
    注意：博客中没有model这个参数，这里加上是因为模型在函数里加载的话，运行批量绘图时会由于多次加载模型导致显存报错。
    Args:
        model: 用来预测锚框的模型
        image_file: images文件夹的图片名称
    Returns:
        out_scores: tensor，维度为（None，）锚框预测的可能值
        out_boxes: tensor，维度为（None，4）预测的锚框的位置
        out_classes: tensor， 维度为（None，）预测的锚框的分类索引
    """
    dir = './data/Yolo/yolo_model/model_data/'
    class_name = yolo_utils.read_classes(dir+'coco_classes.txt')
    anchors = yolo_utils.read_anchors(dir+'yolo_anchors.txt')
    image_shape = (720., 1280.)
    
    # 处理图像，image_data为图像转换为tensor后的数据
    image, image_data = yolo_utils.preprocess_image('./data/Yolo/images/'+image_file,model_image_size=(608,608))
    # 预测图像，结果为（1，19，19，425）最后的维度为5个锚框x85个属性
    yolo_model_output = model.predict(image_data)

    # yolo_head将yolo模型的输出进行转换为各个格子中每个锚框的 （坐标、宽高、预测值、分类值）
    # 原文中yolo_head的输出顺序有误，会导致yolo_eval函数报错，在此已经将yolo_head的输出顺序修改
    yolo_outputs = yolo_utils.yolo_head(yolo_model_output, anchors, len(class_name))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

    # 打印预测信息
    if is_show_info:
        print("在" + str(image_file) + "中找到了" + str(len(boxes)) + "个锚框。")

    # 指定绘制边框的颜色
    colors = yolo_utils.generate_colors(class_name)
    # 绘制边界并保存图片
    yolo_utils.draw_boxes(image, scores, boxes, classes, class_name, colors)
    image.save('./data/Yolo/out/'+image_file, quality=100)
    # 打印出已经绘制了边界框的图
    if is_plot:
        output_image = plt.imread('./data/Yolo/out/' + image_file)
        plt.imshow(output_image)
        plt.show()

    return scores, boxes, classes

def pilianghuitu(model):
    """
    哈哈哈，用tf2写tf1太难了，终于弄完了，累了，这里就名字就随便起起了
    """
    for i in range(76,121):
        filename = str(i).zfill(4) + '.jpg'
        print("当前文件：" + str(filename))
        predict(model, filename)


if __name__ == '__main__':
    # 这里把加载模型这一步写在函数外面了，如果卸载predict函数里面，在运行批量绘图的时候会提示显存空间不足
    yolo_model = keras.models.load_model('./data/Yolo/yolo_model/model_data/yolo.h5')
    pilianghuitu(yolo_model)
