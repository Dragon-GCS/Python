# 吴恩达深度学习课后作业

学习吴恩达Deep Learning系列视频后做的编程作业，大佬的源地址为[吴恩达课后作业目录-CSDN](https://blog.csdn.net/u013733326/article/details/79827273)

## 第一周	[深度学习和神经网络](https://github.com/Dragon-GCS/Python/tree/master/%E5%90%B4%E6%81%A9%E8%BE%BE%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%AF%BE%E5%90%8E%E4%BD%9C%E4%B8%9A/%E7%AC%AC%E4%B8%80%E5%91%A8)

1. 逻辑回归的代码实现

2. 单隐层的神经网络

3. 多隐层的神经网络

## 第二周	[神经网络优化方法](https://github.com/Dragon-GCS/Python/tree/master/%E5%90%B4%E6%81%A9%E8%BE%BE%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%AF%BE%E5%90%8E%E4%BD%9C%E4%B8%9A/%E7%AC%AC%E4%BA%8C%E5%91%A8)

1. 正则化神经网络

2. 梯度优化算法

3. TensorFlow入门

   这个作业CSDN版本用的是TF1.0API，自己是用的TF2.0的API

## 第四周 [卷积神经网](https://github.com/Dragon-GCS/Python/tree/master/%E5%90%B4%E6%81%A9%E8%BE%BE%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%AF%BE%E5%90%8E%E4%BD%9C%E4%B8%9A/%E7%AC%AC%E5%9B%9B%E5%91%A8)

1. 搭建卷积神经网络

2. keras入门与搭建残差网络

   博客里使用的keras，我在写作业的时候均使用tf.keras代替
   
3. 车辆识别

   由于博客中还是使用的tf构建静态图的方法，这里我用tf2的动态图写的，同时修改了yolo_utils里的部分代码。
   Yolo文件夹太大，内容请到网盘自取[提取码：sg4u](https://pan.baidu.com/s/1Sf5pQQGeGyZbmunSxDH5gA ),其中权重文件和cfg文件过大，可以去[这里](https://cloud.tencent.com/developer/article/1436586)参考下载。
   最后的测试部分由于我的显卡不太给力，连续识别19张图显存就不够了，有能力自己改一下批量绘图的部分。
   补：显存报错的原因找到了：由于代码里的predict函数每次运行都要load_model()，只要把模型在函数外加载完在传给函数就可以了。
   
4. 人脸识别与神经网络风格迁移
   
   *  博客中神经网络风格迁移这一部分代码补全，可以参考这篇[博客](https://blog.csdn.net/little_orange__/article/details/106366235)

   * 人脸识别中的权重数据需要将data/face_recognize/中的weight解压使用

   * 神经网络风格迁移预训练的模型过大，请在网盘下载[提取码jou1](https://pan.baidu.com/s/1NKt3BYvzUHeWIGA5xPnKJQ )。这一部分在tf2中没找到对应的接口，所以使用的是tf.compat.v1调用的tf1.0的api

## 第五周 [序列模型](https://github.com/Dragon-GCS/Python/tree/master/%E5%90%B4%E6%81%A9%E8%BE%BE%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%AF%BE%E5%90%8E%E4%BD%9C%E4%B8%9A/%E7%AC%AC%E4%BA%94%E5%91%A8)

​	相关文件下载请[点击这里](https://blog.csdn.net/u013733326/article/details/79827273)

1. 搭建循环神经网络及其应用

2. 词向量的运算与Emoji生成器

3. 机器翻译与触发词检测

   * 机器翻译这部分使用作者给的模型预测结果全是2，不知道哪里出了问题，所以用的是自己训练的模型

     ```lr=5e-3, epoches=50,batch_size=128```

   * 触发词检测用博客中的模型会报错，这里用的是作者给的训练数据自己训练的模型，准确率92%
   
   * 触发词这里作业大部分是说数据处理的，所以这一部分没有太仔细看

序列模型这一部分还是比较难的，作业基本上按照博客的来就可以，不会遇到什么太大问题，个人认为这一部分比较难的就是数据的预处理和RNN层的数据输出问题

