# 前言

本项目采用numpy写了一个三层神经网络，用于mnist数据集手写数字的识别。除了获取数据集用了tensorflow提供的API，其余部分均采用numpy来实现。

# 激活函数

激活函数采用sigmoid

# 数据集预处理

## 数据集处理
数据集的原始shape为(size,28,28)，首先reshape为(size,784)。然后由于数据集为灰度图，值的范围为(0,255)，将数据集归一化，每个值减去127.5再除以127.5，将数据集放在(-1,1)之间。

## 数据集标签的处理
标签shape为(size,)，但是由于这是一个多分类问题，所以标签应该为(size,10)(共10类)。每一行代表一张图的分类结果，例如，第一行:(0,1,0,0,0,0,0,0,0,0)，表示第一张图为1。

# 目标函数

目标函数设计了两种：1.距离矢量和 2.softmax

在训练过程中发现，距离矢量和loss可以收敛到很小，但是softmax的loss却总是收敛到一个很大的值

# 训练

epoch为3000，batch为64，训练集采用2000张图片，验证集采用100张图片

# 训练结果

两种目标函数对测试集预测的准确率都能达到88.0%

下面是训练过程中loss和accuracy的变化

## softmax:



<img src="pics\softmax-Train-Loss.png" alt="softmax-Train-Loss" style="zoom:100%;" />

<img src="pics\softmax-Val-Loss.png" alt="softmax-Val-Loss" style="zoom:100%;" />

<img src="pics\softmax-Train-Accuracy.png" alt="softmax-Train-Accuracy" style="zoom:100%;" />

<img src="pics\softmax-Val-Accuracy.png" alt="softmax-Val-Accuracy" style="zoom:100%;" />

## distance

<img src="pics\distance-Train-Loss.png" alt="distance-Train-Loss" style="zoom:100%;" />

<img src="pics\distance-Val-Loss.png" alt="distance-Val-Loss" style="zoom:100%;" />

<img src="pics\distance-Train-Accuracy.png" alt="distance-Train-Accuracy" style="zoom:100%;" />

<img src="pics\distance-Val-Accuracy.png" alt="distance-Val-Accuracy" style="zoom:100%;" />