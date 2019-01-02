# coding:utf-8
import tensorflow as tf

# 图片尺寸128*128
IMAGE_SIZE = 128
# 三通道
NUM_CHANNELS = 3
# 第一层卷积核大小为5
CONV1_SIZE = 5
# 第一层卷积核个数为24
CONV1_KERNEL_NUM = 24
# 第二层卷积核大小为5
CONV2_SIZE = 5
# 第二层卷积核个数为96
CONV2_KERNEL_NUM = 96
# 全连接层第一层
FC_SIZE = 32 * 32 * 96
# 全连接第二层
FC2_SIZE = 1024
# 输出9种分类，对应2-10分
OUTPUT_NODE = 9


def get_weight(shape, regularizer):
    # 生成基于正态分布的随机数，shape为生成随机数的规格，[行数,列数]
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    # 正则化的一部分：regularizer为w的权值，若非空，就用contrib.layers.l2_regularizer计算正则化公式第二项，加到losses中
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    # 偏置项初值为0
    b = tf.Variable(tf.zeros(shape))
    return b


# 卷积，x是输入描述，w是卷积核描述
def conv2d(x, w):
    # 步长[1,1,1,1]
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化，x是输入描述
def max_pool_2x2(x):
    # 池化核[1,2,2,1]，步长[1,2,2,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 前向计算
def forward(x, train, regularizer):
    # 初始化卷积核的变量
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    # 初始化卷积中的偏置项
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # 进行第一层卷积
    conv1 = conv2d(x, conv1_w)
    # print(conv1.shape)
    # 卷积结果加上偏置，经过relu激活函数
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    # 第一层池化，relu1是上一步的卷积后的结果，作为池化的输入
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # 第二层卷积，pool1是上一步的输出
    conv2 = conv2d(pool1, conv2_w)
    # 卷积结果加上偏置，经过relu激活函数
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    # 第二层池化
    pool2 = max_pool_2x2(relu2)

    # 获取处理后结果的维度
    pool_shape = pool2.get_shape().as_list()
    # pool_shape[1] 为长 pool_shape[2] 为宽 pool_shape[3]为高
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 将pool2转化成矩阵，pool_shape[0]为batch值
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 实现第三层全连接层
    fc1_w = get_weight([nodes, FC2_SIZE], regularizer)
    fc1_b = get_bias([FC2_SIZE])
    # reshaped为经过卷积池化后的输出，使用relu激活函数
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)

    # 如果是训练阶段，则对该层输出使用dropout
    if train: fc1 = tf.nn.dropout(fc1, 0.5)

    # 实现第四层全连接层
    fc2_w = get_weight([FC2_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    # 输出九分类，不需要经过激活函数
    y = tf.matmul(fc1, fc2_w) + fc2_b

    return y
