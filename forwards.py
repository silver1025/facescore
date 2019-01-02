# coding: utf-8

import tensorflow as tf

# 只把70%数据用作参数更新
num_keep_radio = 0.7
slim = tf.contrib.slim


def P_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    '''pnet的结构
    inputs：喂入数据格式
    training：布尔变量，是否是训练
    '''
    with tf.variable_scope('PNet'):
        # 设置一些默认值
        with slim.arg_scope([slim.conv2d], activation_fn=prelu,  # 默认激活函数为prelu
                            weights_initializer=slim.xavier_initializer(),  # Xavier初始化器在初始化深度学习网络的时候让权重不大不小
                            weights_regularizer=slim.l2_regularizer(0.0005),  # L2范式正则化
                            padding='VALID'):  # padding默认为“VALID”
            # 构造网络
            # 卷积操作，卷积核的个数是10，卷积核的形式是[3,3]，步长为1，其余的参数和上面的slim.arg_scope一样
            net = slim.conv2d(inputs, 10, 3, scope='conv1')
            # 池化
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, padding='SAME', scope='pool1')
            # 卷积
            net = slim.conv2d(net, 16, 3, scope='conv2')
            # 卷积
            net = slim.conv2d(net, 32, 3, scope='conv3')
        # 二分类输出通道数为2
        conv4_1 = slim.conv2d(net, 2, 1, activation_fn=tf.nn.softmax, scope='conv4_1')
        # 框回归
        bbox_pred = slim.conv2d(net, 4, 1, activation_fn=None, scope='conv4_2')
        # 五个关键点
        landmark_pred = slim.conv2d(net, 10, 1, activation_fn=None, scope='conv4_3')

        if training:
            # 人脸类别判断损失
            cls_prob = tf.squeeze(conv4_1, [1, 2],
                                  name='cls_prob')  # 本来是[batch,1,1,2],输出的张量shape为[batch,2]   根据张量的shape，删除第1和第2维（第0维开始算）（如果此维的维度为1）
            cls_loss = cls_ohem(cls_prob, label)

            # 人脸框损失
            bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')  # 本来是[batch,1,1,4],输出的shape是[bacth,4]
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)

            # 关键点损失
            landmark_pred = tf.squeeze(landmark_pred, [1, 2],
                                       name='landmark_pred')  # 本来是[batch,1,1,10],输出的shape是[bacth,10]
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)

            # 分类准确率
            accuracy = cal_accuracy(cls_prob, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())  # 多种损失求和
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
        else:
            # 测试时batch_size=1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)  # shape变成[batch,2]
            bbox_pred_test = tf.squeeze(bbox_pred, axis=0)  # shape=[batch,4]
            landmark_pred_test = tf.squeeze(landmark_pred, axis=0)  # shape=[batch,10]
            return cls_pro_test, bbox_pred_test, landmark_pred_test


def R_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    '''RNet结构'''
    with tf.variable_scope('RNet'):
        # 设置默认值
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='VALID'):
            # 构建前向传播网络
            net = slim.conv2d(inputs, 28, 3, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 48, 3, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 64, 2, scope='conv3')
            fc_flatten = slim.flatten(net)  # shape=[one,k]以第一维不变为标准，压平。
            fc1 = slim.fully_connected(fc_flatten, num_outputs=128, scope='fc1')
            # 三个输出分支
            cls_prob = slim.fully_connected(fc1, num_outputs=2, activation_fn=tf.nn.softmax, scope='cls_fc')  # 判断是不是脸
            bbox_pred = slim.fully_connected(fc1, num_outputs=4, activation_fn=None, scope='bbox_fc')  # 人脸框坐标
            landmark_pred = slim.fully_connected(fc1, num_outputs=10, activation_fn=None, scope='landmark_fc')  # 关键点坐标
            if training:  # 用作训练
                cls_loss = cls_ohem(cls_prob, label)  # 人脸判断的损失
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)  # 人脸框坐标损失
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)  # 关键点坐标的损失
                accuracy = cal_accuracy(cls_prob, label)  # 人脸判断准确率
                L2_loss = tf.add_n(slim.losses.get_regularization_losses())  # L2正则化损失
                return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
            else:  # 用作测试
                return cls_prob, bbox_pred, landmark_pred


def O_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    '''ONet结构'''
    with tf.variable_scope('ONet'):
        # 设置默认值
        with slim.arg_scope([slim.conv2d],
                            activation_fn=prelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            padding='VALID'):
            # 构建前向传播网络
            net = slim.conv2d(inputs, 32, 3, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 64, 3, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 64, 3, scope='conv3')
            net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, padding='SAME', scope='pool3')
            net = slim.conv2d(net, 128, 2, scope='conv4')
            fc_flatten = slim.flatten(net)
            fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope='fc1')
            # 三个输出分支
            cls_prob = slim.fully_connected(fc1, num_outputs=2, activation_fn=tf.nn.softmax, scope='cls_fc')  # 判断是不是脸
            bbox_pred = slim.fully_connected(fc1, num_outputs=4, activation_fn=None, scope='bbox_fc')  # 人脸框坐标
            landmark_pred = slim.fully_connected(fc1, num_outputs=10, activation_fn=None, scope='landmark_fc')  # 关键点坐标
            if training:  # 用作训练
                cls_loss = cls_ohem(cls_prob, label)  # 人脸判断的损失
                bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)  # 人脸框坐标损失
                landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)  # 关键点坐标的损失
                accuracy = cal_accuracy(cls_prob, label)  # 人脸判断准确率
                L2_loss = tf.add_n(slim.losses.get_regularization_losses())  # L2正则化损失
                return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
            else:  # 用作测试
                return cls_prob, bbox_pred, landmark_pred


def prelu(inputs):
    '''prelu函数定义
    prelu(x) = relu(x)+0.125(x-abs(x))
    '''
    alphas = tf.get_variable('alphas', shape=inputs.get_shape()[-1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.25))  # 初始化为0.25
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def cls_ohem(cls_prob, label):
    '''计算类别损失
    参数：
      cls_prob：一个张量，预测类别，是否有人，shape为[batch,2]
      label：一个张量，shape为[batch]，标识，1是pos，0是neg，-1是part，-2是关键点
    返回值：
      损失
    '''

    # 把cls_prob拉成一维的
    num_cls_prob = tf.size(cls_prob)  # 返回所有元素的总数
    cls_prob_reshpae = tf.reshape(cls_prob, [num_cls_prob, -1])  # （-1表示长度根据其他维计算得出），一维的张量

    # 根据label标识选择每组二分类数据应该选择的概率数值
    zeros = tf.zeros_like(label)  # 创建一个和label相同形状的全零张量
    label_filter_invalid = tf.where(tf.less(label, 0), zeros,  # 把label里负值置零
                                    label)  # label<0则tf.less返回真，where条件为真的位置取zeros的值，否则取label的值
    label_int = tf.cast(label_filter_invalid, tf.int32)  # label的格式转换成整型
    num_row = tf.to_int32(cls_prob.get_shape()[0])  # 获取batch数
    row = tf.range(num_row) * 2  # tf.range(n)会生成一个[0,1,2,...,n]的列表
    indices_ = row + label_int  # 根据label做出了用于cls_prob_reshpae取概率的下标指示

    # 标识1和0都取各自需要的概率值，刚好cls_prob_reshpae的偶数位置对应不为人脸情况概率，奇数位置上是为人脸情况概率。
    label_prob = tf.squeeze(
        tf.gather(cls_prob_reshpae, indices_))  # tf.gather按照indices_指示的下标值和顺序，从cls_prob_reshpae抽出相应位置的值组成新的张量

    # 统计neg和pos的总数量
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)  # 创建一个全零张量
    ones = tf.ones_like(label_prob, dtype=tf.float32)  # 创建一个全1张量
    valid_inds = tf.where(label < zeros, zeros, ones)  # where条件为真的位置取zeros的值，否则取ones的值（pos和neg的位置都置一,得到了一个标准的标签）
    num_valid = tf.reduce_sum(valid_inds)  # 求和，得到label里面pos和neg的数量和。

    # 计算交叉熵！-∑x*log(y) ,
    loss = -tf.log(label_prob + 1e-10)
    loss = loss * valid_inds  # （这是每个元素各自相乘！！！！）valid_inds是标准的标签，人脸则为1，否则为0
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)  # 选取70%的数据
    loss, _ = tf.nn.top_k(loss, k=keep_num)  # （按照最后一个维度取出前k个最大的数据，默认从大到小进行排序。），取出前70%大的。
    return tf.reduce_mean(loss)  # 前百分之70的损失值的平均数


def bbox_ohem(bbox_pred, bbox_target, label):
    '''计算box的损失'''
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    # 保留pos和part的数据
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    # 计算平方差损失
    square_error = tf.square(bbox_pred - bbox_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    # 保留的数据的个数
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # 保留pos和part部分的损失
    square_error = square_error * valid_inds
    square_error, _ = tf.nn.top_k(square_error, k=keep_num)
    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_target, label):
    '''计算关键点损失'''
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    # 只保留landmark数据
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    # 计算平方差损失
    square_error = tf.square(landmark_pred - landmark_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    # 保留数据个数
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # 保留landmark部分数据损失
    square_error = square_error * valid_inds
    square_error, _ = tf.nn.top_k(square_error, k=keep_num)
    return tf.reduce_mean(square_error)


def cal_accuracy(cls_prob, label):
    '''计算分类准确率'''
    # 预测最大概率的类别，0代表无人，1代表有人
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    # 保留label>=0的数据，即pos和neg的数据
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    # 获取pos和neg的label值
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)
    # 计算准确率
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
    return accuracy_op
