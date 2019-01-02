# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import generateds as generateds
import forward as forward
import os
import numpy as np

# 一次取200张图片训练
BATCH_SIZE = 200
# 初始学习率
LEARNING_RATE_BASE = 0.001
# 学习率衰减率
LEARNING_RATE_DECAY = 0.98
# 正则化权值
REGULARIZER = 0.0001
# 训练200次
STEPS = 200
# 计算滑动平均值时的衰减率
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "model/score/" # 训练时需要修改
MODEL_NAME = "cnn_model"
# 训练集图片数
train_num_examples = 4400


def backward():
    # 输入x的占位符
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    forward.IMAGE_SIZE,
                                    forward.IMAGE_SIZE,
                                    forward.NUM_CHANNELS])
    # 实际分数y_的占位符
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
    # 得到预测分数y
    y = forward.forward(x, True, REGULARIZER)
    # 记录训练了多少步
    global_step = tf.Variable(0, trainable=False)

    # y过一个softmax层，转化为概率，计算y和y_的交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 求平均
    cem = tf.reduce_mean(ce)
    # get_collection取出losses集合的值，add_n把值加起来，表示进行正则化
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 这一轮训练集的准确率
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # AdamOptimizer：根据损失函数对每个参数的梯度的一阶矩估计和二阶矩估计动态调整针对于每个参数的学习速率
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 计算w的滑动平均值，记录每个w过去一段时间内的平均值，避免w迅速变化，导致模型过拟合
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # ema.apply后面的括号是更新列表,每次运行sess.run(ema_op)时,对待训练的参数求滑动平均值
    ema_op = ema.apply(tf.trainable_variables())

    ##将训练过程和计算滑动平均的过程绑定
    with tf.control_dependencies([train_op, ema_op]):
        # 将它们合并为一个训练节点
        train_step = tf.no_op(name='train')

    # 实例化一个 tf.train.Saver，之后可以用saver保存模型或读取模型
    saver = tf.train.Saver()

    # 取BATCH_SIZE数量的训练数据
    img_batch, label_batch = generateds.get_tfrecord(BATCH_SIZE, isTrain=True)

    with tf.Session() as sess:
        # 初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训，#如果地址下存在断点，就把断点恢复
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 创建线程管理器
        coord = tf.train.Coordinator()
        # 启动队列填充，读入文件
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            # 运行img_batch和label_batch，获得下一批训练数据
            xs, ys = sess.run([img_batch, label_batch])
            # 将xs转化为合适的shape准备喂入网络
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                forward.IMAGE_SIZE,
                forward.IMAGE_SIZE,
                forward.NUM_CHANNELS))
            # 运行之前定义的计算节点，获得输出
            _, loss_value, step, acc = sess.run([train_step, loss, global_step, accuracy],
                                                feed_dict={x: reshaped_xs, y_: ys})
            # 每10轮保存一次model
            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g. accuracy is  %g" % (
                    step, loss_value, acc))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        # 把开启的线程加入主线程，等待threads结束
        coord.join(threads)


def main():
    backward()


if __name__ == '__main__':
    main()
