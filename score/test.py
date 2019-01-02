# coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import generateds as generateds
import forward as forward
import backward as backward

TEST_INTERVAL_SECS = 5
# 一次取1100张测试
TEST_NUM = 1100

import numpy as np


def test():
    # 接下来定义的节点在计算图g内，计算图是tensorflow的默认图
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            TEST_NUM,
            forward.IMAGE_SIZE,
            forward.IMAGE_SIZE,
            forward.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        # False表示这是测试过程，正则化参数regularizer设置为None
        y = forward.forward(x, False, None)

        # 实例化一个存储滑动平均值的saver
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 预测分数与实际分数相减取绝对值，求平均，即为平均分差
        distance_prediction = tf.abs(tf.argmax(y, 1) - tf.argmax(y_, 1))
        distance = tf.reduce_mean(tf.cast(distance_prediction, tf.float32))

        # 预测准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 打印预测的分数
        # mse = tf.losses.mean_squared_error(tf.argmax(y, 1), tf.argmax(y_, 1))
        # rmse=tf.sqrt(mse)
        # thisprint = tf.Print(y, [tf.argmax(y, 1)],summarize=1100)
        # with tf.control_dependencies([rmse, thisprint]):
        #     myrmse = tf.no_op(name='train')

        # 读取所有batch，isTrain=False表示读取测试集
        img_batch, label_batch = generateds.get_tfrecord(TEST_NUM, isTrain=False)

        while True:
            with tf.Session() as sess:
                # 找当前存在的model并恢复
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 恢复轮数。分割文件名，-1表示分割后的最后一部分，在这里即为训练轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 批获取
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    # 运行img_batch,label_batch计算节点
                    xs, ys = sess.run([img_batch, label_batch])
                    reshaped_x = np.reshape(xs, (
                        TEST_NUM,
                        forward.IMAGE_SIZE,
                        forward.IMAGE_SIZE,
                        forward.NUM_CHANNELS))
                    # 运行计算节点，得到输出
                    accuracy_score, dis = sess.run([accuracy, distance], feed_dict={x: reshaped_x, y_: ys})
                    print("After %s training step(s), test accuracy = %g, distance = %g " % (
                    global_step, accuracy_score, dis))
                    # 协调器coord发出所有线程终止信号
                    coord.request_stop()
                    # 把开启的线程加入主线程，等待threads结束
                    coord.join(threads)
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    test()


if __name__ == '__main__':
    main()
