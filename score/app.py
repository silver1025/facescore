# coding:utf-8
# 输入一张白底黑字的手写数字图片，预测结果并输出
import tensorflow as tf
import numpy as np
from PIL import Image
import forward as forward
import backward as backward


def restore_model(testPicArr):
    # 接下来定义的节点在计算图tg内，tg是tensorflow的默认图
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [1,
                                        forward.IMAGE_SIZE,
                                        forward.IMAGE_SIZE,
                                        forward.NUM_CHANNELS])
        # 测试数据，不需要正则化，输入参数None
        y = forward.forward(x, False, None)
        # y中最大值的索引号，即为可能性最大的数字，预测的结果
        preValue = tf.argmax(y, 1)

        # 实例化一个存储滑动平均值的saver
        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            # 找到存储的model并恢复
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 运行preValue节点，返回的preValue就是预测结果
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


# 对图像进行预处理
def pre_pic(picName):
    img = Image.open(picName)
    # 重新设置大小，Image.ANTIALIAS：消除锯齿
    reIm = img.resize((128, 128), Image.ANTIALIAS)
    # 彩色图片，转化RGB，变成数组
    im_arr = np.array(reIm.convert('RGB'))
    # reshape数组，变成[1,128*128*3]
    nm_arr = im_arr.reshape([1, 128 * 128 * 3])
    # 转化为float类型
    nm_arr = nm_arr.astype(np.float32)
    # nm_arr矩阵中所有元素乘1.0/255.0，转化为0或1的形式
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)
    return img_ready


def application():
    testNum = input("input the num of test picture:")
    for i in range(int(testNum)):
        # 读取控制台的输入，注意python3.6没有raw_input函数
        testPic = input("the path of test picture:")
        # 对testPic进行预处理
        testPicArr = pre_pic(testPic)
        reshaped_xs = np.reshape(testPicArr, (
            1,
            forward.IMAGE_SIZE,
            forward.IMAGE_SIZE,
            forward.NUM_CHANNELS))
        preValue = restore_model(reshaped_xs)
        # 输出结果是0-8之间。加两分得到2-10之间的实际得分
        print("分数分为2-10分，这张图片是%d分" % (preValue + 2))


def main():
    application()


if __name__ == '__main__':
    main()

'''
D:\房悦\课程资料\人工智能\CNN_face\test_img\1章子怡.jpg
D:\房悦\课程资料\人工智能\CNN_face\test_img\3章泽天.jpg
D:\房悦\课程资料\人工智能\CNN_face\test_img\5路人甲.jpg
D:\房悦\课程资料\人工智能\CNN_face\test_img\7曾志伟.jpg
'''
