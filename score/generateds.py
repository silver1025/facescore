# coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_path = '../Images/'

label_train_path = '../label/train.txt'
tfRecord_train = '../data/train.tfrecords'

label_test_path = '../label/test.txt'
tfRecord_test = '../data/test.tfrecords'
data_path = '../data'


def write_tfRecord(tfRecordName, image_path, label_path):
    # 新建writer
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    # 打开label，label文件中每一行的格式为：图片名 分数
    f = open(label_path, 'r')
    # 按行读取
    contents = f.readlines()
    f.close()
    for content in contents:
        # 用空格切分
        value = content.split(" ")
        # 切分后的第一项即为图片名，和上一层地址拼接得到图片完整地址
        img_path = image_path + value[0]
        # 彩色图片用RGB方式转换
        img = Image.open(img_path).convert('RGB')

        # 将图片变为128*128的尺寸
        img = img.resize((128, 128))
        # 转化为二进制数据
        img_raw = img.tobytes()
        # 九分类，生成一个1*9的全0向量
        labels = [0] * 9

        # value[1]为这张人脸的得分，在1到5之间，乘2，四舍五入，2-10之间
        score = str(round(float(value[1]) * 2))
        # 2-10变成0-8，labels对应的位置赋为1
        labels[int(score) - 2] = 1

        # tf.train.Example创建一个example，要传入一个tf.train.Features
        # tf.train.Features里用键值对表示，值用tf.train.Feature传递
        # 封装img_raw和label
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))

        # 用write把example转化为字符串存储起来
        writer.write(example.SerializeToString())
        num_pic += 1
        print("the number of picture:", num_pic)
    writer.close()
    print("write tfrecord successful")
    # print(num2,num3,num4,num5,num6,num7,num8,num9,num10)


def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
    # 生成训练集，写入tfRecord_train
    write_tfRecord(tfRecord_train, image_path, label_train_path)
    # 生成测试集，写入tfRecord_test
    write_tfRecord(tfRecord_test, image_path, label_test_path)


# 读取数据
def read_tfRecord(tfRecord_path):
    # 新建文件名队列，告知包含哪些文件，tfRecord_path为tfRecord的存放地址
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    reader = tf.TFRecordReader()
    # 把读出的每个样本保存在serialized_example中
    _, serialized_example = reader.read(filename_queue)
    # 对serialized_example进行解序列化
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           # 键要和write_tfRecord里的键相同，标签要给出长度，即几分类
                                           'label': tf.FixedLenFeature([9], tf.int64),
                                           # example中，img是二进制字符串存储的
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    # 将img_raw字符串转化为8位无符号整形
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # 转化为1行128*128*3列
    img.set_shape([128 * 128 * 3])
    # 转化为0到1之间浮点数
    img = tf.cast(img, tf.float32) * (1. / 255)
    # 把标签列表转化为浮点数形式
    label = tf.cast(features['label'], tf.float32)
    return img, label


# backward中会调用此函数，读取num那么大的一批数据
def get_tfrecord(num, isTrain=True):
    # isTrain为true，读取训练集，否则读取测试集
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    # 读出全部img和label
    img, label = read_tfRecord(tfRecord_path)
    # 将读到的img和label分为num个大小的一批批
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    # 每次取num个
                                                    batch_size=num,
                                                    # 开启两个线程
                                                    num_threads=4,
                                                    # 同时拿多少张牌打乱顺序
                                                    capacity=800,
                                                    # 把手里的牌替换掉多少张，越大，数据越乱
                                                    min_after_dequeue=700)
    # 返回的是所有batch，不是一个batch
    return img_batch, label_batch


def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()
