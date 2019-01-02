# coding: utf-8


from forwards import P_Net, R_Net, O_Net
import os
import numpy as np
import tensorflow as tf
import random
import cv2

# 经过多少batch显示数据
display = 100
# 初始学习率
base_lr = 0.001
LEARNING_RATE_DECAY = 0.995
batch_size = 384


def main(size):
    base_dir = os.path.join('data/', str(size))  # 图片的路径

    if size == 12:  # 根据size大小来确定
        net = 'PNet'  # 网络类型
        end_epoch = 30  # 迭代次数
        # 设定几种损失的占比
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 0.5
    elif size == 24:  # 同上
        net = 'RNet'
        end_epoch = 22
        # 设定几种损失的占比
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 0.5
    elif size == 48:  # 同上
        net = 'ONet'
        end_epoch = 22
        # 设定几种损失的占比
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 1

    # 设定一个batch的输入和总数据量num
    if net == 'PNet':  # 如果是PNET
        # 计算一共多少组数据
        label_file = os.path.join(base_dir, 'train_pnet_landmark.txt')  # 打开标签文件
        f = open(label_file, 'r')
        num = len(f.readlines())  # 获得标签行数
        # 从tfrecord读取数据
        dataset_dir = os.path.join(base_dir, 'tfrecord/train_PNet_landmark.tfrecord_shuffle')  # tfrecord文件地址
        image_batch, label_batch, bbox_batch, landmark_batch = read_single_tfrecord(dataset_dir, batch_size,
                                                                                    net)  # 从单个tfrecord读取数据
    else:
        # 计算一共多少组数据
        label_file1 = os.path.join(base_dir, 'pos_%d.txt' % (size))
        f1 = open(label_file1, 'r')  # 打开pos类的标签文件
        label_file2 = os.path.join(base_dir, 'part_%d.txt' % (size))
        f2 = open(label_file2, 'r')  # 打开part类的标签文件
        label_file3 = os.path.join(base_dir, 'neg_%d.txt' % (size))
        f3 = open(label_file3, 'r')  # 打开neg类的标签文件
        label_file4 = os.path.join(base_dir, 'landmark_%d_aug.txt' % (size))
        f4 = open(label_file4, 'r')  # 打开landmark的标签文件
        num = len(f1.readlines()) + len(f2.readlines()) + len(f3.readlines()) + len(f4.readlines())  # 计算总的长度

        # 各个tfrecord文件路径
        pos_dir = os.path.join(base_dir, 'tfrecord/pos_landmark.tfrecord_shuffle')
        part_dir = os.path.join(base_dir, 'tfrecord/part_landmark.tfrecord_shuffle')
        neg_dir = os.path.join(base_dir, 'tfrecord/neg_landmark.tfrecord_shuffle')
        landmark_dir = os.path.join(base_dir, 'tfrecord/landmark_landmark.tfrecord_shuffle')
        dataset_dirs = [pos_dir, part_dir, neg_dir, landmark_dir]  # 组成路径的列表

        # 设置每次取的各种数据占比
        # 使每一个batch的各种数据占比保持不变
        pos_radio, part_radio, landmark_radio, neg_radio = 1.0 / 6, 1.0 / 6, 1.0 / 6, 3.0 / 6
        # 计算各种类型图片一次喂入的大小
        pos_batch_size = int(np.ceil(batch_size * pos_radio))
        assert pos_batch_size != 0, "Batch Size 有误 "  # assert断言，断言不满足则报后面的error
        part_batch_size = int(np.ceil(batch_size * part_radio))
        assert part_batch_size != 0, "BBatch Size 有误 "
        neg_batch_size = int(np.ceil(batch_size * neg_radio))
        assert neg_batch_size != 0, "Batch Size 有误 "
        landmark_batch_size = int(np.ceil(batch_size * landmark_radio))
        assert landmark_batch_size != 0, "Batch Size 有误 "
        batch_sizes = [pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size]  # 汇聚成总的batch大小的列表

        # 取一个batch
        image_batch, label_batch, bbox_batch, landmark_batch = read_multi_tfrecords(dataset_dirs, batch_sizes,
                                                                                    net)  # 从多个tfrecord文件读取数据

    MAX_STEP = int(num / batch_size + 1) * end_epoch  # 训练的总batch数

    # 未知数目的神经网络输入占位
    input_image = tf.placeholder(tf.float32, shape=[batch_size, size, size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[batch_size], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[batch_size, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32, shape=[batch_size, 10], name='landmark_target')

    # 定义损失值
    # 根据网络获得损失值
    if net == 'PNet':  # 如果是PNET
        cls_loss_op, bbox_loss_op, landmark_loss_op, L2_loss_op, accuracy_op = P_Net(input_image,
                                                                                     label, bbox_target,
                                                                                     landmark_target, training=True)
    elif net == 'RNet':  # 如果是RNET
        cls_loss_op, bbox_loss_op, landmark_loss_op, L2_loss_op, accuracy_op = R_Net(input_image,
                                                                                     label, bbox_target,
                                                                                     landmark_target, training=True)
    elif net == 'ONet':  # 如果是ONET
        cls_loss_op, bbox_loss_op, landmark_loss_op, L2_loss_op, accuracy_op = O_Net(input_image,
                                                                                     label, bbox_target,
                                                                                     landmark_target, training=True)
    # 按比例求得整体损失
    total_loss_op = radio_cls_loss * cls_loss_op + radio_bbox_loss * bbox_loss_op + radio_landmark_loss * landmark_loss_op + L2_loss_op

    global_step = tf.Variable(0, trainable=False)  # 全局的轮数
    # 指数衰减学习率
    lr_op = tf.train.exponential_decay(
        base_lr,
        global_step,
        num / batch_size,
        LEARNING_RATE_DECAY,
        staircase=True)
    # 采用momentum优化
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)  # 采用了momentum算法的优化方法
    train_op = optimizer.minimize(total_loss_op, global_step)  # 这里加上global_step，使得global_step每轮训练自动加一。

    saver = tf.train.Saver(max_to_keep=1)  # 实例化saver对象

    # 网络初始化
    init = tf.global_variables_initializer()  # 变量初始化
    sess = tf.Session()
    sess.run(init)  # 变量初始化

    # 断点续训
    model_path = os.path.join('model/', net)  # 根据网络确定模型路径
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    prefix = os.path.join(model_path, net)  # 模型名

    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("使用现有模型:", ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()  # 开启线程协调器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 开启线程
    try:
        for i in range(MAX_STEP):
            if coord.should_stop():  # 线程协调器，线程调用request_stop()则should_stop返回True
                break
            image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array = sess.run(
                [image_batch, label_batch, bbox_batch, landmark_batch])

            # 随机翻转图像，增强学习
            #    image_batch_array, landmark_batch_array = random_flip_images(image_batch_array, label_batch_array,
            #                                                                 landmark_batch_array)

            # 喂入数据训练
            _, _, step = sess.run([train_op, lr_op, global_step],
                                  feed_dict={input_image: image_batch_array, label: label_batch_array,
                                             bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})
            # 展示当前参数，并存储模型
            if (i + 1) % 10 == 0:  # 每10轮展示一次
                cls_loss, bbox_loss, landmark_loss, L2_loss, lr, acc = sess.run(
                    # 每display次训练，用一个batch的数据计算一次损失值和学习率、准确率
                    [cls_loss_op, bbox_loss_op, landmark_loss_op, L2_loss_op, lr_op, accuracy_op],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,
                               landmark_target: landmark_batch_array})
                #  计算整体损失
                total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + radio_landmark_loss * landmark_loss + L2_loss
                print(
                    "Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                        step + 1, MAX_STEP, acc, cls_loss, bbox_loss, landmark_loss, L2_loss, total_loss,
                        lr))  # 显示损失值和学习率、准确率等
                # 存储模型
                saver.save(sess, prefix, global_step=global_step)

    except tf.errors.OutOfRangeError:
        print("完成！！！")
    finally:
        coord.request_stop()  # 关闭线程协调器
    coord.join(threads)
    sess.close()


def read_single_tfrecord(tfrecord_file, batch_size, net):
    '''从单个tfrecord文件读取数据'''
    filename_queue = tf.train.string_input_producer([tfrecord_file], shuffle=True)  # 生成一个先入先出的队列，文件阅读器会使用它来读取数据。
    reader = tf.TFRecordReader()  # 创建一个reader
    _, serialized_example = reader.read(filename_queue)  # 把读出的每个样本保存在serialized_example中进行解序列化
    image_features = tf.parse_single_example(serialized_example,
                                             features={
                                                 'image/encoded': tf.FixedLenFeature([], tf.string),  # 编码后的原始大图片
                                                 'image/label': tf.FixedLenFeature([], tf.int64),
                                                 # 标识，1是pos，0是neg，-1是part，-2是关键点
                                                 'image/roi': tf.FixedLenFeature([4], tf.float32),  # 人脸框位置
                                                 'image/landmark': tf.FixedLenFeature([10], tf.float32)  # 关键点位置
                                             }
                                             )
    if net == 'PNet':
        image_size = 12
    elif net == 'RNet':
        image_size = 24
    elif net == 'ONet':
        image_size = 48
    # 变形处理
    image = tf.decode_raw(image_features['image/encoded'], tf.uint8)  # 将image/encoded字符串转换为8位无符号整型
    image = tf.reshape(image, [image_size, image_size, 3])  # 转换成三维数组
    image = (tf.cast(image, tf.float32) - 127.5) / 128  # 变成[-1,1]内的浮点数
    label = tf.cast(image_features['image/label'], tf.float32)  # 把标签列表变成浮点数
    roi = tf.cast(image_features['image/roi'], tf.float32)  # 把roi列表变成浮点数
    landmark = tf.cast(image_features['image/landmark'], tf.float32)  # 把landmark列表变成浮点数

    # 批处理，拿一个batch_size出来
    image, label, roi, landmark = tf.train.batch([image, label, roi, landmark],  # 列表中的样本
                                                 batch_size=batch_size,  # 一次拿出大小
                                                 num_threads=2,  # 线程数
                                                 capacity=batch_size)  # 队列中元素的最大数量
    # 拿出来后再次变形
    label = tf.reshape(label, [batch_size])  # 标识，1是pos，0是neg，-1是part，-2是关键点
    roi = tf.reshape(roi, [batch_size, 4])  # 人脸框位置
    landmark = tf.reshape(landmark, [batch_size, 10])  # 关键点位置
    return image, label, roi, landmark


def read_multi_tfrecords(tfrecord_files, batch_sizes, net):
    '''读取多个tfrecord文件放一起'''
    # 四个类型的tfrecord路径
    pos_dir, part_dir, neg_dir, landmark_dir = tfrecord_files

    # 四个类型的batchsize
    pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size = batch_sizes

    # 四组读取
    pos_image, pos_label, pos_roi, pos_landmark = read_single_tfrecord(pos_dir, pos_batch_size, net)  # 读取pos数据
    part_image, part_label, part_roi, part_landmark = read_single_tfrecord(part_dir, part_batch_size, net)  # 读取part数据
    neg_image, neg_label, neg_roi, neg_landmark = read_single_tfrecord(neg_dir, neg_batch_size, net)  # 读取neg数据
    landmark_image, landmark_label, landmark_roi, landmark_landmark = read_single_tfrecord(landmark_dir,
                                                                                           landmark_batch_size,
                                                                                           net)  # 读取landmark数据

    # pos、part、neg、landmark的相同类型数据各自在第1维上连接起来
    images = tf.concat([pos_image, part_image, neg_image, landmark_image], 0, name="concat/image")
    labels = tf.concat([pos_label, part_label, neg_label, landmark_label], 0, name="concat/label")
    rois = tf.concat([pos_roi, part_roi, neg_roi, landmark_roi], 0, name="concat/roi")
    landmarks = tf.concat([pos_landmark, part_landmark, neg_landmark, landmark_landmark], 0, name="concat/landmark")
    return images, labels, rois, landmarks


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)  # 随机调整对比度
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)  # 随机调整亮度
    # inputs = tf.image.random_hue(inputs, max_delta=0.2)  # 随机调整色相
    # inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)  # 随机调整饱和度

    return inputs


# ？？？疑问，bbox不翻转的话不会出问题吗？
def random_flip_images(image_batch, label_batch, landmark_batch):
    '''随机翻转图像'''
    if random.choice([0, 1]) > 0:  # 随机选择0,1
        num_images = image_batch.shape[0]  # 图片数量
        fliplandmarkindexes = np.where(label_batch == -2)[
            0]  # 返回满足条件的元素的下标，即标识为关键点的label_batch列表的下标构成的列表（本来是个元组，第0维即为所需列表）。
        flipposindexes = np.where(label_batch == 1)[0]  # 返回满足条件的元素的下标，即标识为pos的label_batch列表的下标构成的列表。
        flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))  # 列表拼接起来

        # 图片水平翻转
        for i in flipindexes:  # 所有标识为关键点和pos的image进行翻转 
            cv2.flip(image_batch[i], 1, image_batch[i])  # 第一个参数为源文件，第二个为翻转方向（1为水平，0为垂直，-1为水平垂直），第三个参数为目的文件。

        # 坐标水平翻转
        for i in fliplandmarkindexes:  # 所有标识为关键点的landmark进行翻转
            landmark_ = landmark_batch[i].reshape((-1, 2))  # 转化成两两一组的坐标组
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])  # 坐标水平翻转
            landmark_[[0, 1]] = landmark_[[1, 0]]  # 左右眼存储顺序互换
            landmark_[[3, 4]] = landmark_[[4, 3]]  # 左右嘴角存储顺序互换
            landmark_batch[i] = landmark_.ravel()  # 变成一维

    return image_batch, landmark_batch


if __name__ == '__main__':
    size = int(input("请输入训练网络种类，PNET为12，RNET为24，ONET为48:"))
    if size == 12 or size == 24 or size == 48:
        main(size)
    else:
        print("输入错误，请输入12,24,48中的一个值")
