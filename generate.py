# coding: utf-8

'''
截取pos，neg,part三种类型图片并resize成12x12大小作为PNet的输入
'''
import pickle
import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import *
import random
from forwards import P_Net, R_Net
from app import Detector, FcnDetector, MtcnnDetector

npr = np.random
data_dir = 'data'
# 最小脸大小设定
min_face = 20
# 生成hard_example的batch
batches = [2048, 256, 16]
# pent对图像缩小倍数
stride = 2
# 三个网络的阈值
thresh = [0.6, 0.7, 0.7]


def gen_12x12_pic():
    # face对应label的txt
    anno_file = 'data/wider_face_train.txt'
    # 图片地址
    im_dir = 'data/WIDER_train/images'
    # 裁剪后pos，part,neg图片放置位置
    pos_save_dir = 'data/12/positive'
    part_save_dir = 'data/12/part'
    neg_save_dir = 'data/12/negative'
    # PNet数据地址
    save_dir = 'data/12'

    # 创建路径
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)
    # 创建并打开人脸框位置记录文件
    f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
    f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
    f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
    # 打开widerface人脸框位置标签文件
    with open(anno_file, 'r') as f:
        annotations = f.readlines()  # 逐行读取
    num = len(annotations)  # 图片数
    print('总共的图片数： %d' % num)
    # 记录pos,neg,part三类生成数
    p_idx = 0
    n_idx = 0
    d_idx = 0
    # 记录读取图片数
    idx = 0
    for annotation in tqdm(annotations):  # 对每一个图片的记录
        annotation = annotation.strip().split(' ')  # 按照空格切分，strip为移除头尾空格
        im_path = annotation[0]  # 第一个元素为文件路径
        box = list(map(float, annotation[1:]))  # 之后的元素为人脸框坐标，利用map()函数让后面的值都经过float转换

        boxes = np.array(box, dtype=np.float32).reshape(-1, 4)  # 4个一组代表一个人脸框

        img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))  # 读取这个图片
        idx += 1  # 读取图片计数
        height, width, channel = img.shape  # 获取图片高、宽、通道数

        neg_num = 0
        # 先采样一定数量neg图片
        while neg_num < 50:

            # 随机选取截取图像大小
            size = npr.randint(12, min(width, height) / 2)  # randint(a,b)生成一个a，b间的整数
            # 随机选取左上坐标
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            # 截取box
            crop_box = np.array([nx, ny, nx + size, ny + size])
            # 计算iou值
            Iou = IOU(crop_box, boxes)
            # 截取图片并resize成12x12大小
            cropped_im = img[ny:ny + size, nx:nx + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            # iou值小于0.3判定为neg图像
            if np.max(Iou) < 0.3:  # 与所有标签框的iou值都小于0.3
                save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)  # 存储路径
                f2.write(neg_save_dir + '/%s.jpg' % n_idx + ' 0\n')  # 写入目录
                cv2.imwrite(save_file, resized_im)  # 保存截取后的图片
                n_idx += 1  # 计数
                neg_num += 1

        # 对每一个预设的人脸框
        for box in boxes:
            # 左上，右下坐标
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1  # 人脸框宽度w
            h = y2 - y1 + 1  # 人脸框高度h
            # 舍去图像过小和box在图片外的图像
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
            # 随机取5次框，和所有标记框做iou，若为neg则存储
            for i in range(5):
                size = npr.randint(12, min(width, height) / 2)  # 随机生成框大小
                # 这5个框，限制偏移量的标准是整个图片的大小
                # 随机生成关于x1,y1的偏移量，同时保证x1+delta_x>0,y1+delta_y>0（新的起点在正域）
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                # 截取后的左上角坐标，若在负域则置零
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # 排除右下角坐标位置超出图片尺度的
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                Iou = IOU(crop_box, boxes)
                cropped_im = img[ny1:ny1 + size, nx1:nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                    f2.write(neg_save_dir + '/%s.jpg' % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
            # 随机取20次，只和这一个标记框box做iou
            for i in range(20):
                # 再次随机选取size范围，更多截取pos和part图像
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))  # 此随机框大小和标志框相近
                # 除去尺度小的
                if w < 5:
                    continue
                # 偏移量在此标记框附近很小的范围内
                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)
                # 截取图像左上坐标计算是先计算x1+w/2表示的中心坐标，再+delta_x偏移量（得到新取框的中心坐标），再-size/2，
                # 变成新的左上坐标
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
                # 排除超出图片范围的框
                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                # 人脸框相对于截取图片的偏移量（归一化处理）
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[ny1:ny2, nx1:nx2, :]  # 截取图片
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)  # 图片变形
                # box扩充一个维度作为iou输入（第一位为1，表示只有一个框）
                box_ = box.reshape(1, -1)
                iou = IOU(crop_box, box_)  # box_只有一个框，只和这一个标记框做了iou

                if iou >= 0.65:  # 存为pos图片
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    f1.write(pos_save_dir + '/%s.jpg' % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1,
                                                                                              offset_y1, offset_x2,
                                                                                              offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif iou >= 0.4:  # 存为part图片
                    save_file = os.path.join(part_save_dir, '%s.jpg' % d_idx)
                    f3.write(part_save_dir + '/%s.jpg' % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1,
                                                                                                offset_y1, offset_x2,
                                                                                                offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

    print('%s 个图片已处理，pos：%s  part: %s neg:%s' % (idx, p_idx, d_idx, n_idx))
    f1.close()
    f2.close()
    f3.close()


def gen_pnet_imglist():
    '''将pos,part,neg,landmark四者混在一起'''
    size = 12
    with open(os.path.join(data_dir, '12/pos_12.txt'), 'r') as f:
        pos = f.readlines()
    with open(os.path.join(data_dir, '12/neg_12.txt'), 'r') as f:
        neg = f.readlines()
    with open(os.path.join(data_dir, '12/part_12.txt'), 'r') as f:
        part = f.readlines()
    with open(os.path.join(data_dir, '12/landmark_12_aug.txt'), 'r') as f:
        landmark = f.readlines()
    dir_path = os.path.join(data_dir, '12')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path, 'train_pnet_landmark.txt'), 'w') as f:
        nums = [len(neg), len(pos), len(part)]
        base_num = 250000
        print('neg数量：{} pos数量：{} part数量:{} 基数:{}'.format(len(neg), len(pos), len(part), base_num))
        if len(neg) > base_num * 3:
            neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)  # size为采样数量，replace表明可以重复
        else:
            neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
        sum_p = len(neg_keep) // 3
        pos_keep = npr.choice(len(pos), sum_p, replace=True)
        part_keep = npr.choice(len(part), sum_p, replace=True)
        print('neg数量：{} pos数量：{} part数量:{}'.format(len(neg_keep), len(pos_keep), len(part_keep)))
        for i in pos_keep:
            f.write(pos[i])
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])
        for item in landmark:
            f.write(item)


def gen_landmark(size):
    '''用于处理带有landmark的数据'''
    # 是否对图像变换
    argument = True
    if size == 12:
        net = 'PNet'
    elif size == 24:
        net = 'RNet'
    elif size == 48:
        net = 'ONet'
    image_id = 0
    # 数据输出路径为data/（size）
    OUTPUT = os.path.join(data_dir, str(size))
    if not os.path.exists(OUTPUT):  # 不存在则创建路径
        os.mkdir(OUTPUT)
    # 图片处理后输出路径为data/(size)/train_XNET_landmark_aug
    dstdir = os.path.join(OUTPUT, 'train_%s_landmark_aug' % (net))
    if not os.path.exists(dstdir):  # 不存在则创建
        os.mkdir(dstdir)
    # 人脸关键点标签文件
    ftxt = os.path.join(data_dir, 'trainImageList.txt')
    # 记录输出的图片类别和关键点坐标的txt文件
    f = open(os.path.join(OUTPUT, 'landmark_%d_aug.txt' % (size)), 'w')
    # 获取图像路径，box，关键点
    data = getDataFromTxt(ftxt, data_dir)  # 返回值包含(图像路径，人脸box，关键点坐标)
    idx = 0  # 计数
    for (imgPath, box, landmarkGt) in tqdm(data):
        # 存储人脸图片和关键点
        F_imgs = []
        F_landmarks = []
        img = cv2.imread(imgPath)  # 读取图片

        img_h, img_w, img_c = img.shape  # 获得高、宽、通道数
        gt_box = np.array([box.left, box.top, box.right, box.bottom])  # 人脸框坐标
        # 人脸图片
        f_face = img[box.top:box.bottom + 1, box.left:box.right + 1]  # 截取人脸图片
        # resize成网络输入大小
        f_face = cv2.resize(f_face, (size, size))

        landmark = np.zeros((5, 2))
        for index, one in enumerate(landmarkGt):  # enumerate函数返回一个（下标，内容）构成的列表
            # 关键点相对于左上坐标偏移量并归一化
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            landmark[index] = rv
        F_imgs.append(f_face)  # 存储图片
        F_landmarks.append(landmark.reshape(10))  # 存储关键点
        landmark = np.zeros((5, 2))
        if argument:
            # 对图像变换
            idx = idx + 1
            x1, y1, x2, y2 = gt_box
            gt_w = x2 - x1 + 1
            gt_h = y2 - y1 + 1
            # 除掉过小和负域的图像
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            for i in range(10):  # 取10个和现有框差不多大，位置也在现有框附近的框
                # 随机裁剪图像大小
                box_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                # 随机左上坐标偏移量
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                # 计算左上坐标
                nx1 = int(max(x1 + gt_w / 2 - box_size / 2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h / 2 - box_size / 2 + delta_y, 0))
                nx2 = nx1 + box_size
                ny2 = ny1 + box_size
                # 除去超过边界的
                if nx2 > img_w or ny2 > img_h:
                    continue
                # 裁剪边框，图片
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (size, size))
                iou = IOU(crop_box, np.expand_dims(gt_box, 0))
                # 只保留pos图像及其关键点
                if iou > 0.65:
                    F_imgs.append(resized_im)

                    # 通过图片变换进行增强训练
                    # 关键点相对偏移
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / box_size, (one[1] - ny1) / box_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1, 2)
                    box = BBox([nx1, ny1, nx2, ny2])
                    # 随机进行镜像处理
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    # 随机进行逆时针翻转
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), 5)
                        # 关键点偏移
                        landmark_rorated = box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))

                        # 左右翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    # 随机进行顺时针翻转
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), -5)
                        # 关键点偏移
                        landmark_rorated = box.projectLandmark(landmark_rorated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rorated.reshape(10))

                        # 左右翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        for i in range(len(F_imgs)):  # 对所有的截取的图片
            # 剔除关键点位置超出边际的图片
            if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:  # 比0小的跳过
                continue
            if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:  # 比1大的跳过
                continue
            # 符合条件的写入
            cv2.imwrite(os.path.join(dstdir, '%d.jpg' % (image_id)), F_imgs[i])
            landmarks = list(map(str, list(F_landmarks[i])))
            f.write(os.path.join(dstdir, '%d.jpg' % (image_id)) + ' -2 ' + ' '.join(landmarks) + '\n')
            image_id += 1  # 裁剪图片计数加一
    f.close()
    return F_imgs, F_landmarks


def flip(face, landmark):
    # 镜像
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return (face_flipped_by_x, landmark_)


def rotate(img, box, landmark, alpha):
    # 旋转
    center = ((box.left + box.right) / 2, (box.top + box.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[box.top:box.bottom + 1, box.left:box.right + 1]
    return (face, landmark_)


def gen_tfrecords(size):
    '''生成tfrecords文件'''
    # 原始数据存放地址
    dataset_dir = 'data/'
    # tfrecord存放地址
    output_dir = os.path.join(dataset_dir, str(size) + '/tfrecord')
    if not os.path.exists(output_dir):  # 不存在则创建
        os.mkdir(output_dir)
    # pnet只生成一个混合的tfrecords，rnet和onet要分别生成4个
    if size == 12:
        net = 'PNet'  # 网络
        tf_filenames = [os.path.join(output_dir, 'train_%s_landmark.tfrecord' % (net))]  # 文件名
        items = ['12/train_pnet_landmark.txt']  # 所用记录的txt文件
    elif size == 24:
        net = 'RNet'  # 网络
        tf_filename1 = os.path.join(output_dir, 'pos_landmark.tfrecord')
        item1 = '%d/pos_%d.txt' % (size, size)
        tf_filename2 = os.path.join(output_dir, 'part_landmark.tfrecord')
        item2 = '%d/part_%d.txt' % (size, size)
        tf_filename3 = os.path.join(output_dir, 'neg_landmark.tfrecord')
        item3 = '%d/neg_%d.txt' % (size, size)
        tf_filename4 = os.path.join(output_dir, 'landmark_landmark.tfrecord')
        item4 = '%d/landmark_%d_aug.txt' % (size, size)
        tf_filenames = [tf_filename1, tf_filename2, tf_filename3, tf_filename4]
        items = [item1, item2, item3, item4]
    elif size == 48:
        net = 'ONet'  # 网络
        tf_filename1 = os.path.join(output_dir, 'pos_landmark.tfrecord')
        item1 = '%d/pos_%d.txt' % (size, size)
        tf_filename2 = os.path.join(output_dir, 'part_landmark.tfrecord')
        item2 = '%d/part_%d.txt' % (size, size)
        tf_filename3 = os.path.join(output_dir, 'neg_landmark.tfrecord')
        item3 = '%d/neg_%d.txt' % (size, size)
        tf_filename4 = os.path.join(output_dir, 'landmark_landmark.tfrecord')
        item4 = '%d/landmark_%d_aug.txt' % (size, size)
        tf_filenames = [tf_filename1, tf_filename2, tf_filename3, tf_filename4]
        items = [item1, item2, item3, item4]

    if tf.gfile.Exists(tf_filenames[0]):
        print('tfrecords文件早已生成，无需此操作')
        return
    # 获取数据
    for tf_filename, item in zip(tf_filenames, items):
        print('开始读取数据')
        dataset = get_dataset(dataset_dir,
                              item)  # 返回值是一个列表，列表的每一个元素是一个名为data_example的字典，字典包括filename,label,bbox三个词条，其中bbox词条是一个字典。
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)  # 讲列表顺序打乱
        print('开始转换tfrecords')
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:  # 新建一个writer，写入tf_filename
            for image_example in tqdm(dataset):  # 对列表的每一个元素，
                filename = image_example['filename']  # 取出文件名
                try:
                    _add_to_tfrecord(filename, image_example, tfrecord_writer)  # 添加tfrecord记录
                except:
                    print(filename)
    print('完成转换')


def get_dataset(dir, item):
    '''利用txt的一条记录获取图片数据
    参数：
      dir：存放数据目录
      item:txt目录
    返回值：
      包含label,box，关键点的data
    '''
    dataset_dir = os.path.join(dir, item)  # 拼接得到目录地址
    imagelist = open(dataset_dir, 'r')  # 打开记录txt文件
    dataset = []
    for line in tqdm(imagelist.readlines()):  # 对此txt文件每一行
        info = line.strip().split(' ')  # 按空格切分成列表，分别为路径，标识（1是pos，0是neg，-1是part，-2是关键点），人脸框或者关键点的标签
        data_example = dict()  # 新建一个字典
        bbox = dict()  # 新建一个字典
        data_example['filename'] = info[0]  # 文件名存入字典
        data_example['label'] = int(info[1])  # 标识也存入字典
        # neg的box默认为0,part,pos的box只包含人脸框，landmark的box只包含关键点

        bbox['xmin'] = 0  # 框的位置在字典中初始化
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0

        bbox['xlefteye'] = 0  # 关键点位置在字典中初始化
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0

        if len(info) == 6:  # 长度为6则为pos,neg,part三种的数据
            bbox['xmin'] = float(info[2])  # 存入框位置数据
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:  # 长度为12则为landmark
            bbox['xlefteye'] = float(info[2])  # 存入关键点数据
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
        data_example['bbox'] = bbox  # 把bbox这个字典也当做一个部分存入data_example字典
        dataset.append(data_example)  # 把data_example字典存入dataset
    return dataset


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    '''转换成tfrecord文件
    参数：
      filename：图片文件名
      image_example:字典类型的数据，包括filename,label,bbox三个词条，其中bbox词条是一个字典。
      tfrecord_writer:写入用的writer
    '''
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _process_image_withoutcoder(filename):
    '''读取图片文件,返回String类型的图片还有高和宽'''
    image = cv2.imread(filename)
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


# 不同类型数据的转换
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example_simple(image_example, image_buffer):
    '''转换成tfrecord接受形式'''
    class_label = image_example['label']
    bbox = image_example['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'], bbox['ynose'],
                bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark)
    }))
    return example


def gen_hard_example(size):
    '''通过PNet或RNet生成下一个网络的输入'''
    batch_size = batches  # 单次喂入量
    min_face_size = min_face  # 最小识别脸的尺寸
    # 模型地址
    model_path = ['../model/PNet/', '../model/RNet/', '../model/ONet']
    if size == 12:
        net = 'PNet'
        save_size = 24  # 存储图片尺寸
    elif size == 24:
        net = 'RNet'
        save_size = 48  # 存储图片尺寸
    # 原始图片数据地址
    base_dir = 'data/WIDER_train/'
    # 处理后的图片存放地址
    data_dir = 'data/%d' % (save_size)
    neg_dir = os.path.join(data_dir, 'negative')  # neg类地址
    pos_dir = os.path.join(data_dir, 'positive')  # pos类地址
    part_dir = os.path.join(data_dir, 'part')  # part类地址
    for dir_path in [neg_dir, pos_dir, part_dir]:  # 若不存在则建立
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    detectors = [None, None, None]  # 初始化识别器列表detectors
    PNet = FcnDetector(P_Net, model_path[0])  # 创建一个单张图片识别器的实例
    detectors[0] = PNet  # 存入识别器列表
    if net == 'RNet':
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])  # 创建一个多张图片识别器的实例
        detectors[1] = RNet  # 存入识别器列表
    basedir = 'data/'  # 数据存储根目录
    filename = 'data/wider_face_train_bbx_gt.txt'  # label，人脸框标签文件。
    # 读取文件的image和box对应函数在utils中
    data = read_annotation(base_dir, filename)  # 从原始图片中读取每一个图片的具体路径和框的坐标。

    mtcnn_detector = MtcnnDetector(detectors, min_face_size=min_face_size,  # 创建一个用于生成人脸图像的实例
                                   stride=stride, threshold=thresh)
    save_path = data_dir
    save_file = os.path.join(save_path, 'detections.pkl')
    if not os.path.exists(save_file):
        # 将data制作成迭代器
        print('载入数据')
        test_data = TestLoader(data['images'])
        detectors, _ = mtcnn_detector.detect_face(test_data)
        print('完成识别')

        with open(save_file, 'wb') as f:
            pickle.dump(detectors, f, 1)
    print('开始生成图像')
    save_hard_example(save_size, data, neg_dir, pos_dir, part_dir, save_path)


# In[2]:


def save_hard_example(save_size, data, neg_dir, pos_dir, part_dir, save_path):
    '''将网络识别的box用来裁剪原图像作为下一个网络的输入'''

    im_idx_list = data['images']

    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    # save files
    neg_label_file = "data/%d/neg_%d.txt" % (save_size, save_size)
    neg_file = open(neg_label_file, 'w')

    pos_label_file = "data/%d/pos_%d.txt" % (save_size, save_size)
    pos_file = open(pos_label_file, 'w')

    part_label_file = "data/%d/part_%d.txt" % (save_size, save_size)
    part_file = open(part_label_file, 'w')
    # read detect result
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    # print(len(det_boxes), num_of_images)

    assert len(det_boxes) == num_of_images, "弄错了"

    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0

    for im_idx, dets, gts in tqdm(zip(im_idx_list, det_boxes, gt_boxes_list)):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # 转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            Iou = IOU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size),
                                    interpolation=cv2.INTER_LINEAR)

            # 划分种类
            if np.max(Iou) < 0.3 and neg_num < 60:

                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)
                neg_file.write(save_file + ' 0\n')
                if not os.path.exists(save_file):
                    cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:

                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # 偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos和part
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    if not os.path.exists(save_file):
                        cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    if not os.path.exists(save_file):
                        cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


class TestLoader:
    # 制作迭代器
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)

        self.cur = 0
        self.data = None
        self.label = None

        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.imdb)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        imdb = self.imdb[self.cur]
        im = cv2.imread(imdb)
        self.data = im


if __name__ == '__main__':
    tfrecords_type = int(input("请输入种类，PNET为1，RNET为2，ONET为3:"))
    if tfrecords_type == 1:
        gen_12x12_pic()
        print("生成三种pnet数据成功")
        gen_landmark(12)
        print("生成pnet的landmark数据成功")
        gen_pnet_imglist()
        print("四种pnet数据整合成功")
        gen_tfrecords(12)
        print("生成pnet的tfrecords文件成功")
    elif tfrecords_type == 2:
        gen_hard_example(12)
        print("生成三种rnet数据成功")
        gen_landmark(24)
        print("生成rnet的landmark数据成功")
        gen_tfrecords(24)
        print("生成rnet的tfrecords文件成功")
    elif tfrecords_type == 3:
        gen_hard_example(24)
        print("生成三种onet数据成功")
        gen_landmark(48)
        print("生成onet的landmark数据成功")
        gen_tfrecords(48)
        print("生成onet的tfrecords文件成功")
    else:
        print("输入错误，请输入1,2,3中的一个值")
