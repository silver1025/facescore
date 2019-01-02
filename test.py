# coding: utf-8

from forwards import P_Net, R_Net, O_Net
import cv2
import random
import numpy as np
from app import Detector, FcnDetector, MtcnnDetector
from utils import *

# 测试图片放置位置
pic_dir = 'data/WIDER_val/images'
# 测试图片标签文件路径
label_dir = 'data/wider_face_split/wider_face_val_bbx_gt.txt'
# 模型放置位置
model_path = ['model/PNet/', 'model/RNet/', 'model/ONet']
# 最后测试选择的网络
test_mode = 'ONet'
# pent对图像缩小倍数
stride = 2
# 三个网络的阈值
thresh = [0.65, 0.75, 0.75]
# 最小脸大小设定
min_face = 20
batch_size = [2048, 256, 16]
# 误检数
test_num = 1000


def main():
    detectors = [None, None, None]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face,
                                   stride=stride, threshold=thresh)
    imagelist = open(label_dir, 'r')
    dataset = []
    contents = imagelist.readlines()
    pos = 0
    fail_num = 0
    success_num = 0
    real_num = 0
    while pos < len(contents):
        img_path = os.path.join(pic_dir, contents[pos].strip())
        pos += 1
        face_num = int(contents[pos])
        pos += 1
        bboxes = np.zeros((face_num, 4), dtype=np.int16)
        for i in range(face_num):
            info = contents[pos].strip().split(' ')
            pos += 1
            bboxes[i:i + 1, :] = [int(info[0]), int(info[1]), int(info[0]) + int(info[2]), int(info[1]) + int(info[3])]
        data = [img_path, bboxes]
        dataset.append(data)
    random.shuffle(dataset)
    pos = 0
    while pos < len(dataset) and fail_num < test_num:
        data = dataset[pos]
        pos += 1
        img_path, bboxes = data
        face_num = bboxes.shape[0]
        real_num += face_num
        img = cv2.imread(img_path)
        boxes_pre, landmarks = mtcnn_detector.detect(img)
        for i in range(boxes_pre.shape[0]):
            box_pre = boxes_pre[i, :4]
            cropbbox = [int(box_pre[0]), int(box_pre[1]), int(box_pre[2]), int(box_pre[3])]
            # 计算iou值
            iou = IOU(cropbbox, bboxes)
            if np.max(iou) < 0.5:
                fail_num += 1
            else:
                success_num += 1
        print("现在误检数为%d,成功检验数为%d,人脸总数为%d\r" % (fail_num, success_num, real_num))

    recall = (success_num + 0.0) / real_num
    accuracy = (success_num + 0.0) / (success_num + fail_num)
    print("%d误检时的召回率为：%.4f, 正确率为：%.4f" % (fail_num, recall, accuracy))


if __name__ == '__main__':
    main()
