# coding: utf-8

from forwards import P_Net, R_Net, O_Net
import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import *
import score.app as app
import score.forward as forward

# 最后测试选择的网络
test_mode = 'ONet'
# 测试图片放置位置
test_dir = 'picture/'
# 测试输出位置
out_path = 'output/'
# pent对图像缩小倍数
stride = 2
# 三个网络的阈值
thresh = [0.6, 0.7, 0.7]
# 最小脸大小设定
min_face = 20
batch_size = [2048, 256, 16]


class FcnDetector:
    '''识别单张图片'''

    def __init__(self, net_factory, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            # 预测值
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            self.sess = tf.Session()
            # 重载模型
            saver = tf.train.Saver()
            model_file = tf.train.latest_checkpoint(model_path)
            saver.restore(self.sess, model_file)

    def predict(self, databatch):  # 将图片喂入网络
        height, width, _ = databatch.shape  # 尺寸
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                            feed_dict={self.image_op: databatch,
                                                       self.width_op: width,
                                                       self.height_op: height})

        return cls_prob, bbox_pred


class Detector:
    '''识别多组图片'''

    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, [None, data_size, data_size, 3])
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session()
            # 重载模型
            saver = tf.train.Saver()
            model_file = tf.train.latest_checkpoint(model_path)
            saver.restore(self.sess, model_file)
        self.data_size = data_size
        self.batch_size = batch_size

    def predict(self, databatch):
        scores = []
        batch_size = self.batch_size
        minibatch = []
        cur = 0
        # 所有数据总数
        n = databatch.shape[0]
        # 将数据整理成固定batch
        while cur < n:
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):  # 组合为一个索引序列，同时列出数据和数据下标
            m = data.shape[0]
            real_size = self.batch_size
            # 最后一组数据不够一个batch的处理
            if m < batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            cls_prob, bbox_pred, landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred, self.landmark_pred],
                                                               feed_dict={self.image_op: data})  # 传入Rnet预测

            cls_prob_list.append(cls_prob[:real_size])
            bbox_pred_list.append(bbox_pred[:real_size])
            landmark_pred_list.append(landmark_pred[:real_size])

        return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(
            landmark_pred_list, axis=0)


def py_nms(dets, thresh):
    '''剔除太相似的box'''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)

        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# In[ ]:


class MtcnnDetector:
    '''来生成人脸的图像'''

    def __init__(self, detectors,
                 min_face_size=20,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.79  # 图像金字塔的缩小率
                 ):
        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    def detect_face(self, test_data):
        all_boxes = []
        landmarks = []
        batch_idx = 0  # 计数
        num_of_img = test_data.size  # 图片数量
        empty_array = np.array([])  # 创建空列表
        for databatch in tqdm(test_data):
            batch_idx += 1
            im = databatch
            if self.pnet_detector:
                boxes, boxes_c, landmark = self.detect_pnet(im)
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            if self.rnet_detector:
                boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)

                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)

                    continue
            if self.onet_detector:

                boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)

                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)

                    continue

            all_boxes.append(boxes_c)
            landmark = [1]
            landmarks.append(landmark)
        return all_boxes, landmarks

    def detect_pnet(self, im):
        '''通过pnet筛选box和landmark
        参数：
          im:输入图像
        '''
        h, w, c = im.shape  # 高、宽、通道数
        net_size = 12  # 网络输入size
        # 网络输入尺寸和输入图像最小尺寸的比率（用于将最小的图像和网络输入尺寸对齐）
        current_scale = float(net_size) / self.min_face_size  # 12/20
        im_resized = self.processed_image(im, current_scale)  # 输入图片按上面比例缩小
        current_height, current_width, _ = im_resized.shape  # 新的尺寸
        all_boxes = list()
        # 图像金字塔
        while min(current_height, current_width) > net_size:  # 如果当前尺寸还比12大
            # 类别和box
            cls_cls_map, reg = self.pnet_detector.predict(im_resized)  # 将此图片喂入网络计算得到预测的类别和人脸框
            boxes = self.generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, self.thresh[0])  # 得到很多组预测的人脸框位置
            current_scale *= self.scale_factor  # 继续缩小图像做金字塔，直到尺寸小于等于12
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            # 非极大值抑制留下重复低的box
            keep = py_nms(boxes[:, :5], 0.5)
            boxes = boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes) == 0:  # 图片中没有人脸
            return None, None, None
        all_boxes = np.vstack(all_boxes)  # 将所有数组堆叠成一个数组
        # 将金字塔之后的box也进行非极大值抑制
        keep = py_nms(all_boxes[:, 0:5], 0.7)
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        # box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None

    def detect_rnet(self, im, dets):
        '''通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
        '''
        h, w, c = im.shape
        # 将pnet的box变成包含它的正方形，可以避免信息损失
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        # 调整超出图像的box
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        delete_size = np.ones_like(tmpw) * 20
        ones = np.ones_like(tmpw)
        zeros = np.zeros_like(tmpw)
        num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
            if tmph[i] < 20 or tmpw[i] < 20:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]  # 留下超过阈值的
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]  # 超过阈值的人脸框绝对坐标存入
            boxes[:, 4] = cls_scores[keep_inds]  # 存入置信度
            reg = reg[keep_inds]  # 存入预测的人脸框（相对裁剪的图片的坐标）
        else:
            return None, None, None

        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes, boxes_c, None

    def detect_onet(self, im, dets):
        '''将onet的选框继续筛选基本和rnet差不多但多返回了landmark'''
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)

        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:

            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None

        w = boxes[:, 2] - boxes[:, 0] + 1

        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        boxes = boxes[py_nms(boxes, 0.6)]
        keep = py_nms(boxes_c, 0.6)
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark

    def processed_image(self, img, scale):
        '''预处理数据，根据最小输入尺寸和网络输入尺寸转化图像尺度并对像素归一到[-1,1]
        '''
        height, width, channels = img.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
        img_resized = (img_resized - 127.5) / 128
        return img_resized

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
         得到对应原图的box坐标，分类分数，box偏移量
        """
        # pnet大致将图像size缩小2倍
        stride = 2

        cellsize = 12

        # 将置信度高的留下
        t_index = np.where(cls_map > threshold)

        # 没有人脸
        if t_index[0].size == 0:
            return np.array([])
        # 偏移量
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        # 对应原图的box坐标，分类分数，box偏移量
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])
        # shape[n,9]
        return boundingbox.T

    def pad(self, bboxes, w, h):
        '''将超出图像的box进行处理
        参数：
          bboxes:人脸框
          w,h:图像长宽
        返回值：
          dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
          edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
          y, x : 调整后的box在原图上左上角的坐标
          ex, ex : 调整后的box在原图上右下角的坐标
          tmph, tmpw: 原始box的长宽
        '''
        # box的长宽
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1
        # box左上右下的坐标
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        # 找到超出右下边界的box并将ex,ey归为图像的w,h
        # edx,edy为调整后的box右下角相对原box左上角的相对坐标
        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1
        # 找到超出左上角的box并将x,y归为0
        # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def calibrate_box(self, bbox, reg):
        '''校准box
        参数：
          bbox:pnet生成的box

          reg:rnet生成的box偏移值
        返回值：
          调整后的box是针对原图的绝对坐标
        '''

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    def detect(self, img):
        '''用于测试单个图像的函数'''
        boxes = None

        # pnet
        if self.pnet_detector:  # 有Pnet检测器
            boxes, boxes_c, _ = self.detect_pnet(img)  # 检测图片
            if boxes_c is None:
                return np.array([]), np.array([])

        # rnet
        if self.rnet_detector:  # 有Rnet检测器
            boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)  # 检测图片
            if boxes_c is None:
                return np.array([]), np.array([])

        # onet
        if self.onet_detector:  # 有Onet检测器
            boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)  # 检测图片
            if boxes_c is None:
                return np.array([]), np.array([])

        return boxes_c, landmark


def main():
    detectors = [None, None, None]
    # 模型放置位置
    model_path = ['model/PNet/', 'model/RNet/', 'model/ONet']
    PNet = FcnDetector(P_Net, model_path[0])  # 创建根据Pnet的检测器
    detectors[0] = PNet  # 存入列表

    if test_mode in ["RNet", "ONet"]:  # 创建根据Rnet的检测器
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    if test_mode == "ONet":  # 创建根据Onet的检测器
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face,  # 构建mtcnn模型的检测器
                                   stride=stride, threshold=thresh)
    # 选用图片还是摄像头,1是图像，2是摄像头
    input_mode = str(input("choose mode:"))
    if input_mode == '1':
        # 选用图片
        path = test_dir
        # print(path)
        for item in os.listdir(path):  # 对检测路径下的每一个图片
            img_path = os.path.join(path, item)  # 图片路径
            img = cv2.imread(img_path)  # 打开图片
            boxes_c, landmarks = mtcnn_detector.detect(img)  # 执行检测
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                score = boxes_c[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                # 截取人脸并打分
                cropped_im = img[corpbbox[0]:corpbbox[2], corpbbox[1]:corpbbox[3], :]
                resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR)
                nm_arr = resized_im.reshape([1, 128 * 128 * 3])
                # 转化为float类型
                nm_arr = nm_arr.astype(np.float32)
                # nm_arr矩阵中所有元素乘1.0/255.0，转化为0或1的形式
                img_ready = np.multiply(nm_arr, 1.0 / 255.0)
                reshaped_xs = np.reshape(img_ready, (
                    1,
                    forward.IMAGE_SIZE,
                    forward.IMAGE_SIZE,
                    forward.NUM_CHANNELS))
                preValue = app.restore_model(reshaped_xs)
                # 画置信度和颜值分数
                cv2.putText(img, '{:.2f}{}'.format(score, preValue + 2),
                            (corpbbox[0], corpbbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
            # 画关键点
            for i in range(landmarks.shape[0]):
                for j in range(len(landmarks[i]) // 2):
                    cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
            cv2.imshow('im', img)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.imwrite(out_path + item, img)
        cv2.destroyAllWindows()

    if input_mode == '2':
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_path + 'out.mp4', fourcc, 10, (640, 480))
        while True:
            t1 = cv2.getTickCount()
            ret, frame = cap.read()
            if ret == True:
                boxes_c, landmarks = mtcnn_detector.detect(frame)
                t2 = cv2.getTickCount()
                t = (t2 - t1) / cv2.getTickFrequency()
                fps = 1.0 / t
                for i in range(boxes_c.shape[0]):
                    bbox = boxes_c[i, :4]
                    score = boxes_c[i, 4]
                    corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                    # 画人脸框
                    cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                                  (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                    # 截取人脸并打分
                    cropped_im = frame[corpbbox[0]:corpbbox[2], corpbbox[1]:corpbbox[3], :]
                    resized_im = cv2.resize(cropped_im, (128, 128), interpolation=cv2.INTER_LINEAR)
                    nm_arr = resized_im.reshape([1, 128 * 128 * 3])
                    # 转化为float类型
                    nm_arr = nm_arr.astype(np.float32)
                    # nm_arr矩阵中所有元素乘1.0/255.0，转化为0或1的形式
                    img_ready = np.multiply(nm_arr, 1.0 / 255.0)
                    reshaped_xs = np.reshape(img_ready, (
                        1,
                        forward.IMAGE_SIZE,
                        forward.IMAGE_SIZE,
                        forward.NUM_CHANNELS))
                    preValue = app.restore_model(reshaped_xs)
                    # 画置信度和颜值分数
                    cv2.putText(frame, '{:.2f}{}'.format(score, preValue + 2),
                                (corpbbox[0], corpbbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)
                    # 画fps值
                cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                # 画关键点
                for i in range(landmarks.shape[0]):
                    for j in range(len(landmarks[i]) // 2):
                        cv2.circle(frame, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
                a = out.write(frame)
                cv2.imshow("result", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
