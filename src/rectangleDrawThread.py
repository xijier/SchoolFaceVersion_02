from PyQt5.QtGui import *
from cv2 import *
import threading
import time
from PyQt5.QtGui import QPixmap
import tools_matrix as tools
import numpy as np
import imutils
import cv2
import facenet
import tensorflow as tf
import os
import pickle

class rectangleThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID,frame,Pnet,Rnet,Onet,lock,imgeLabel_0, text_box, imgeLabel_1,imgeLabel_2):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.frame = frame
        self.preTime = 0
        self.Pnet = Pnet
        self.Rnet = Rnet
        self.Onet = Onet
        self.lock = lock
        self.threshold = [0.6, 0.6, 0.7]
        self.textbox = text_box
        self.imgeLabel_0 =imgeLabel_0
        self.imgeLabel_1 = imgeLabel_1
        self.imgeLabel_2 = imgeLabel_2

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        self.lock.acquire()
        rectangles = self.detectFace(self.frame, self.threshold)
        frame = self.rectangleDraw(rectangles, self.frame)
        self.lock.release()

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def crop(self, image, random_crop, image_size):
        # if image.shape[1] > image_size:
            # sz1 = int(image.shape[1] // 2)
            # sz2 = int(image_size // 2)
            # # if random_crop:
            # #     diff = sz1 - sz2
            # #     (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
            # # else:
            # #     (h, v) = (0, 0)
            # image = image[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
        image = cv2.resize(image, (160, 160))
        return image

    def flip(self, image, random_flip):
        if random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image
    # to identify one pic
    def recognizeFace(self, image):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model('models/20180408-102900')
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                emb_array = np.zeros((1, embedding_size))
                images = np.zeros((1, 160, 160, 3))
                img = self.prewhiten(image)
                img = self.crop(img, False, 160)
                img = self.flip(img, False)
                images[0, :, :, :] = img
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[0:1, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename_exp = os.path.expanduser('2018zhongzhuanv2.pkl')

                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                resultText = class_names[best_class_indices[0]]
                print("resultText is : %s" % resultText)
                self.textbox.setText(resultText)


    def detectFace(self,img, threshold):

        caffe_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, ch = caffe_img.shape
        scales = tools.calculateScales(img)
        out = []
        t0 = time.time()
        # del scales[:4]

        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(caffe_img, (ws, hs))
            input = scale_img.reshape(1, *scale_img.shape)
            ouput = self.Pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
            out.append(ouput)
        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            cls_prob = out[i][0][0][:, :,
                       1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
            roi = out[i][1][0]
            out_h, out_w = cls_prob.shape
            out_side = out_w
            if out_h>out_w:
                out_side = out_h
            #out_side = max(out_h, out_w)
            # print('calculating img scale #:', i)
            cls_prob = np.swapaxes(cls_prob, 0, 1)
            roi = np.swapaxes(roi, 0, 2)
            rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            rectangles.extend(rectangle)
        rectangles = tools.NMS(rectangles, 0.7, 'iou')

        t1 = time.time()
        print('time for 12 net is: ', t1 - t0)

        if len(rectangles) == 0:
            return rectangles

        crop_number = 0
        out = []
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)
            crop_number += 1

        predict_24_batch = np.array(predict_24_batch)

        out = self.Rnet.predict(predict_24_batch)

        cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
        cls_prob = np.array(cls_prob)  # convert to numpy
        roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
        roi_prob = np.array(roi_prob)
        rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        t2 = time.time()
        print('time for 24 net is: ', t2 - t1)

        if len(rectangles) == 0:
            return rectangles

        crop_number = 0
        predict_batch = []
        for rectangle in rectangles:
            # print('calculating net 48 crop_number:', crop_number)
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)
            crop_number += 1

        predict_batch = np.array(predict_batch)

        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]  # index
        rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        t3 = time.time()
        print('time for 48 net is: ', t3 - t2)

        return rectangles

    def rectangleDraw(self , rectangles, img):
        draw = img.copy()
        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                           int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue
                cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                              (255, 0, 0), 1)
                self.recognizeFace(imutils.resize(crop_img, width=160))
                crop_img = imutils.resize(crop_img, width=100)
                height, width = crop_img.shape[:2]
                temp_image = QImage(crop_img.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                #加消息队列线程实现图片更新
                self.imgeLabel_0.setPixmap(temp_pixmap)
                #cv2.imwrite('data/' + str(self.threadID) + 'test.jpg', crop_img)
        return draw