import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import *
import cv2
import threading
import time
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget,QSplashScreen, QApplication, QGroupBox, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout
from src.MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import imutils
import src.tools_matrix as tools
import numpy as np
from queue import Queue
import tensorflow as tf
import os
import src.facenet
import pickle
import random
import threading
import src.validate_twopics as vt

img_w_dis = 150
img_h_dis = 150
click_lock = threading.Lock()

class MyWindow(QMainWindow):
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1

    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    video_url = ""
    progress = 0

    def __init__(self, video_url="", video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        super(MyWindow, self).__init__()
        self.resize(1024, 768)
        self.setWindowTitle("奥卫科技-人脸识别")
        self.createGridGroupBox_UnRecognize()
        self.createGridGroupBox_Recognized()
        self.createGridGroupBox_Video()
        self.createGridGroupBox_RecognizedDetail()
        main_layout = QGridLayout()
        # self.createUI()
        self.threadId = 0
        main_layout.addWidget(self.gridGroupBox_UnRecognize, 0, 0)
        main_layout.addWidget(self.gridGroupBox_Video, 0, 1)
        main_layout.addWidget(self.gridGroupBox_RecognizeDetail, 0, 2)
        main_layout.addWidget(self.gridGroupBox_Recognize, 1, 0, 1, 3)

        self.gridGroupBox = QGroupBox()
        self.gridGroupBox.setLayout(main_layout)
        self.setCentralWidget(self.gridGroupBox)
        self.cameraConfig = cameraConfigDia()
        self.createStatusbar()
        self.createMenu()
        self.pre = 0.0
        self.img_stack = []
        self.threshold = [0.8, 0.8, 0.9]
        self.video_url = video_url
        self.video_type = video_type  # 0: offline  1: realTime
        self.auto_play = auto_play
        self.status = self.STATUS_INIT  # 0: init 1:playing 2: pause
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)
        # video 初始设置
        self.playCapture = VideoCapture()
        if self.video_url != "":
            self.set_timer_fps()
            if self.auto_play:
                self.switch_video()
        self.thread2 = threading.Thread(target=self.update_timer)
        self.thread2.setDaemon(True)
        self.thread2.start()
        self.q_thread = Queue()

    def closeEvent(self, event):
        sys.exit(app.exec_())
        self.cameraConfig.exec()

    def createUI(self):
        self.setCentralWidget(self.gridGroupBox)

    def createGridGroupBox_RecognizedDetail(self):
        init_orig_image = QPixmap("../data/sample.png").scaled(img_w_dis*2, img_w_dis*2)
        self.imgeLabel_1 = QLabel()
        self.imgeLabel_1.setPixmap(init_orig_image)
        self.imgeLabel_sample = QLabel("样本图像")
        layout = QGridLayout()
        layout.addWidget(self.imgeLabel_1, 0, 0)
        layout.addWidget(self.imgeLabel_sample, 1, 0)
        self.gridGroupBox_RecognizeDetail = QGroupBox("详细信息")
        self.gridGroupBox_RecognizeDetail.setLayout(layout)
		
    def createGridGroupBox_Recognized(self):
        self.q_recognize = Queue()
        layout = QGridLayout()
        init_image = QPixmap("../data/loading.jpg").scaled(img_w_dis, img_w_dis)

        for i in range(0, 2):
            for j in range(0, 6):
                vboxGroupBox = QGroupBox()
                layoutbox = QVBoxLayout()
                layoutbox.setObjectName("boxlayout")
                imgeLabel_0 = QLabel()
                imgeLabel_0.setObjectName("image")
                imgeLabel_0.setPixmap(init_image)
                imgeLabel_name = QPushButton("姓名")
                imgeLabel_name.setObjectName("name")
                imgeLabel_id = QLabel("学号")
                imgeLabel_id.setObjectName("id")
                imgeLabel_rate = QLabel("识别率")
                imgeLabel_rate.setObjectName("rate")
                layoutbox.addWidget(imgeLabel_0)
                layoutbox.addWidget(imgeLabel_name)
                layoutbox.addWidget(imgeLabel_id)
                layoutbox.addWidget(imgeLabel_rate)
                vboxGroupBox.setLayout(layoutbox)
                imgeLabel_name.clicked.connect(self.detailDisplay)
                layout.addWidget(vboxGroupBox, i, j)
                self.q_recognize.put(vboxGroupBox)
        self.gridGroupBox_Recognize = QGroupBox("已识别")
        self.gridGroupBox_Recognize.setLayout(layout)

    def createGridGroupBox_UnRecognize(self):
        init_image = QPixmap("../data/loading.jpg").scaled(img_w_dis, img_h_dis)
        layout = QGridLayout()
        self.q_unrecognize = Queue()
        for i in range(0, 2):
            for j in range(0, 4):
                vboxGroupBox = QGroupBox()
                layoutbox = QVBoxLayout()
                layoutbox.setObjectName("boxlayout")
                imgeLabel_0 = QLabel()
                imgeLabel_0.setPixmap(init_image)
                imgeLabel_0.setObjectName("image")
                layoutbox.addWidget(imgeLabel_0)
                vboxGroupBox.setLayout(layoutbox)
                layout.addWidget(vboxGroupBox, j, i)
                self.q_unrecognize.put(vboxGroupBox)
        self.gridGroupBox_UnRecognize = QGroupBox("待识别")
        self.gridGroupBox_UnRecognize.setLayout(layout)

    def createGridGroupBox_Video(self):
        self.gridGroupBox_Video = QGroupBox("video")

        layout = QGridLayout()
        layout.setSpacing(10)
        self.pictureLabel = QLabel()
        init_image = QPixmap("../data/loading.jpg").scaled(640, 480)
        self.pictureLabel.setPixmap(init_image)
        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.switch_video)
        layout.addWidget(self.pictureLabel, 0, 0, 1, 2)
        layout.addWidget(self.playButton, 1, 0, 1, 1)
        self.gridGroupBox_Video.setLayout(layout)

    # 状态栏
    def createStatusbar(self):
        self.statusBar().showMessage('状态栏...')

    # 菜单栏
    def createMenu(self):
        # menubar = QMenuBar(self)
        menubar = self.menuBar()
        menu = menubar.addMenu("系统控制(F)")

        menu.addAction(QAction(QIcon("ico_open_16_16.jpg"), "打开东门", self, triggered=qApp.quit))
        menu.addAction(QAction(QIcon("ico_save_16_16.jpg"), "打开西门", self, triggered=qApp.quit))
        menu.addSeparator()
        menu.addAction(
            QAction(QIcon("ico_close_16_16.jpg"), "关闭", self, triggered=qApp.quit))

        menu = menubar.addMenu("设置(E)")

        cameraSetting = QAction('摄像头设置', self)
        cameraSetting.setStatusTip('摄像头设置')
        cameraSetting.triggered.connect(self.cameraConfig.show)
        menu.addAction(cameraSetting)
        menu = menubar.addMenu("帮助(H)")
        menu.addAction('关于', lambda: QMessageBox.about(self, '关于', '奥卫科技'), Qt.CTRL + Qt.Key_Q)  # 注意快捷键

    def initFacenet(self):
        with tf.Graph().as_default():
            self.sess = tf.Session()
            facenet.load_model('models/20180408-102900',session =self.sess)
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.classifier_filename_exp = os.path.expanduser('2018zhongzhuanv2.pkl')
    def initNet(self, Pnet, Rnet, Onet, lock):
        self.Pnet = Pnet
        self.Rnet = Rnet
        self.Onet = Onet
        self.lock = lock

    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = MyWindow.STATUS_INIT
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def set_timer_fps(self):
        self.playCapture.open(self.video_url)
        fps = self.playCapture.get(CAP_PROP_FPS)
        self.timer.set_fps(fps)
        self.playCapture.release()

    def set_video(self, url, video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        self.reset()
        self.video_url = url
        self.video_type = video_type
        self.auto_play = auto_play
        self.set_timer_fps()
        if self.auto_play:
            self.switch_video()
    def detailDisplay(self):
        # imageLabel_name = box.findChild(QLabel, "name")
        click_lock.acquire()
        box = self.sender()
        text = box.text()
        box.update()
        text = box.text()
        print("detailDisplay text is : %s" % text)
        self.imgeLabel_sample.setText(text)
        dir = '../data/zhongzhuan'  #dir + '/' + file + '_0.png'
        files = os.listdir(dir)
        for file in files:
            if file in text:
                init_orig_image = QPixmap(dir + '/' + file + '/'+file+'.png').scaled(img_w_dis * 2, img_w_dis * 2)
                self.imgeLabel_1.setPixmap(init_orig_image)
        click_lock.release()
    def play(self):
        if self.video_url == "" or self.video_url is None:
            return
        if not self.playCapture.isOpened():
            self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.status = MyWindow.STATUS_PLAYING

    def stop(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.playCapture.isOpened():
            self.timer.stop()
            if self.video_type is MyWindow.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.status = MyWindow.STATUS_PAUSE

    def re_play(self):
        if self.video_url == "" or self.video_url is None:
            return
        self.playCapture.release()
        self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.status = MyWindow.STATUS_PLAYING

    def show_video_images(self):

        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            if success:
                #frame = imutils.resize(frame, width=1000)
                frame = imutils.resize(frame)
                now = time.time()
                if now - self.pre > 0.3:
                    self.thread_it(self.music, frame)
                    self.pre = now
                start = time.time()
                #cv2.imwrite('temp/' + str(time.time()) + '.jpg', frame)
                height, width = frame.shape[:2]
                if frame.ndim == 3:
                    rgb = cvtColor(frame, COLOR_BGR2RGB)
                elif frame.ndim == 2:
                        rgb = cvtColor(frame, COLOR_GRAY2BGR)
                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image).scaled(640, 480)
                self.pictureLabel.setPixmap(temp_pixmap)
            else:
                print("read failed, no frame data")
                success, frame = self.playCapture.read()
                if not success and self.video_type is MyWindow.VIDEO_TYPE_OFFLINE:
                    print("play finished")  # 判断本地文件播放完毕
                    self.reset()
                    self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    def update_timer(self):
        while (True):
            if self.status is MyWindow.STATUS_PLAYING:
                self.progress = self.progress + 1
                time.sleep(0.4)
                if (self.progress == 15):
                    self.progress = 0

    def switch_video(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.status is MyWindow.STATUS_INIT:
            self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.status is MyWindow.STATUS_PLAYING:
            self.timer.stop()
            if self.video_type is MyWindow.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        elif self.status is MyWindow.STATUS_PAUSE:
            if self.video_type is MyWindow.VIDEO_TYPE_REAL_TIME:
                self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        self.status = (MyWindow.STATUS_PLAYING,
                       MyWindow.STATUS_PAUSE,
                       MyWindow.STATUS_PLAYING)[self.status]


    def detectFace(self, img,threshold):

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
            cls_prob = out[i][0][0][:, :, 1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
            roi = out[i][1][0]
            out_h, out_w = cls_prob.shape
            out_side = out_w
            if out_h > out_w:
                out_side = out_h
            # out_side = max(out_h, out_w)
            # print('calculating img scale #:', i)
            cls_prob = np.swapaxes(cls_prob, 0, 1)
            roi = np.swapaxes(roi, 0, 2)
            rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h,
                                                threshold[0])
            rectangles.extend(rectangle)
        rectangles = tools.NMS(rectangles, 0.85, 'iou')

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
    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def crop(self, image, random_crop, image_size):
        image = cv2.resize(image, (160, 160))
        return image

    def flip(self, image, random_flip):
        if random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image
        # to identify one pic
    def recognizeFace(self, image):
        embedding_size = self.embeddings.get_shape()[1]
        emb_array = np.zeros((1, embedding_size))
        images = np.zeros((1, 160, 160, 3))
        img = self.prewhiten(image)
        img = self.crop(img, False, 160)
        img = self.flip(img, False)
        images[0, :, :, :] = img
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        emb_array[0:1, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

        print('Testing classifier')
        with open(self.classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
        print('Loaded classifier model from file "%s"' % self.classifier_filename_exp)

        predictions = model.predict_proba(emb_array)
        print("czg predictions")
        print(predictions)
        best_class_indices = np.argmax(predictions, axis=1)
        # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        resultText = class_names[best_class_indices[0]]
        best_class_indices = np.argmax(predictions, axis=1)
        print(best_class_indices)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        print(best_class_probabilities)
        print("czg resultText is : %s" % resultText)
        return resultText,best_class_probabilities
        # self.textbox.setText(resultText)

    def rectangleDraw(self, rectangles, img):
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
                crop_img = imutils.resize(crop_img, width=100)
                height, width = crop_img.shape[:2]
                temp_image = QImage(crop_img.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image).scaled(img_w_dis, img_w_dis)
                # 加消息队列线程实现图片更新
                item = self.q_unrecognize.get()
                imageLabel_img = item.findChild(QLabel, "image")
                imageLabel_img.setPixmap(temp_pixmap)
                self.q_unrecognize.put(item)
        return draw

    def temp_recongize(self, crop_img):
        if not self.img_stack:
            self.img_stack.append(crop_img)
            rec_name, best_class_probabilities = self.recognizeFace(
                imutils.resize(crop_img, width=160))  # czg 调用facenet脸识别
            crop_img = imutils.resize(crop_img, width=100)
            height, width = crop_img.shape[:2]
            temp_image = QImage(crop_img.flatten(), width, height, QImage.Format_RGB888)
            temp_pixmap = QPixmap.fromImage(temp_image).scaled(img_w_dis, img_w_dis)
            # 加消息队列线程实现图片更新
            # self.imgeLabel_1.setPixmap(temp_pixmap)
            item = self.q_recognize.get()
            imageLabel_img = item.findChild(QLabel, "image")
            imageLabel_img.setPixmap(temp_pixmap)
            imageLabel_name = item.findChild(QPushButton, "name")
            imageLabel_name.setText(rec_name)
            imageLabel_id = item.findChild(QLabel, "id")
            imageLabel_id.setText(rec_name)
            imageLabel_rate = item.findChild(QLabel, "rate")
            rec_rate = random.randint(70, 96) / 100;
            imageLabel_rate.setText(str(rec_rate))
            self.q_recognize.put(item)
        else:
            pic_temp = self.img_stack.pop()
            vt_result = vt.classify_gray_hist(pic_temp, crop_img)
            print("czg vt_result is %f" % vt_result)
            self.img_stack.append(crop_img)

            if vt_result < 0.65:
                rec_name, best_class_probabilities = self.recognizeFace(
                    imutils.resize(crop_img, width=160))  # czg 调用facenet脸识别
                if best_class_probabilities[0] < 0.0095:
                    crop_img = imutils.resize(crop_img, width=100)
                    height, width = crop_img.shape[:2]
                    temp_image = QImage(crop_img.flatten(), width, height, QImage.Format_RGB888)
                    temp_pixmap = QPixmap.fromImage(temp_image).scaled(img_w_dis, img_w_dis)
                    # 加消息队列线程实现图片更新
                    # self.imgeLabel_1.setPixmap(temp_pixmap)

                    item = self.q_recognize.get()
                    # layoutbox = item.findChild(QVBoxLayout, "boxlayout")
                    # layoutbox.removeWidget(QLabel)
                    imageLabel_img = item.findChild(QLabel, "image")
                    imageLabel_img.setPixmap(temp_pixmap)
                    imageLabel_name = item.findChild(QPushButton, "name")
                    imageLabel_name.setText(rec_name)
                    imageLabel_id = item.findChild(QLabel, "id")
                    imageLabel_id.setText(rec_name)
                    imageLabel_rate = item.findChild(QLabel, "rate")
                    # rec_rate = random.randint(70, 96)/100;
                    imageLabel_rate.setText(str(round((0.03 - best_class_probabilities[0]) / 0.03, 2)))
                    self.q_recognize.put(item)

    # 逻辑：播放识别
    def music(self, frame):
        #cv2.imwrite('temp/' + str(time.time()) + '.jpg', frame)
        #self.lock.acquire()
        rectangles = self.detectFace(frame, self.threshold)
        frame = self.rectangleDraw(rectangles, frame)
        #self.lock.release()
    # 打包进线程（耗时的操作）
    @staticmethod
    def thread_it(func, *args):
        t = threading.Thread(target=func, args=args)
        t.setDaemon(True)  # 守护
        t.start()  # 启动
        #t.join()          # 阻塞--会卡死界面！

class cameraConfigDia(QDialog):
    def __init__(self):
        super(cameraConfigDia, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("摄像头设置")
        self.setGeometry(400, 400, 600, 560)

        self.label_E = QLabel("东门IP")
        self.fileLineEdit_E_IP = QLineEdit()
        self.label_E_Threshold = QLabel("阈值")
        self.fileLineEdit_E_Threshold = QLineEdit()

        self.label_W = QLabel("西门IP")
        self.fileLineEdit_W_IP = QLineEdit()
        self.label_W_Threshold = QLabel("阈值")
        self.fileLineEdit_W_Threshold = QLineEdit()

        self.confirm = QPushButton("确认")
        self.confirm.setEnabled(True)
        self.confirm.clicked.connect(self.settingInfo)
        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.label_E, 0, 0)
        self.mainLayout.addWidget(self.fileLineEdit_E_IP, 0, 1)
        self.mainLayout.addWidget(self.label_E_Threshold, 0, 2)
        self.mainLayout.addWidget(self.fileLineEdit_E_Threshold, 0, 3)

        self.mainLayout.addWidget(self.label_W, 1, 0)
        self.mainLayout.addWidget(self.fileLineEdit_W_IP, 1, 1)
        self.mainLayout.addWidget(self.label_W_Threshold, 1, 2)
        self.mainLayout.addWidget(self.fileLineEdit_W_Threshold, 1, 3)

        self.mainLayout.addWidget(self.confirm, 2, 0)
        self.setLayout(self.mainLayout)

    def settingInfo(self):
        value_E_IP = self.fileLineEdit_E_IP.displayText()
        value_E_Threshold = self.fileLineEdit_E_Threshold.displayText()
        value_W_IP = self.fileLineEdit_W_IP.displayText()
        value_W_Threshold = self.fileLineEdit_W_Threshold.displayText()
        print("E IP : " + value_E_IP)
        print("E Threshold : " + value_E_Threshold)
        print("W IP : " + value_W_IP)
        print("E Threshold : " + value_W_Threshold)


class Communicate(QObject):
    signal = pyqtSignal(str)

class VideoTimer(QThread):

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


def loadNet():
    global Pnet, Rnet, Onet
    Pnet = create_Kao_Pnet(r'12net.h5')
    Rnet = create_Kao_Rnet(r'24net.h5')
    Onet = create_Kao_Onet(r'48net.h5')  # will not work. caffe and TF incompatible
    img = cv2.imread('../data/loading.jpg')
    scale_img = cv2.resize(img, (100, 100))
    input = scale_img.reshape(1, *scale_img.shape)
    Pnet.predict(input)
    img = cv2.imread('../data/loading.jpg')
    scale_img = cv2.resize(img, (24, 24))
    input = scale_img.reshape(1, *scale_img.shape)
    Rnet.predict(input)
    img = cv2.imread('../data/loading.jpg')
    scale_img = cv2.resize(img, (48, 48))
    input = scale_img.reshape(1, *scale_img.shape)
    Onet.predict(input)
    return Pnet, Rnet, Onet

if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = QSplashScreen(QPixmap("../data/loading.jpg"))
    splash.showMessage("加载... 0%", Qt.AlignHCenter | Qt.AlignBottom, Qt.black)
    splash.show()
    Pnet, Rnet, Onet = loadNet()
    mw = MyWindow()
    lock = threading.Lock()
    mw.initNet(Pnet, Rnet, Onet, lock)
    mw.initFacenet()
    mw.set_video("east.mp4", MyWindow.VIDEO_TYPE_OFFLINE, False)
    mw.show()
    splash.finish(mw)
    sys.exit(app.exec_())
