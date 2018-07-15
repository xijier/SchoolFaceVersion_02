import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import *
import threading
import time
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QHBoxLayout,  QVBoxLayout, QGridLayout, QFormLayout, QLineEdit, QTextEdit
from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import imutils
from rectangleDrawThread import rectangleThread
# from PIL import Image
# from pypinyin import pinyin, lazy_pinyin
# import pypinyin
import cv2

class VideoBox(QWidget):
    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1

    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    video_url = ""
    progress = 0

    def __init__(self, video_url="", video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        super(VideoBox, self).__init__()
        self.createGridGroupBox()
        self.creatVboxGroupBox()
        self.preTime = 0
        mainLayout = QVBoxLayout()
        hboxLayout = QHBoxLayout()
        hboxLayout.addStretch()
        hboxLayout.addWidget(self.gridGroupBox)
        hboxLayout.addWidget(self.vboxGroupBox)
        mainLayout.addLayout(hboxLayout)
        self.setLayout(mainLayout)
        self.threshold = [0.6, 0.6, 0.7]

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

    def initNet(self,Pnet,Rnet,Onet,lock):
        self.Pnet=Pnet
        self.Rnet = Rnet
        self.Onet = Onet
        self.lock = lock

    def createGridGroupBox(self):
        self.gridGroupBox = QGroupBox("Grid layout")
        layout = QGridLayout()
        self.pictureLabel = QLabel()
        init_image = QPixmap("1.png").scaled(1000, 700)
        self.pictureLabel.setPixmap(init_image)
        self.threadId = 0
        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.switch_video)

        control_box = QHBoxLayout()
        control_box.setContentsMargins(0, 0, 0, 0)
        control_box.addWidget(self.playButton)
        llayout = QVBoxLayout()
        llayout.addWidget(self.pictureLabel)
        llayout.addLayout(control_box)
        layout = QHBoxLayout()
        layout.addLayout(llayout)
        self.gridGroupBox.setLayout(layout)
        self.setWindowTitle('Basic Layout')

    def creatVboxGroupBox(self):
        self.vboxGroupBox = QGroupBox("Vbox layout")
        layout = QVBoxLayout()
        init_image = QPixmap("1.png").scaled(200, 200)
        self.imgeLabel_0 = QLabel()
        self.imgeLabel_0.setPixmap(init_image)
        self.textbox = QLineEdit(self)
        self.textbox.setText('Name')
        self.imgeLabel_1 = QLabel()
        self.imgeLabel_1.setPixmap(init_image)
        self.imgeLabel_2 = QLabel()
        self.imgeLabel_2.setPixmap(init_image)

        layout.addWidget(self.imgeLabel_0)
        layout.addWidget(self.textbox)
        # layout.addWidget(self.imgeLabel_1)
        # layout.addWidget(self.imgeLabel_2)
        self.vboxGroupBox.setLayout(layout)

    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = VideoBox.STATUS_INIT
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

    def play(self):
        if self.video_url == "" or self.video_url is None:
            return
        if not self.playCapture.isOpened():
            self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.status = VideoBox.STATUS_PLAYING

    def stop(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.playCapture.isOpened():
            self.timer.stop()
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.status = VideoBox.STATUS_PAUSE

    def re_play(self):
        if self.video_url == "" or self.video_url is None:
            return
        self.playCapture.release()
        self.playCapture.open(self.video_url)
        self.timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.status = VideoBox.STATUS_PLAYING

    def show_video_images(self):

        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            if success:
                start = time.time()
                frame = imutils.resize(frame, width=1000)
                thread1 =rectangleThread(self.threadId, frame, self.Pnet, self.Rnet, self.Onet, lock, self.imgeLabel_0,\
                                         self.textbox, self.imgeLabel_1, self.imgeLabel_2)
                self.threadId = self.threadId + 1
                thread1.start()
                end = time.time()
                height, width = frame.shape[:2]
                if frame.ndim == 3:
                    rgb = cvtColor(frame, COLOR_BGR2RGB)
                elif frame.ndim == 2:
                    rgb = cvtColor(frame, COLOR_GRAY2BGR)

                temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                self.pictureLabel.setPixmap(temp_pixmap)

            else:
                print("read failed, no frame data")
                success, frame = self.playCapture.read()
                if not success and self.video_type is VideoBox.VIDEO_TYPE_OFFLINE:
                    print("play finished")  # 判断本地文件播放完毕
                    self.reset()
                    self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                return
        else:
            print("open file or capturing device error, init again")
            self.reset()

    def update_timer(self):
        while (True):
            if self.status is VideoBox.STATUS_PLAYING:
                self.progress = self.progress + 1
                time.sleep(0.4)
                if (self.progress == 15):
                    self.progress = 0

    def switch_video(self):
        if self.video_url == "" or self.video_url is None:
            return
        if self.status is VideoBox.STATUS_INIT:
            self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif self.status is VideoBox.STATUS_PLAYING:
            self.timer.stop()
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.release()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        elif self.status is VideoBox.STATUS_PAUSE:
            if self.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                self.playCapture.open(self.video_url)
            self.timer.start()
            self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        self.status = (VideoBox.STATUS_PLAYING,
                       VideoBox.STATUS_PAUSE,
                       VideoBox.STATUS_PLAYING)[self.status]

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
    img = cv2.imread('1.png')
    scale_img = cv2.resize(img, (100, 100))
    input = scale_img.reshape(1, *scale_img.shape)
    Pnet.predict(input)
    img = cv2.imread('1.png')
    scale_img = cv2.resize(img, (24, 24))
    input = scale_img.reshape(1, *scale_img.shape)
    Rnet.predict(input)
    img = cv2.imread('1.png')
    scale_img = cv2.resize(img, (48, 48))
    input = scale_img.reshape(1, *scale_img.shape)
    Onet.predict(input)
    return Pnet,Rnet,Onet

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    mapp = QApplication(sys.argv)
    Pnet, Rnet, Onet = loadNet()

    mw = VideoBox()
    lock = threading.Lock()
    mw.initNet(Pnet,Rnet,Onet,lock)
    mw.set_video("east.mp4", VideoBox.VIDEO_TYPE_OFFLINE, False)
    mw.show()

    sys.exit(mapp.exec_())