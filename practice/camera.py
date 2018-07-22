#-*- coding: utf-8 -*-

import cv2
import sys
from PIL import Image
import numpy as np
import dlib
import time
import threading

class myDetect():
    def __init__(self):
        self.name = "1"
        self.detector = dlib.get_frontal_face_detector()
    def CatchUsbVideo(self,window_name, camera_idx):
        cv2.namedWindow(window_name)
        # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头

        #cap = cv2.VideoCapture(camera_idx)
        cap = cv2.VideoCapture("../src/east.mp4")

        while cap.isOpened():
            ok, frame = cap.read()  # 读取一帧数据
            if not ok:
                break
            start = time.time()
            #frame = cv2.resize(frame, (1280,720))
            self.thread_it(self.music, frame)
            # dets, scores, idx = detector.run(frame, 0)
            # print(time.time()- start)
            # for i, d in enumerate(dets):  # 依次区分截图中的人脸
            #     x1 = d.top() if d.top() > 0 else 0
            #     y1 = d.bottom() if d.bottom() > 0 else 0
            #     x2 = d.left() if d.left() > 0 else 0
            #     y2 = d.right() if d.right() > 0 else 0
            #     frame = cv2.rectangle(frame, (x2, x1), (y2, y1), (255, 0, 0), 2)  # 人脸画框
            frame = cv2.resize(frame, (1027, 768))
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(10)
            if c & 0xFF == ord('q'):
                break

                # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()
    def music(self, frame):
        start  = time.time()
        dets, scores, idx = self.detector.run(frame)

        print(time.time() - start)
        # for i, d in enumerate(dets):  # 依次区分截图中的人脸
        #     x1 = d.top() if d.top() > 0 else 0
        #     y1 = d.bottom() if d.bottom() > 0 else 0
        #     x2 = d.left() if d.left() > 0 else 0
        #     y2 = d.right() if d.right() > 0 else 0
        #     frame = cv2.rectangle(frame, (x2, x1), (y2, y1), (255, 0, 0), 2)  # 人脸画框
        #     crop_img = frame[x1:y1, x2:y2]
        #     cv2.imwrite('temp/' + str(time.time()) + '.jpg', crop_img)
        #print(scores)

    # 打包进线程（耗时的操作）
    @staticmethod
    def thread_it(func, *args):
        t = threading.Thread(target=func, args=args)
        t.setDaemon(True)  # 守护
        t.start()  # 启动

if __name__ == '__main__':

    detect = myDetect()
    detect.CatchUsbVideo("截取视频流", 0)