#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 15:24
# @Author  : He Hangjiang
# @Site    :
# @File    : compare.py
# @Software: PyCharm

import requests
from json import JSONDecoder
import cv2

def drawFace(face_rectangle,img):
    width = face_rectangle['width']
    top = face_rectangle['top']
    left = face_rectangle['left']
    height = face_rectangle['height']
    start = (left, top)
    end = (left + width, top + height)
    color = (55, 255, 155)
    thickness = 3
    cv2.rectangle(img, start, end, color, thickness)

compare_url = "https://api-cn.faceplusplus.com/facepp/v3/compare"
key = "2PX56nNhKpVd57QNKc9sQt9ASRczWlIr"
secret = "7gwhn-gVp6J3we_vdj_bIlL1bGpuZdlp"

faceId1 = "1.png"
faceId2 = "3.png"

data = {"api_key": key, "api_secret": secret}
files = {"image_file1": open(faceId1, "rb"), "image_file2": open(faceId2, "rb")}
# file_2 = {"image_file2": open(faceId2, "rb")}
response = requests.post(compare_url, data=data, files=files)

req_con = response.content.decode('utf-8')
req_dict = JSONDecoder().decode(req_con)

print(req_dict)

#置信度，越高说明越像
confindence = req_dict['confidence']
print(confindence)

#将人脸框出来
face_rectangle_1 = req_dict['faces1'][0]['face_rectangle']
# print(face_rectangle_1)
face_rectangle_2 = req_dict['faces2'][0]['face_rectangle']

img1 = cv2.imread(faceId1)
img2 = cv2.imread(faceId2)

if confindence>=70:
    drawFace(face_rectangle_1,img1)
    drawFace(face_rectangle_2,img2)

#图片过大，调整下大小
img1 = cv2.resize(img1,(500,500))
img2 = cv2.resize(img2,(500,500))

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()