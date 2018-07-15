from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

def createAug():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    rootdir = 'C:\\Users\\fengmaniu\\Desktop\\zhongzhuan'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    count = 0
    for index in range(0, len(list)):
        print (count)
        count = count +1
        path = rootdir + "/" + list[index]
        listname = os.listdir(path)
        for i in range(0, len(listname)):
            #print(path + '/' + listname[i])
            imgpath = path + '/' + listname[0]
            img = load_img(imgpath)  # 这是一个PIL图像
            x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
            x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
            #  下面是生产图片的代码
            #  生产的所有图片保存在 `preview/` 目录下
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=path+ "/", save_prefix=listname[i].split('.')[0], save_format='png'):
                i += 1
                if i > 10:
                    break  # 否则生成器会退出循环
                    
if __name__ == "__main__":
    createAug()