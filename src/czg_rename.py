# -*- coding: utf-8 -*-
"""
Created on Sun May  6 15:28:55 2018

@author: fengmaniu
"""

import os
import shutil

file_names = os.listdir('C:\\Users\\fengmaniu\\Desktop\\hhh\\')
path = 'C:\\Users\\fengmaniu\\Desktop\\hhh\\'

i = 0
for file in file_names:
    new_path = path+str(i)
    os.rename(path+file, new_path)
    pics = os.listdir(new_path)
    for pic in pics:
        os.rename(new_path+'\\'+pic, new_path+'\\'+str(i)+'.png')
    i = i + 1

j = 0
for file in file_names:
    new_path = path+str(j)
    if os.path.exists(new_path + '\\' + str(j)+'.png'):
        shutil.move(new_path + '\\' + str(j)+'.png', path)
    os.removedirs(new_path)
    j = j + 1
    '''
    print("czg: %s" % name)
    if name[-3:].lower() == 'jpg'.lower():
        folder = os.path.exists(path+name[0:3])
        if not folder: 
            os.makedirs(path+name[0:3])
            os.rename(path+name, path+name[0:3]+'.jpg')
            shutil.move(path+name[0:3]+'.jpg',path+name[0:3])
        else:
            folder = os.path.exists(path+name[0:3]+'2')
            if not folder: 
                os.makedirs(path+name[0:3]+'2')
                os.rename(path+name, path+name[0:3]+'2'+'.jpg')
                shutil.move(path+name[0:3]+'2'+'.jpg',path+name[0:3]+'2')
    '''