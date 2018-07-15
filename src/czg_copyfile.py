import os
import shutil
alllist=os.listdir(u"C:/Users/fengmaniu/Desktop/zhongzhuan1")
topath = "C:/Users/fengmaniu/Desktop/keras"
i = 0
for file in alllist:
    i = i + 1
    subfile = "C:/Users/fengmaniu/Desktop/zhongzhuan1/" + file
    filesub = os.listdir(subfile)[0]
    shutil.copyfile(subfile+'/'+filesub, 'C:/Users/fengmaniu/Desktop/keras/' + str(i) + '.png')