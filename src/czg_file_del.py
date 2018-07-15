import os

file_name = os.listdir('C:\\Users\\fengmaniu\\Desktop\\zhongzhuantrain\\')
path = 'C:\\Users\\fengmaniu\\Desktop\\zhongzhuantrain\\'

for name in file_name:
    file_path = path + name
    pics = os.listdir(file_path)
    if(len(pics)>10):
        for i in range(8):
            os.remove(file_path+'\\'+pics[-i])
    

