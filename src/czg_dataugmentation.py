import os
# import shutil
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import scipy.misc

tfe.enable_eager_execution()

file_name = os.listdir('C:\\Users\\fengmaniu\\Desktop\\zhongzhuan\\')
path = 'C:\\Users\\fengmaniu\\Desktop\\zhongzhuan\\'

for name in file_name:
    i = 0
    path_temp = path + '\\' + name
    file_namein = os.listdir(path_temp)
    ll = len(file_namein)
    print("ll is %d" % ll)
    if ll <= 0:
        os.rmdir(path_temp)
    '''
    else:
        png_path = path_temp + '\\' + file_namein[0]
        image_data = tf.gfile.FastGFile(png_path, 'rb').read()
        image_data = tf.image.decode_png(image_data)
    
        output = tf.image.random_flip_left_right(image_data)
        output = tf.image.random_hue(output, 0.3)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    
        output = tf.image.random_flip_up_down(image_data)
        output = tf.image.random_saturation(output, 0.1, 0.7)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    
        output = tf.image.random_flip_up_down(image_data)
        output = tf.image.random_flip_left_right(output)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    
        output = tf.image.random_hue(image_data, 0.3)
        output = tf.image.random_saturation(output, 0.1, 0.7)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    
        output = tf.image.random_brightness(image_data,0.5)
        output = tf.image.random_contrast(output, 0.1, 0.7)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    
        output = tf.image.random_contrast(image_data, 0.2, 0.6)
        output = tf.image.random_hue(output, 0.3)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    
        output = tf.image.random_flip_up_down(image_data)
        output = tf.image.random_contrast(output, 0.1, 0.7)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    
        output = tf.image.random_flip_up_down(image_data)
        output = tf.image.random_hue(output, 0.3)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    
        output = tf.image.random_flip_up_down(image_data)
        output = tf.image.random_contrast(output, 0.2, 0.6)
        i = i + 1
        scipy.misc.imsave(png_path[0:-4]+ str(i) + '.png', output)
    '''
    '''
    file = path + name + '\\'+ name + '_0' + '.png'
    print("czg: %s" % file)
    
    image_data = tf.gfile.FastGFile(file, 'rb').read()
    image_data = tf.image.decode_png(image_data)
    
    if i%2 == 0:
        output = tf.image.random_flip_left_right(image_data)
        output = tf.image.random_hue(output, 0.3)
        i = i + 1
    else:
        output = tf.image.random_flip_up_down(image_data)
        output = tf.image.random_saturation(output, 0.1, 0.7)
        i = i + 1
        
    scipy.misc.imsave(file, output)
    '''
