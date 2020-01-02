#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:25:51 2020

@author: Slaton
"""
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img
from keras.preprocessing.image import ImageDataGenerator



deep_dir = "/Users/Slaton/Documents/Maths/M2MO/CRM/image_test/cutted_deep/file_"

deep_ = load_img("/Users/Slaton/Documents/Maths/M2MO/CRM/image_test/IMG_9543.JPG")
array_deep = img_to_array(deep_)


def cut_image(img_array):
    img_height = img_array.shape[0]
    img_width = img_array.shape[1]
    img_list = list()
    
    for h in range(img_height):
        for w in range(img_width):
            if h%500==0 and w%400==0:
                img_ = array_deep[h:h+500,w:w+400]
                img_list.append(img_)
    return img_list
                
#image_list = cut_image(data)

deep_list = cut_image(array_deep)

def save_image(img_data):
    for i in range(len(img_data)):
        save_img(deep_dir + str(i) + '.jpeg',img_data[i])
        
        


        
datagen = ImageDataGenerator(rotation_range=30,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             fill_mode='nearest')

cutted_folder_deep = "/Users/Slaton/Documents/Maths/M2MO/CRM/image_test/cutted_deep"


def image_gen():
    cutted_images = os.listdir(cutted_folder_deep)
    for image in cutted_images:
        clipped = load_img(os.path.join(cutted_folder_deep,image))
        x_clipped = img_to_array(clipped)
        x_clipped = x_clipped.reshape((1,) + x_clipped.shape)
        i = 0
        for batch in datagen.flow(x_clipped,batch_size=1,
                          save_to_dir="/Users/Slaton/Documents/Maths/M2MO/CRM/image_test/deep_gen",
                          save_format="jpeg"):
                i+=1
                if i > 1000:
                    break


        



           


            
