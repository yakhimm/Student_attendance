import tensorflow as tf
import skimage
import pandas as pd
import numpy as np
import cv2
import os

from tensorflow.keras.utils import load_img, img_to_array
from deepface import DeepFace
from PIL import ImageOps, Image
from os import walk

def img_to_csv(path, filename = None):
    new_path = next(walk(path), (None, None, []))[1]
    padding = (112, 112)
    image = []
    
    # Tiền xử lý hình ảnh
    def preprocessing(path_img):
        img = Image.open(path_img)
        # res_img = img.resize(padding)
        face_objs = DeepFace.extract_faces(
            img_path = path_img, 
            target_size = padding, 
            detector_backend = 'mtcnn'
            )
                
        ### ------------------------------------------------------###
        #Crop image and scale to 112x112
        x, y, w, h = face_objs[0]['facial_area'].values()
        box = (x, y, x + w, y + h)

        #Crop Image
        cropImage = img.crop(box).resize(padding)
        gray = skimage.color.rgb2gray(cropImage)
        # gray = skimage.color.rgb2gray(res_img)
       
        aimg = img_to_array(gray).reshape(padding[0] ** 2)
        
        return aimg

    if len(new_path) != 0:
        # Tập train
        for name, i in zip(new_path, range(len(new_path))):
            path_img = path + name + '/'
            list_img = next(walk(path_img), (None, None, []))[2]
            try:
                for img_name in list_img:
                    new_path_img = path_img + img_name
                    print()
                    print(new_path_img)
                    img = preprocessing(new_path_img)
                    img = np.append(img, i)
                    image.append(img)
            except:
                continue

    else:
        # Tập test
        list_img = next(walk(path), (None, None, []))[2]
        for img_name, i in zip(list_img, range(len(list_img))):
            new_path_img = path + img_name 
            print()
            print(new_path_img)
            img = preprocessing(new_path_img)
            image.append(img)
            
    df = pd.DataFrame(image)
    df.to_csv(filename, index = False)