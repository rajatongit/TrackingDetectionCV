import numpy as np
import cv2
# import os
# import matplotlib.pyplot as plt
# import scipy.io
# import imageio

def _random_rotate(img, name):
    h, w,_ = img.shape
    rotate_angle = np.random.random() * 20  - 10
    M = cv2.getRotationMatrix2D((w/2,h/2),rotate_angle,1)
    rotated_img = cv2.warpAffine(img,M,(w,h))
    cv2.imwrite(name, rotated_img)
        
def _random_scale(img, name):
    h,w,_ = img.shape
    random_scale = 1.2 + (np.random.random_sample(2))*0.5# something between 1.2 and 1.5 for each dim
    h_new, w_new = int(h*random_scale[0]),int(w*random_scale[1])
    scaled_img =  cv2.resize(img,(w_new,h_new))
    cv2.imwrite(name, scaled_img)
    
def _horizontal_flip(img, name):
    flipped_img = np.flip(img,1)
    cv2.imwrite(name, flipped_img)

def _vertical_flip(img, name):
    flipped_img = np.flip(img, 0)
    cv2.imwrite(name, flipped_img)
    
def _color_brightness_transform(img, name):
    # randomly change each channel (r,g,b) by +- 10%
    random_transform = 1 + (np.random.random_sample(3)*0.3 - 0.15)
    img_transform = np.ones(img.shape)
    img_transform[:,:,0] = random_transform[0]*img[:,:,0]
    img_transform[:,:,1] = random_transform[1]*img[:,:,1]
    img_transform[:,:,2] = random_transform[2]*img[:,:,2]
    img_transform[img_transform > 255] = 255
    color_transformed_img = np.uint8(img_transform)
    cv2.imwrite(name, color_transformed_img)
 
def read_image(augment_factor = 5):
    folder = [0, 1, 2, 3]
    images_in_folder = [53, 81, 51, 290]
    # imagePath = (imageFolderPath + ('%04d' % images_in_folder[0]) + '.jpg')
    # img_r = cv2.imread(imagePath)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    print("updated")    
    print("start read training")
    for folder_idx in range(len(folder)):
        imageFolderPath = (('%02d' % folder[folder_idx]) + '/')
        print('started reading images from ', folder[folder_idx])
        augmentation_idx = images_in_folder[folder_idx]
        for train_idx in range(0,images_in_folder[folder_idx]):
            imagePath = (imageFolderPath + ('%04d' % train_idx) + '.jpg')
            img_r = cv2.imread(imagePath)
            # cv2.imshow('img', img_r)
            # cv2.waitKey(0)
            # each_file_name = ('%04d' % train_idx) + '.jpg'
            # print(each_file_name)     
            # cv2.imwrite(imageFolderPath +  ('%04d' % augmentation_idx) + '.jpg', img_r)
            print(imagePath)
            # augment here
            for each_aug in range(augment_factor):
                _random_scale(img_r, imageFolderPath + ('%04d' % augmentation_idx) + '.jpg') # random scale
                augmentation_idx += 1
                _random_rotate(img_r, imageFolderPath +  ('%04d' % augmentation_idx) + '.jpg') # random rotate
                augmentation_idx += 1
                if np.random.random() > 0.5: # 50% chance of flip
                    _horizontal_flip(img_r, imageFolderPath +('%04d' % augmentation_idx) + '.jpg')
                    augmentation_idx += 1
                    _vertical_flip(img_r, imageFolderPath +('%04d' % augmentation_idx) + '.jpg')
                    augmentation_idx += 1
                _color_brightness_transform(img_r, imageFolderPath +('%04d' % augmentation_idx) + '.jpg') # color modify
                augmentation_idx += 1
                
if __name__ == "__main__":
    read_image(2)