"""This script is used to process images (resizing, clahe...) and split a given dataset into a train, a valid and a test set"""

import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(images_path, masks_path, split=0.2):
    """ Load the images and the masks"""
    i = 0
    images = []
    masks = []

    images_files = sorted(os.listdir(images_path))
    for file in images_files:
        images.append(os.path.join(images_path,file))

    masks_files = sorted(os.listdir(masks_path))
    for file in masks_files:
        masks.append(os.path.join(masks_path,file))
    
    images = sorted(images)
    masks = sorted(masks)
    
    """ Split the data """
    split_size = int(len(images) * split)
    train_x, test_x = train_test_split(images, test_size = split_size, random_state = 42)
    train_y, test_y = train_test_split(masks, test_size = split_size, random_state = 42)
    
    return (train_x, train_y), (test_x, test_y)

def create_split_dataset_with_augment_data_option(images, masks, save_path, clahe_processing = True, augment = True):
    """Performing data augmentation and resize the images and masks"""
    
    H = 512
    W = 512
    
    if (clahe_processing):
        clahe_model = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total = len(images)):
        """Extracting the image name"""
        #name = x.split("\\")[-1].split(".png")[0]
        name = x.replace('\\', '/').split("/")[-1].split(".jpg")[0]
        #name = dir_name + "_" + x.split()
        
        """Read the image and mask"""
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        
        if clahe_processing == True:
            colorimage_b = clahe_model.apply(x[:,:,0])
            colorimage_g = clahe_model.apply(x[:,:,1])
            colorimage_r = clahe_model.apply(x[:,:,2])
            x = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
        
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image = x, mask =y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
            
            aug = VerticalFlip(p=1)
            augmented = aug(image = x, mask =y)
            x2 = augmented["image"]
            y2 = augmented["mask"]
            
            aug = Rotate(limit = 45, p=1)
            augmented = aug(image = x, mask =y)
            x3 = augmented["image"]
            y3 = augmented["mask"]
            
            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
            
        else:
            X = [x]
            Y = [y]
            
        idx = 0
        for i, m in zip(X,Y):
            i = cv2.resize(i, (W,H))
            m = cv2.resize(m, (W,H))
            m = m/255.0
            m = (m>0.5)*255
            
            if len(X) == 1:
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{idx}.jpg"
                tmp_mask_name = f"{name}_{idx}.jpg"
                
            images_path = os.path.join(save_path, "images/", tmp_image_name)
            masks_path = os.path.join(save_path, "masks/", tmp_mask_name)

            cv2.imwrite(images_path, i)
            cv2.imwrite(masks_path, m)
            
            idx += 1
        


if __name__ == "__main__":
    """ Load the dataset """
    dataset_path = "Data/patches_dataset"
    split_dataset_path = "Data/split_dataset"
    images_path = "Data/patches_dataset/images"
    masks_path = "Data/patches_dataset/masks"
    
    (train_x, train_y), (test_x, test_y) = load_data(images_path=images_path, masks_path=masks_path, split=0.2)

    print("Train: ", len(train_x))
    print("Test: ", len(test_x))

    create_dir(split_dataset_path + "/train_patches/images")
    create_dir(split_dataset_path + "/train_patches/masks")
    create_dir(split_dataset_path + "/test_patches/images")
    create_dir(split_dataset_path + "/test_patches/masks")
    
    create_split_dataset_with_augment_data_option(train_x, train_y, split_dataset_path + "/train_patches", clahe_processing=True, augment = False)
    create_split_dataset_with_augment_data_option(test_x, test_y, split_dataset_path + "/test_patches", clahe_processing=True, augment = False)