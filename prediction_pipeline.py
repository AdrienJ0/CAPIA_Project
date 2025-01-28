"""This script creates a pipeline where we can predict the segmentation of a given capillaroscopy"""

import tensorflow as tf
from keras.models import load_model
import cv2
from unet_training import dice_coef, dice_loss, iou
import numpy as np
import os

H = 512
W = 512

def segment_capillaroscopy(model, image_path):
    
    #image = tf_parse(image_path)
    
    image = cv2.imread(image_path)
    
    image = cv2.resize(image, (W,H))
    x = image/255.0
    x = np.expand_dims(x, axis=0)
    
    """ Prediction """
    y_pred = model.predict(x)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    
    return y_pred
    
    
if __name__ == "__main__":
    
    image_path = "Data/initial_dataset/images/N22a.jpg"
    
    unet_model = load_model("Models/model_with_mask_plein_clahe20240411/files/model.h5", custom_objects={'dice_loss':dice_loss, 'dice_coef':dice_coef, 'iou':iou}, compile=False)
    
    segmented_image = segment_capillaroscopy(unet_model, image_path)
    
    cv2.imwrite("segmented_image.jpg", segmented_image)
    cv2.imshow(segmented_image)
    


