"""This script is used to evaluate the performance of a model on a given test set"""

import os
import numpy as np
import pandas as pd
import cv2
import pathlib
from pathlib import Path
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from sklearn.utils import shuffle
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from unet_training import iou, dice_coef, dice_loss
import keras

############### TEST ###############
    
H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    ## i - m - y
    line = np.ones((H, 10, 3)) * 128

    """ Mask """
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)

    """ Predicted Mask """
    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    
    split_dataset_path = "Data/split_dataset"
    
    model_infos_path = "Models/model_with_mask_plein_clahe20241024"

    files_path = model_infos_path + "/files"
    
    model_path = os.path.join(files_path, "model.keras")
    csv_path = os.path.join(files_path, "data.csv")

    model_infos_text_path = os.path.join(files_path, "model_infos.txt")
    
    log_dir = ""
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    masks_pred_dir_path = model_infos_path + "/masks_predicted"
    results_path = model_infos_path + "/results"
    create_dir(masks_pred_dir_path)
    create_dir(results_path)

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = keras.saving.load_model(files_path + "/model.keras", compile=False)

    """ Load the dataset """
    test_x = sorted(glob(os.path.join(split_dataset_path, "test_mask_plein_clahe", "images", "*")))
    test_y = sorted(glob(os.path.join(split_dataset_path, "test_mask_plein_clahe", "masks", "*")))
    print(f"Test: {len(test_x)} - {len(test_y)}")

    print(zip(test_x, test_y))

    """ Evaluation and Prediction """
    SCORE = []

    for x, y in tqdm((zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x.replace('\\', '/').split("/")[-1].split(".jpg")[0]
        #name = x.split("/")[-1].split(".png")[0]
        
        mask_pred_path = os.path.join(masks_pred_dir_path,name + "_mask_pred.jpg")

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y = mask/255.0
        y = y > 0.5
        y = y.astype(np.int32)

        """ Prediction """
        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = results_path + f"/{name}.jpg"
        cv2.imwrite(mask_pred_path, y_pred) #Just in the masks predicted folder 
        save_results(image, mask, y_pred, save_image_path) #Save the concatenated images in the results folder 

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary", zero_division=1)
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])


########### SAVE RESULTS ###########


if __name__ == "__main__":

  """ Metrics values """
  score = [s[1:]for s in SCORE]
  score = np.mean(score, axis=0)

  #model_metrics_result_path = os.path.join(files_path, "metrics_result.txt")

  print(f"Accuracy: {score[0]:0.5f}")
  print(f"F1: {score[1]:0.5f}")
  print(f"Jaccard: {score[2]:0.5f}")
  print(f"Recall: {score[3]:0.5f}")
  print(f"Precision: {score[4]:0.5f}")

  #creating a text file with the command function "w"
  f = open(model_infos_text_path, "a")
  f.write(f"Accuracy: {score[0]:0.5f}\n")
  f.write(f"F1: {score[1]:0.5f}\n")
  f.write(f"Jaccard: {score[2]:0.5f}\n")
  f.write(f"Recall: {score[3]:0.5f}\n")
  f.write(f"Precision: {score[4]:0.5f}\n")
  f.close()

  df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
  df.to_csv(files_path + "/score.csv")