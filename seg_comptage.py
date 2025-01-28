"""This script is use to segment a group of capillaroscopies and count the number of capillaries in them"""

#import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
#import matplotlib.pyplot as plt

from unet_training import dice_coef, dice_loss, iou
import os

def clahe(image_init):
    clahe_model = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    colorimage_b = clahe_model.apply(image_init[:,:,0])
    colorimage_g = clahe_model.apply(image_init[:,:,1])
    colorimage_r = clahe_model.apply(image_init[:,:,2])
    image_clahe = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
    #print("type image_clahe clahe ", type(image_clahe))
    #print("taille image_clahe segmentation ", image_clahe.shape)
    return image_clahe

def count(image, image_name, txt_file):
    image = np.array(image, dtype=np.uint8)
    #blur = cv2.GaussianBlur(image, (11, 11), 0)
    canny = cv2.Canny(image, 0, 150, 7)
    dilated = cv2.dilate(canny, (1, 1), iterations=10)
    (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #res = image_name + " - " + (str(len(cnt)))+" capillaires détectés"
    print(image_name + " - " + (str(len(cnt)))+" capillaires détectés")
    txt_file.write(f"{image_name} - {(str(len(cnt)))} capillaires détectés\n")
    #return res

def segmentation(image_init, image_name):
    H = 512
    W = 512
    #print("type image_init segmentation ", type(image_init))
    #print("taille image_init segmentation ", image_init.shape)
    image_clahe = clahe(image_init)
    image_resize = cv2.resize(image_clahe, (W,H))
    x = image_resize/255.0
    x = np.expand_dims(x, axis=0)
    """ Prediction """
    y_pred = seg_model.predict(x)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    #print("type y_pred resulat segmentation ", type(y_pred))
    #print("taille y_pred segmentation ", y_pred.shape)
    #cv2.imwrite("res.jpg", y_pred)
    #y_pred = np.expand_dims(y_pred, axis=2)
    #res = count(y_pred, image_name)
    images_seg_comptage_infos_txt = "images_seg_comptage_infos_model_sclero_&_clahe.txt"
    txt_file = open(images_seg_comptage_infos_txt, "a")
    count(y_pred, image_name, txt_file)
    return y_pred


seg_model = load_model("Models/model_with_sclero_&_clahe_20240406/files/model.h5", custom_objects={'dice_loss':dice_loss, 'dice_coef':dice_coef, 'iou':iou}, compile = False)

source_folder = "images_comptage"

destination_folder = "images_comptage_seg_model_sclero_&_clahe"

# On s'assure que le dossier de destination existe
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

processed_files = []
for filename in os.listdir(source_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")) and filename not in processed_files:
        try:
            # Ajouter le nom du fichier à la liste des fichiers traités
            processed_files.append(filename)

            # Chemin complet de l'image d'origine 
            source_path = os.path.join(source_folder, filename)

            # Charger l'image à traiter
            image = cv2.imread(source_path)

            image_seg = segmentation(image, filename)

            filename_without_extension = os.path.splitext(filename)[0]
            new_filename = filename_without_extension + "_seg"

            # Chemin complet de l'image modifiée dans le dossier de destination
            destination_path = os.path.join(destination_folder, filename)

            # Enregistrer l'image modifiée dans le dossier de destination
            cv2.imwrite(destination_path, image_seg)

        except cv2.error:
            print(f"Image '{filename}' n'a pas pu être traitée")
