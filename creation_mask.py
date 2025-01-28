"""This script is used in order to create the masks from the manual annotations made by the clinicians. (The code is not clean yet and need to be optimized)"""

import cv2
import os
import numpy as np

def compare_images(img_path1, img_path2, output_mask_path, output_image_path):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        print(f"Impossible de charger une ou plusieurs images : {img_path1} ou {img_path2}")
        return

    # S'assurer que les deux images ont la même taille
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    # Calculer la différence absolue entre les deux images
    diff = cv2.absdiff(img1, img2)
    # Convertir l'image différence en nuances de gris
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Appliquer un seuil pour binariser l'image
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Écrire directement l'image binaire comme résultat
    cv2.imwrite(output_image_path, img1)
    cv2.imwrite(output_mask_path, thresh)
    print(f"L'image montrant les différences a été sauvegardée sous : {output_mask_path}")

def compare_natif_pairs(directory):
    files = os.listdir(directory)
    for file in files:
        if file.endswith("NATIF.jpg"):
            original = os.path.join(directory, file)
            # Supposant que le fichier sans le "NATIF" soit juste le nom sans " NATIF"
            target_name = file.replace(" NATIF.jpg", ".jpg")
            if target_name in files:
                compare_path = os.path.join(directory, target_name)
                output_name = file.replace(" NATIF.jpg", "_mask.jpg")
                output_mask_path = os.path.join(directory+"masks/" ,output_name)
                output_image_path = os.path.join(directory+"images/" ,target_name)
                compare_images(original, compare_path, output_mask_path, output_image_path)
        elif file.endswith(" - Copie.jpg"):
            original_name = file.replace(" - Copie.jpg", ".jpg")
            if original_name in files:
                original = os.path.join(directory, original_name)
                compare_path = os.path.join(directory, file)
                output_name = file.replace(" - Copie.jpg", "_mask.jpg")
                output_mask_path = os.path.join(directory +"masks/", output_name)
                output_image_path = os.path.join(directory+"images/" ,original_name)
                compare_images(original, compare_path, output_mask_path, output_image_path)

if __name__ == "__main__":
    # Chemin vers le dossier contenant vos images NATIF et leurs correspondances
    directory = "Data/patches_combined_annotations_&_natif/"
    compare_natif_pairs(directory)
