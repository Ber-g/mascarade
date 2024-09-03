import cv2
import numpy as np
import cairosvg
from PIL import Image
import os
import time

# Charger le modèle pré-entraîné de détection d'objets
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Définir les classes pour MobileNetSSD
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor', 'phone']

# Chemin du dossier contenant les fichiers SVG
svg_folder = '/assets/PhoneDecoys/'
svg_files = [f for f in os.listdir(svg_folder) if f.endswith('.svg')]

# Fonction pour convertir SVG en PNG
def convert_svg_to_png(svg_path, png_path):
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    return cv2.imread(png_path)

# Dictionnaire pour stocker les images PNG
phone_images = {}
for svg_file in svg_files:
    svg_path = os.path.join(svg_folder, svg_file)
    png_path = svg_path.replace('.svg', '.png')
    phone_images[svg_file] = convert_svg_to_png(svg_path, png_path)

# Initialiser la caméra
cap = cv2.VideoCapture(0)

current_image_idx = 0
last_change_time = time.time()
image_change_interval = 30  # Changer d'image toutes les 30 secondes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo")
        break

    # Prétraiter l'image pour le modèle
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1/255.0, (300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Vérifier le temps écoulé pour changer l'image SVG
    current_time = time.time()
    if current_time - last_change_time > image_change_interval:
        current_image_idx = (current_image_idx + 1) % len(svg_files)
        last_change_time = current_time

    # Sélectionner l'image PNG à afficher
    svg_file = svg_files[current_image_idx]
    phone_image = phone_images[svg_file]

    # Boucle pour détecter les objets
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if classes[idx] == 'phone':
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ajuster la taille de phone_image à la taille du téléphone détecté
                phone_image_resized = cv2.resize(phone_image, (endX - startX, endY - startY))
                frame[startY:endY, startX:endX] = phone_image_resized

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()