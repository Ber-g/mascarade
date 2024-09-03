import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import librosa
import time

# Initialiser Mediapipe pour la détection du visage et du corps
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Connexions pour le visage (triangulation)
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION
BODY_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Configuration audio pour analyser le tempo
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=1024)
tempo = 0.5  # Intervalle de changement de couleur en secondes
offset = 50  # Décalage en pixels
blue_line_duration = 0.1  # Délai pour que les lignes bleues disparaissent (100ms)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    last_time = time.time()
    blue_lines = []  # Liste pour stocker les lignes bleues avec leur temps d'apparition

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire le flux vidéo")
            break

        # Convertir l'image en RGB (Mediapipe attend des images en RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des points clés du corps et du visage
        pose_results = pose.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        # Créer une image noire de la même taille que la vidéo
        black_frame = np.zeros_like(frame)

        # Changer la couleur en fonction du tempo pour le corps seulement
        current_time = time.time()
        if current_time - last_time > tempo:
            color1, color2 = (0, 0, 255), (255, 0, 0)  # Rouge et bleu
            last_time = current_time
        else:
            color1, color2 = (255, 0, 0), (0, 0, 255)  # Bleu et rouge

        # Dessiner les vecteurs pour le corps avec les effets de couleur et d'offset
        if pose_results.pose_landmarks:
            for (start_idx, end_idx) in BODY_CONNECTIONS:
                start_landmark = pose_results.pose_landmarks.landmark[start_idx]
                end_landmark = pose_results.pose_landmarks.landmark[end_idx]
                start_point = int(start_landmark.x * black_frame.shape[1]), int(start_landmark.y * black_frame.shape[0])
                end_point = int(end_landmark.x * black_frame.shape[1]), int(end_landmark.y * black_frame.shape[0])

                # Lignes rouges sans offset
                cv2.line(black_frame, (start_point[0] , start_point[1] ),
                         (end_point[0] , end_point[1]), color1, 4)

                # Lignes bleues sans offset
               # cv2.line(black_frame, start_point, end_point, color2, 2)

                # Ajouter les lignes bleues à la liste avec le temps actuel
                blue_lines.append((start_point, end_point, current_time))

        # Dessiner les vecteurs pour le visage en vert sans effets
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for (start_idx, end_idx) in FACE_CONNECTIONS:
                    start_landmark = face_landmarks.landmark[start_idx]
                    end_landmark = face_landmarks.landmark[end_idx]
                    start_point = int(start_landmark.x * black_frame.shape[1]), int(start_landmark.y * black_frame.shape[0])
                    end_point = int(end_landmark.x * black_frame.shape[1]), int(end_landmark.y * black_frame.shape[0])

                    # Lignes vertes sans offset
                    cv2.line(black_frame, start_point, end_point, (0, 255, 0), 2)

        # Afficher les lignes bleues avec un délai avant de disparaître pour le corps
        for line in blue_lines:
            if current_time - line[2] < blue_line_duration:
                cv2.line(black_frame, line[0], line[1], (255, 0, 0), 2)

        # Supprimer les lignes bleues qui ont dépassé la durée
        blue_lines = [line for line in blue_lines if current_time - line[2] < blue_line_duration]

        cv2.imshow('Video Stream - Vectors on Black Background', black_frame)

        # Quitter la vidéo en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()