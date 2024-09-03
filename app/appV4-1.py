import cv2
import mediapipe as mp
import numpy as np
import time

# Initialiser Mediapipe pour la détection du visage et du corps
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Connexions pour le visage (triangulation)
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

# Connexions pour le corps (ajuster selon les besoins)
BODY_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

last_tempo_update = time.time()
offset = 30  # Décalage en pixels pour les lignes rouges
red_line_duration = 0.1  # Délai pour que les lignes rouges apparaissent (100ms)
red_lines = []  # Liste pour stocker les lignes rouges avec leur temps d'apparition

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

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

        # Dessiner les vecteurs pour le corps avec les effets de couleur
        current_time = time.time()
        if current_time - last_tempo_update > 60 / 120:  # Valeur de tempo par défaut (120 BPM)
            last_tempo_update = current_time
            red_lines = []  # Réinitialiser les lignes rouges à chaque coup de tempo

        if pose_results.pose_landmarks:
            for (start_idx, end_idx) in BODY_CONNECTIONS:
                start_landmark = pose_results.pose_landmarks.landmark[start_idx]
                end_landmark = pose_results.pose_landmarks.landmark[end_idx]
                start_point = int(start_landmark.x * black_frame.shape[1]), int(start_landmark.y * black_frame.shape[0])
                end_point = int(end_landmark.x * black_frame.shape[1]), int(end_landmark.y * black_frame.shape[0])

                # Ajouter les lignes rouges avec offset
                if any(line[2] < current_time < line[2] + red_line_duration for line in red_lines):
                    cv2.line(black_frame, (start_point[0] + offset, start_point[1] + offset),
                             (end_point[0] + offset, end_point[1] + offset), (0, 0, 255), 4)
                
                # Ajouter les lignes rouges à la liste avec le temps actuel
                red_lines.append((start_point, end_point, current_time))

                # Lignes bleues sans offset
                cv2.line(black_frame, start_point, end_point, (255, 0, 0), 2)

        # Dessiner les vecteurs pour le visage en temps réel
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for (start_idx, end_idx) in FACE_CONNECTIONS:
                    start_landmark = face_landmarks.landmark[start_idx]
                    end_landmark = face_landmarks.landmark[end_idx]
                    start_point = int(start_landmark.x * black_frame.shape[1]), int(start_landmark.y * black_frame.shape[0])
                    end_point = int(end_landmark.x * black_frame.shape[1]), int(end_landmark.y * black_frame.shape[0])
                    cv2.line(black_frame, start_point, end_point, (0, 255, 0), 2)

        # Afficher l'image traitée
        cv2.imshow('Video Stream - Vectors on Black Background', black_frame)

        # Quitter la vidéo en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()