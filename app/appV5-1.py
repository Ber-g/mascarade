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
tempo_interval = 60 / 120  # Intervalle de temps entre chaque battement de tempo (120 BPM)
offset_range = 30  # Décalage maximum en pixels pour les lignes rouges

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

        # Calculer l'offset dynamique
        current_time = time.time()
        time_since_last_tempo = current_time - last_tempo_update

        if time_since_last_tempo > tempo_interval:
            last_tempo_update = current_time
            time_since_last_tempo = 0  # Réinitialiser après chaque battement de tempo

        # Offset linéaire de 0 à offset_range selon le temps écoulé
        dynamic_offset = int((time_since_last_tempo / tempo_interval) * offset_range)

        if pose_results.pose_landmarks:
            for (start_idx, end_idx) in BODY_CONNECTIONS:
                start_landmark = pose_results.pose_landmarks.landmark[start_idx]
                end_landmark = pose_results.pose_landmarks.landmark[end_idx]
                start_point = int(start_landmark.x * black_frame.shape[1]), int(start_landmark.y * black_frame.shape[0])
                end_point = int(end_landmark.x * black_frame.shape[1]), int(end_landmark.y * black_frame.shape[0])

                # Ligne rouge avec offset dynamique positif
                cv2.line(black_frame, (start_point[0] + dynamic_offset, start_point[1] + dynamic_offset),
                         (end_point[0] + dynamic_offset, end_point[1] + dynamic_offset), (0, 0, 255), 4)
                # Ligne rouge avec offset dynamique négatif
                cv2.line(black_frame, (start_point[0] - dynamic_offset, start_point[1] - dynamic_offset),
                         (end_point[0] - dynamic_offset, end_point[1] - dynamic_offset), (255, 255, 0), 4)

                # Lignes bleues sans offset
                cv2.line(black_frame, start_point, end_point, (255, 0, 0),10)

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