import cv2
import mediapipe as mp

# Initialiser Mediapipe pour la détection du visage et du corps
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

# Connexions pour le visage (triangulation)
FACE_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION

# Connexions pour le corps (ajuster selon les besoins)
BODY_CONNECTIONS = mp_pose.POSE_CONNECTIONS

cap = cv2.VideoCapture(0)

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

        # Convertir l'image en BGR pour l'affichage avec OpenCV
        frame.flags.writeable = True

        # Dessiner les vecteurs pour le corps
        if pose_results.pose_landmarks:
            for (start_idx, end_idx) in BODY_CONNECTIONS:
                start_landmark = pose_results.pose_landmarks.landmark[start_idx]
                end_landmark = pose_results.pose_landmarks.landmark[end_idx]
                start_point = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                end_point = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Dessiner les vecteurs pour le visage
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for (start_idx, end_idx) in FACE_CONNECTIONS:
                    start_landmark = face_landmarks.landmark[start_idx]
                    end_landmark = face_landmarks.landmark[end_idx]
                    start_point = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                    end_point = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        cv2.imshow('Video Stream - Vectors on Keypoints', frame)

        # Quitter la vidéo en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()