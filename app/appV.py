import cv2
import mediapipe as mp

# Initialiser Mediapipe pour la détection du visage et du corps
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

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

        # Dessiner les points clés et les vecteurs pour le corps
        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Dessiner les points clés et les vecteurs pour le visage
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for i, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        cv2.imshow('Video Stream - Vectors on Keypoints', frame)

        # Quitter la vidéo en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()