import cv2

# Ouvre le flux vidéo depuis le caméscope
cap = cv2.VideoCapture(0)  # Si ton caméscope est connecté en tant que caméra par défaut

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo")
        break

    cv2.imshow('Video Stream', frame)

    # Quitter la vidéo en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
