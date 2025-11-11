import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

DATA_DIR = 'asl_dataset'
os.makedirs(DATA_DIR, exist_ok=True)

labels = ['A', 'B', 'C']
for label in labels:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

cap = cv2.VideoCapture(0)
for label in labels:
    print(f"\nCollecting data for letter '{label}'...")
    print("Press 's' to save frame, 'n' to move to next letter, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(image, f'Letter: {label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Data Collection', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            data = []
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])
            np.savetxt(os.path.join(DATA_DIR, label,
                        f"{len(os.listdir(os.path.join(DATA_DIR, label)))}.csv"),
                       [data], delimiter=',')
            print("Frame saved.")
        elif key == ord('n'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
