import os
import pickle

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir))[:1]:
        data_aux = []
        xList = []
        yList = []

        img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
        # Convert img to rgb to provide it to mediapipe (for landmark detection)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(dir)
                    # xList.append(x)
                    # yList.append(y)

        # plt.imshow(img_rgb)
        # plt.show()

# Serialize the data and labels using pickle
with open('data.pkl', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data serialized and saved to 'data.pkl'")


# NOTE: START AT [27:00] @ https://www.youtube.com/watch?v=MJCSjXepaAM&t=141s
# NOTE: Figure our the data, labels stuff cuz it diff in the yt vid and the github repo