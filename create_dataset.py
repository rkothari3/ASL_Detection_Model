import os
import pickle
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

# Initialize lists to store data and labels
data = []
labels = []

# Process each directory (class label) in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):  # Skip non-directory files
        continue

    # Process each image in the directory
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        data_aux = []
        x_list = []
        y_list = []

        # Read and convert image to RGB for Mediapipe processing
        img = cv2.imread(img_full_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks using Mediapipe
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_list.append(x)
                    y_list.append(y)

                # Normalize coordinates relative to the minimum x and y values
                # this is done so our model can focus on hand shapes rather than positions.
                # Example: If your hand is at the far left of the image, 
                # normalization ensures coordinates start at 0 instead of 100+ pixels.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_list))
                    data_aux.append(y - min(y_list))

            data.append(data_aux)
            labels.append(dir_)

# Serialize the data and labels using pickle
output_file = 'data.pickle'
with open(output_file, 'wb') as f: # wb -> write binary
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data serialized and saved to '{output_file}'")
