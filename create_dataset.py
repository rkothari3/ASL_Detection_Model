import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

DATA_DIR = './data'

for dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
        # Convert img to rgb to provide it to mediapipe (for landmark detection)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# NOTE: START AT [15:00] @ https://www.youtube.com/watch?v=MJCSjXepaAM&t=141s
