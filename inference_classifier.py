import pickle
import cv2
import mediapipe as mp
import numpy as np
import time  # Add this import for tracking time

# Load the trained model from a file
modelDict = pickle.load(open('./model.pickle', 'rb'))
model = modelDict['model']

# Start capturing video from the default camera (usually webcam)
cap = cv2.VideoCapture(0)

# Set up MediaPipe for hand detection and drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hand detection model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map prediction numbers to letters
labelsDict = {0:'A', 1: 'B', 2: 'L'}

# Create a named window for displaying the video feed
window_name = 'Hand Gesture Recognition'
cv2.namedWindow(window_name)

# Initialize variables for sentence construction
current_letter = None
start_time = None
no_hand_start_time = None  # Track time when no hand is detected
sentence = ""

while True:
    # Lists to store hand landmark coordinates
    data_aux = []
    xList = []
    yList = []

    # Capture a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get the dimensions of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        no_hand_start_time = None  # Reset no-hand timer when a hand is detected

        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract and store x, y coordinates of landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                xList.append(x)
                yList.append(y)

            # Normalize the coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(xList))
                data_aux.append(y - min(yList))

        # Calculate the bounding box for the hand
        x1 = int(min(xList) * W) - 10
        y1 = int(min(yList) * H) - 10
        x2 = int(max(xList) * W) - 10
        y2 = int(max(yList) * H) - 10

        # Use the model to predict the hand gesture
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labelsDict[int(prediction[0])]

        # Check if the predicted character is the same as the previous one
        if predicted_character == current_letter:
            # If the same letter, check how long it has been held
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= 2.5:  # 2.5 seconds threshold
                sentence += predicted_character  # Add to the sentence
                start_time = None  # Reset the timer
        else:
            # If a new letter is predicted, reset the timer
            current_letter = predicted_character
            start_time = time.time()

        # Draw a semi-transparent rectangle around the hand
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 120, 255), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw the border of the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
        
        # Display the predicted letter with a background
        text_size = cv2.getTextSize(predicted_character, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
        cv2.rectangle(frame, (x1, y1 - 40), (x1 + text_size[0] + 10, y1), (0, 120, 255), -1)
        cv2.putText(frame, predicted_character, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                  1.3, (255, 255, 255), 3, cv2.LINE_AA)
    else:
        # If no hand is detected, start the no-hand timer
        if no_hand_start_time is None:
            no_hand_start_time = time.time()
        elif time.time() - no_hand_start_time >= 5:  # 5 seconds threshold
            sentence += " "  # Add a space to the sentence
            print("Space added to sentence")  # Print confirmation
            no_hand_start_time = None  # Reset the no-hand timer

    # Add instruction text and display the sentence
    cv2.putText(frame, "Press 'q' to quit", (10, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 
              0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Sentence: {sentence}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 
              0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow(window_name, frame)
    
    # Check for key press (1ms wait)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break
    
    # Check if the window was closed by clicking the X button
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
