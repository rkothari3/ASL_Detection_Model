import os
import sys
import cv2
import platform
import time

# Create data directory
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

def init_camera():
    """Initialize camera with platform-specific settings if needed"""
    # Try default approach first
    cap = cv2.VideoCapture(0)
    
    # If not working, try platform-specific approaches
    if not cap.isOpened():
        system = platform.system().lower()
        if system == 'windows':
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows DirectShow
            if not cap.isOpened():
                cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # Windows Media Foundation
        elif system == 'linux':
            cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # Linux V4L
        elif system == 'darwin':  # macOS
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS AVFoundation
    
    # Try alternative camera indices if still not working
    if not cap.isOpened():
        for i in range(1, 3):  # Try camera indices 1-2
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
    
    return cap

# Initialize camera
cap = init_camera()
if not cap.isOpened():
    print("Error: Could not open video capture. Please check your camera connection.")
    sys.exit(1)

# Try to set camera properties
try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
except Exception as e:
    print(f"Warning: Could not set camera properties: {e}")

# Create directories for each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# Create a single named window for the entire process
window_name = 'Data Collection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Main processing loop
for j in range(number_of_classes):
    print(f'Collecting data for class {j}')
    
    # Wait for user to press 'q' to start collection
    print(f"Press 'q' to start collecting data for class {j}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Retrying...")
            time.sleep(0.5)
            continue
        
        # Create a nice-looking overlay for instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 30), (frame.shape[1]-50, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Add text with better formatting
        cv2.putText(frame, f'Class {j+1} of {number_of_classes}', (70, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to start collection", (70, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow(window_name, frame)
        
        # Use a single waitKey call with proper checking
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
            
        # Check if window was closed manually
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    
    # Now collect the dataset
    counter = 0
    start_time = time.time()
    
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Retrying...")
            time.sleep(0.5)
            continue
        
        # Create a copy for display purposes
        display_frame = frame.copy()
        
        # Draw a semi-transparent background for text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (50, 30), (display_frame.shape[1]-50, 170), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0)
        
        # Add text with better formatting
        cv2.putText(display_frame, f'Class {j+1} - Capturing {counter+1}/{dataset_size}', 
                   (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f'Progress: {int((counter/dataset_size)*100)}%', 
                   (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw a progress bar
        bar_width = display_frame.shape[1] - 280
        bar_height = 20
        filled_width = int(bar_width * (counter / dataset_size))
        cv2.rectangle(display_frame, (140, 140), (140 + bar_width, 140 + bar_height), (255, 255, 255), 2)
        cv2.rectangle(display_frame, (140, 140), (140 + filled_width, 140 + bar_height), (0, 255, 0), -1)
        
        # Show the frame
        cv2.imshow(window_name, display_frame)
        
        # Save the original frame (without overlays)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        counter += 1
        
        # Single waitKey with proper checking to prevent window reappearing issues
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
            
        # Check if window was closed manually
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    
    print(f'Finished collecting {counter} images for class {j}')

# Properly release resources
cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
