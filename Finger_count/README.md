# Finger_count

<img src="https://github.com/GKrizz/Finger_count/assets/103564745/32bcecb0-064f-43c5-8581-5e091f68c5b5" width="700" height="700">

<img src="https://github.com/GKrizz/Finger_count/assets/103564745/14286440-bded-4000-91c7-a62b2e50bc5e" width="700" height="700">


# Hand Tracking Project

This project uses OpenCV and MediaPipe to perform real-time hand tracking and finger counting using a webcam. The program identifies the position of the hand landmarks and determines how many fingers are raised.

## Features

- **Real-time Hand Tracking**: Detects hand landmarks in real-time using MediaPipe.
- **Finger Counting**: Counts the number of fingers raised based on the position of the hand landmarks.
- **Visualization**: Draws the hand landmarks and connections on the video feed for visualization.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/GKrizz/Finger_count.git
    cd Finger_count
    ```

2. **Install Dependencies**

    ```bash
    pip install opencv-python mediapipe
    ```

3. **Run the Script**

    ```bash
    python hand_tracking.py
    ```

## Code Explanation

### Import Libraries

```python
import cv2
import mediapipe as mp

##Initialize MediaPipe Hands and Drawing Utilities

mpHands = mp.solutions.hands
Hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

##Define Finger and Thumb Coordinates

fingersCoordinate = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinate = (4, 3)

##Capture Video Feed
cap = cv2.VideoCapture(0)

##Main Loop for Hand Tracking
while cap.isOpened():
    upcount = 0
    success, img = cap.read()  # Read frame
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = Hands.process(converted_image)  # Process image for hand tracking
    handNo = 0
    lmList = []

    if results.multi_hand_landmarks:  # Check if hand landmarks are detected
        for id, lm in enumerate(results.multi_hand_landmarks[handNo].landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((cx, cy))
        
        for hand_in_frame in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_in_frame, mpHands.HAND_CONNECTIONS)
        
        for point in lmList:
            cv2.circle(img, point, 5, (0, 255, 0), cv2.FILLED)
        
        for coordinate in fingersCoordinate:
            if lmList[coordinate[0]][1] < lmList[coordinate[1]][1]:
                upcount += 1
        
        if lmList[thumbCoordinate[0]][0] > lmList[thumbCoordinate[1]][0]:
            upcount += 1
        
        cv2.putText(img, str(upcount), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (0, 0, 255), 12)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) == 113:  # Press 'q' to exit
        break

##Release Video Capture
cap.release()
cv2.destroyAllWindows()

