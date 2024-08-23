import cv2
import mediapipe as mp
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands and face detection
mpHands = mp.solutions.hands
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mpFaceDetection.FaceDetection(0.75)

pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hand detection
    with mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Face detection
    results = mpFaceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255, 2))
            cv2.putText(img, f'{int(detection.score[0]*100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
