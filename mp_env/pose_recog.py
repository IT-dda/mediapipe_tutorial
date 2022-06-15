import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
}

# MediaPipe Pose model
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('C:\\mediapipe-tutorial\\mp_env\\data\\pose_train_fy.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = pose.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pose_landmarks = result.pose_landmarks
    if pose_landmarks is not None:
      joint = np.zeros((33,4))
      for j, lm in enumerate(pose_landmarks.landmark):
        joint[j] = [lm.x, lm.y, lm.z, 0]
      
      v1 = joint[[2,11,0,0],:]
      v2 = joint[[5,12,11,12],:]
      v = v2 - v1
      v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

      angle = np.arccos(np.einsum('nt,nt->n',
      v[[0,1,2],:],
      v[[1,2,3],:]))

      angle = np.degrees(angle)
      data = np.array([angle], dtype=np.float32)
      ret, results, neighbours, dist = knn.findNearest(data, 3)
      idx = int(results[0][0])
      blue = (255, 0, 0)
      img = cv2.putText(img, "pose: "+ str(idx), (350,40), cv2.FONT_HERSHEY_PLAIN, 2, blue, 1, cv2.LINE_AA)

            # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Filter', img)
    if cv2.waitKey(1) == ord('q'):
        break
