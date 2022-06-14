import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np

# Gesture recognition data
file = np.genfromtxt('C:\\mediapipe-tutorial\\mp_env\\data\\pose_train.csv', delimiter=',')
print(file.shape)

def click(event, x, y, flags, param):
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        file = np.vstack((file, data))
        print(file.shape)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 추가
    output_frame = image.copy()

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # 저장
    pose_landmarks = results.pose_landmarks
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
      data = np.append(data, 0)
      # pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
      # frame_height, frame_width = output_frame.shape[:2]
      # pose_landmarks *= np.array([frame_width, frame_height, frame_width])
      # pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.str).tolist()
      
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Dataset', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

np.savetxt('C:\\mediapipe-tutorial\\mp_env\\data\\pose_train_fy.csv', file, delimiter=',')