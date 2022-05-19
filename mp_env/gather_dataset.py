import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1  # 인식할 수 있는 손 개수
gesture = {
    0: "fist",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "rock",
    8: "spiderman",
    9: "yeah",
    10: "ok",
    11: "fy",
}  # 12가지의 제스처, 제스처 데이터는 손가락 관절의 각도와 각각의 라벨을 뜻한다.

# MediaPipe hands model
mp_hands = mp.solutions.hands
# 웹캠 영상에서 손가락의 뼈마디 부분, 연두색으로 표시되는 부분을 그릴 수 있게 도와주는 유틸리티
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands, min_detection_confidence=0.5, min_tracking_confidence=0.5
)  # 참고: https://google.github.io/mediapipe/solutions/hands#min_detection_confidence

# Gesture recognition data
file = np.genfromtxt(
    "C:\\Users\\jyj24\\Documents\\mediapipe-tutorial\\mp_env\\data\\gesture_train.csv", delimiter=","
)  # csv 파일을 파이썬에서 활용할 수 있는 numpy 배열로 변환
print(file.shape)

cap = cv2.VideoCapture(0)  # 카메라 열기. 0: default camera


def click(event, x, y, flags, param):
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        file = np.vstack((file, data))  # 배열 세로(vertical) 결합
        print(file.shape)


cv2.namedWindow("Dataset")  # 카메라창 생성 및 이름 설정
# 마우스 콜백 함수 설정 : cv2.setMouseCallback(윈도우, 콜백 함수, 사용자 정의 데이터)
cv2.setMouseCallback("Dataset", click)

while cap.isOpened():
    ret, img = cap.read()  # ret(프레임 읽기 성공 여부), img(읽어온 비디오의 한 프레임)
    if not ret:
        continue

    img = cv2.flip(img, 1)  # 좌우대칭(1), 상하대칭(0), 상하좌우대칭(-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR > RGB 색상 공간 변환

    # Processes an RGB image and returns the hand landmarks and handedness of each detected hand.
    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB > BGR 색상 공간 변환

    # "multi_hand_landmarks" field that contains the hand landmarks on each detected hand.
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))  # 21(hand landmarks 개수), 3(x, y, z축)

            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0,
                        13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # (20,3)

            # Normalize v
            # v = 벡터 / 벡터 크기 = 1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            # 내적의 값: v1 * v2 * cos(angle) = 1 * 1 * cos(angle) => angle 값을 바로 추출 가능

            # Get angle using arcos of dot product
            # arccos(코싸인 역함수): angle 값을 추출하기 위해 사용하는 것
            angle = np.arccos(
                np.einsum(
                    "nt,nt->n",
                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
                )
            )  # (15,)

            angle = np.degrees(angle)  # Convert radian to degree

            # dtype=np.float32 : 실수형; 단 정밀도 부동소수점형
            data = np.array([angle], dtype=np.float32)  # (1, 15)

            data = np.append(data, 11)  # 11 means fy(gesture[11])

            # res : [Landmark{x:_, y:_, z:_} Landmark{x:_, y:_, z:_} Landmark{x:_, y:_, z:_} ...(21개)]
            # mp_hands.HAND_CONNECTIONS : hands_connections.py 파일 참고. 손바닥과 각 손가락의 landmark id
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Dataset", img)

    # waitKey() 함수는 입력 키를 기다리는 함수로, imshow() 함수와 함께 사용
    # imshow() 함수를 붙잡아 두려면, waitKey() 함수를 사용해야 하기 때문
    # waitKey() 함수가 없다면, imshow() 함수는 순식간에 지나가서 우리 눈으로 볼 수가 없음
    if cv2.waitKey(1) == ord("q"):
        break

np.savetxt("C:\\Users\\jyj24\\Documents\\mediapipe-tutorial\\mp_env\\data\\gesture_train_fy.csv", file, delimiter=",")
