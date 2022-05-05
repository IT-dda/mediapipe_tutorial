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
hands = mp_hands.Hands(max_num_hands=max_num_hands,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
# 참고: https://google.github.io/mediapipe/solutions/hands#min_detection_confidence

# Gesture recognition model
file = np.genfromtxt(
    "C:\\Users\\jyj24\\Documents\\mediapipe-tutorial\\mp_env\\data\\gesture_train_fy.csv", delimiter=","
)  # 학습한 모델 가져오기, csv 파일을 파이썬에서 활용할 수 있는 numpy 배열로 변환

angle = file[:, :-1].astype(np.float32)  # 1열 ~ 마지막-1열 가져오기
label = file[:, -1].astype(np.float32)  # 마지막열(index) 가져오기

knn = cv2.ml.KNearest_create()  # KNN 알고리즘 객체 생성
# 모델 훈련. ROW_SAMPLE: each training sample is a row of samples
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)  # 카메라 열기. 0: default camera

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

            # Inference gesture : 추론
            # dtype=np.float32 : 실수형; 단 정밀도 부동소수점형
            data = np.array([angle], dtype=np.float32)  # (1, 15)

            # knn.findNearest(테스트 데이터 벡터가 행단위로 저장된 행렬, 사용할 최근접 이웃개수)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            # 5.0 [[5.]] [[5. 4. 5.]] [[3115.4932 4014.0447 4029.0078]]
            # print(ret, results, neighbours, dist)
            # ret : 첫번째 테스트 데이터에 대한 예측 결과
            # results : 테스트 데이터에 대한 모든 예측 결과 반환 (N, 1)
            # neighborResponses : 예측에 사용된 k개의 최근접 이웃 클래스 정보를 담고있는 행렬
            # dist : 입력 벡터와 예측에 사용된 k개의 최근접 이웃과의 거리를 저장한 행렬

            idx = int(results[0][0])

            if idx == 11:  # fy
                x1, y1 = tuple(
                    (joint.min(axis=0)[:2] * [img.shape[1], img.shape[0]] * 0.95).astype(int))
                x2, y2 = tuple(
                    (joint.max(axis=0)[:2] * [img.shape[1], img.shape[0]] * 1.05).astype(int))

                # 모자이크 할 손의 사각형 영역 가져오기
                fy_img = img[y1:y2, x1:x2].copy()
                # 모자이크 처리
                # ex) 2픽셀을 1픽셀로 압축한 뒤, 2픽셀 크기로 늘림
                fy_img = cv2.resize(fy_img, dsize=None, fx=0.05,
                                    fy=0.05, interpolation=cv2.INTER_NEAREST)  # dsize(절대크기)=None은 (0, 0). fxfy(상대크기)는 절대크기에 대한 상대 값
                fy_img = cv2.resize(fy_img, dsize=(
                    x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                # cv2.INTER_NEAREST: 최근방 이웃 보간법

                img[y1:y2, x1:x2] = fy_img

            # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Filter", img)

    # waitKey() 함수는 입력 키를 기다리는 함수로, imshow() 함수와 함께 사용
    # imshow() 함수를 붙잡아 두려면, waitKey() 함수를 사용해야 하기 때문
    # waitKey() 함수가 없다면, imshow() 함수는 순식간에 지나가서 우리 눈으로 볼 수가 없음
    if cv2.waitKey(1) == ord("q"):
        break
