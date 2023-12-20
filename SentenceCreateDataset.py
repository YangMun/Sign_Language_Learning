import cv2 # OpenCV 라이브러리를 사용하여 이미지 및 비디오 처리를 위한 기능을 제공합니다. 이 코드에서는 비디오 캡처, 이미지 출력, 이미지 변환 등에 사용
import mediapipe as mp #  MediaPipe 라이브러리를 사용하여 손 인식 모델을 적용합니다. MediaPipe는 머신러닝 기반 비디오 및 오디오 분석을 위한 플랫폼
import numpy as np # NumPy 라이브러리를 사용하여 다차원 배열 및 수학적 연산을 수행합니다. 이 코드에서는 관절 위치와 각도 데이터를 다루기 위해 사용
import time, os #  time은 시간 관련 함수를 제공하며, os는 파일 및 폴더 조작을 위한 함수를 제공합니다. 이 코드에서는 시간 기록 및 데이터 저장 폴더 생성에 사용

actions = ['next', 'see']
seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands # MediaPipe 라이브러리의 손 모델을 사용하기 위해 mp.solutions.hands에서 손 모델을 가져와 mp_hands 변수에 할당
mp_drawing = mp.solutions.drawing_utils # 손 모델을 시각화하기 위한 도구를 제공하는 mp.solutions.drawing_utils 모듈을 가져와 mp_drawing 변수에 할당

hands = mp_hands.Hands( # 손 모델 객체를 생성, 최소한의 감지 신뢰도와 추적 신뢰도를 설정
    max_num_hands=2, # 동시에 인식할 손 개수 제한
    min_detection_confidence=0.5, # 동시에 인식할 손 개수 제한
    min_tracking_confidence=0.5) # 추적을 위한 최소 신뢰도

cap = cv2.VideoCapture(0) # 비디오 캡처 객체 생성 0은 기본 웹캠을 의미 웹캠으로부터 프레임을 읽어오기 위해 사용

created_time = int(time.time()) # 데이터 생성 시간 기록
os.makedirs('sentenceDataset', exist_ok=True)  # 데이터 저장 폴더 생성 이미 폴더가 존재하는 경우에도 오류를 발생시키지 않고 넘어감 이 코드에서는 데이터를 저장하기 위한 폴더를 생성하는데 사용

while cap.isOpened(): # 비디오 스트림이 열려 있는 동안 반복
    for idx, action in enumerate(actions):
        data = [] # 수집한 손 동작 데이터를 저장할 리스트 초기화

        ret, img = cap.read() # 프레임 읽어오기

        img = cv2.flip(img, 1)  # 이미지 좌우 반전

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2) # 화면에 동작 수집 대기 메시지 표시
        cv2.imshow('img', img) # 이미지 출력
        cv2.waitKey(3000) # 3초 대기

        start_time = time.time() # 동작 수집 시작 시간 기록

        while time.time() - start_time < secs_for_action: # 동작 수집 시간 동안 반복
            ret, img = cap.read()  # 프레임 읽어오기

            img = cv2.flip(img, 1) # 이미지 좌우 반전
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img) # 손 인식 수행
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None: # 손이 인식되었을 때
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4)) # 관절 위치를 저장할 배열 초기화
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] # 관절 위치와 가시성

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # v를 정규화
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # 각도로 변환

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx) # 각도 데이터와 동작 인덱스 결합

                    d = np.concatenate([joint.flatten(), angle_label])  # 관절 위치와 각도 데이터 결합

                    data.append(d) # 데이터 리스트에 추가

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 관절과 연결 선 그리기

            cv2.imshow('img', img) # 이미지 출력
            if cv2.waitKey(1) == ord('q'): # 'q' 키를 누르면 수집 중지
                break

        data = np.array(data) # 데이터를 NumPy 배열로 변환
        print(action, data.shape)
        np.save(os.path.join('sentenceDataset', f'raw_{action}_{created_time}'), data) # 데이터를 파일로 저장

        # 시퀀스 데이터 생성
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('sentenceDataset', f'seq_{action}_{created_time}'), full_seq_data) # 시퀀스 데이터를 파일로 저장
    break  # 첫 번째 동작만 수집 후 종료