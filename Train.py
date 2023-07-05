import numpy as np #파이썬에서 수치 계산을 위한 강력한 라이브러리
import os  #파이썬의 운영 체제 관련 기능을 제공하는 모듈
import glob #파일 경로 패턴 매칭을 위한 패턴 매칭 기능을 제공하는 모듈
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 사용할 GPU 장치를 설정합니다.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU 메모리 증가를 허용합니다.

actions = ['me', 'you', 'have', 'nothing', 'home', 'mountain', 'promise', 'name']  # 액션 목록을 정의합니다.

data = []
for action in actions:
    file_pattern = f'C:/Users/user/Project_CapStone/dataset/seq_{action}_*.npy'  # 데이터 파일의 경로 패턴을 정의합니다.
    file_list = glob.glob(file_pattern)  # 경로 패턴과 일치하는 모든 파일을 찾습니다.
    for file_path in file_list:
        data.append(np.load(file_path))  # 데이터 파일을 읽어와 리스트에 추가합니다.

data = np.concatenate(data, axis=0)  # 데이터를 모두 연결하여 하나의 배열로 만듭니다.
data.shape #배열의 형태를 나타내므로, (데이터의 개수, 시퀀스 길이, 특성 개수)와 같은 형태

x_data = data[:, :, :-1]  # 입력 데이터를 정의합니다. 마지막 열은 레이블입니다.
labels = data[:, 0, -1]  # 레이블을 정의합니다.

y_data = to_categorical(labels, num_classes=len(actions))  # 레이블을 원-핫 인코딩 형식으로 변환합니다.
y_data.shape # (라벨의 개수, actions 개수)

x_data = x_data.astype(np.float32)  # 입력 데이터를 float32 자료형으로 변환합니다.
y_data = y_data.astype(np.float32)  # 레이블 데이터를 float32 자료형으로 변환합니다.

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)  # 훈련 데이터와 검증 데이터로 분할합니다.

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),  # LSTM 레이어를 추가합니다.
    Dense(32, activation='relu'),  # 완전 연결 레이어를 추가합니다.
    Dense(len(actions), activation='softmax')  # 출력 레이어를 추가합니다.
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])  # 모델을 컴파일합니다.
model.summary()
"""
    model.summary()를 호출하면 다음과 같은 정보가 출력됩니다:

    모델의 총 파라미터 개수 | 각 레이어의 이름 | 각 레이어의 출력 형태 (batch 크기를 제외한 출력 크기) | 각 레이어의 파라미터 개수
"""


history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=40,
    callbacks=[
        ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),  # 검증 정확도가 가장 높을 때 모델을 저장합니다.
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')  # 검증 정확도 개선이 없는 경우 학습률을 감소시킵니다.
    ]
)
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

