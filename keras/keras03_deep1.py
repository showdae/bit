#1. 데이터
import numpy as np
x = np.array([1,2,3])                                       # 1차원 배열
y = np.array([1,2,3])

#2. 모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential              # Sequential: class (순차적)
from tensorflow.keras.layers import Dense                   # Dense: 일차 함수, layers: 노드의 층

model = Sequential()                                        # 모델 정의
model.add(Dense(3, input_dim=1))                            # 3: output data, 1: input data (input layer)
model.add(Dense(10))                                        # (hidden layer), 튜닝 한다
model.add(Dense(3))                                         # (hidden layer)
model.add(Dense(1))                                         # (output layer)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                 # mse: 평균제곱오차(Mean Squared Error) -값 제곱으로 상쇠
                                                            # mae: 평균절대오차(Mean Absolute Error)
                                                            # adam: 일단 디폴트로
model.fit(x, y, epochs=100)                                 # fit: 훈련시키는 함수

#4. test
# loss: 0.0018 (loss 값에는 음수가 없다)
# loss: 0.0386
# loss: 1.5381
# loss: 0.0015
# loss: 0.0334