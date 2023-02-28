# [실습] 6을 예측 한다

# 전처리기
import numpy as np
from tensorflow.keras.models import Sequential  # 노란줄은 warring, Sequential class 호출
from tensorflow.keras.layers import Dense       # 노란줄은 warring, Dense class 호출

#1단계. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,6,4])

#2단계. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3단계. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=110)         # fit: 훈련


#4단계. 평가, 예측
loss = model.evaluate(x, y)         # evaluate: 평가 (원래는 새로운 데이터 값을 평가 한다)
print('loss: ', loss)

result = model.predict([6])         # predict: 예측 (4 x loss)
print('[6]예측값: ', result)
