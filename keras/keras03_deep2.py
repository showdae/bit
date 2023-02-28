#1단계. 데이터 준비 (가장 중요하다)
import numpy as np                  # 행렬/배열 처리 및 연산, 난수 생성
x = np.array([1,2,3])
y = np.array([1,2,3])


#2단계. 모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3단계. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200)         # fit: 훈련


#4단계. 평가, 예측
loss = model.evaluate(x, y)         # evaluate: 평가 (원래는 새로운 데이터 값을 평가 한다)
print('loss: ', loss)

result = model.predict([4])         # predict: 예측 (4 x loss)
print('[4]예측값: ', result)
