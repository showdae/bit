#1. 데이터
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1,2,3,4,5])                   # 통으로 들어간다 (10억개), 1차원 배열
y = np.array([1,2,3,5,4])


#2. 모델 구성
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
model.fit(x, y, epochs=110, batch_size=32)   # batch: 데이터의 연산 수 (defult: 32)


#4단계. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict([6])
print('[6]예측값: ', result)
