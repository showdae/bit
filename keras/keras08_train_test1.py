# 훈련 데이타와 평가 데이타를 나누어서 작성

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 분리
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1])

# print(x)
# print(y)

# 가중치 1씩 증가

x_train = np.array([1,2,3,4,5,6,7])     # x 훈련 데이타 선언
y_train = np.array([1,2,3,4,5,6,7])     # y 훈련 데이타 선언

x_test = np.array([8,9,10])             # x 평가 데이타 선언
y_test = np.array([8,9,10])             # y 평가 데이타 선언

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일/훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size = 2)     # 훈련 데이타를 매개변수로

#4. 평가/예측
loss = model.evaluate(x_test, y_test)                       # 평가 데이타를 매개변수로
print('평가: ', loss)

result = model.predict([11])                                # y 11번째 스칼렛 예측
print('예측[11]: ', result)