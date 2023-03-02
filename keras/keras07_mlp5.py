# 예측: x = 9, 30, 210 -> 예상 y = 10, 1.9

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])            # 3, 10 (range함수로 범위 지정)

x = x.transpose()                                                    # 10, 3 !!!
print(x.shape)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])     # 2, 10

y = y.transpose()                                                    # 10, 2 !!!
print(y.shape)

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim = 3))                  # input_dim = x 열에 갯수와 동일
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))                                 # Dense = y 열에 갯수와 동일


#3단계. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 200, batch_size = 3)       # batch_size = 열에 갯수와 동일


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict([[9, 30, 210]])
print('[9, 30, 210] 예측값: ', result)               # [9.972497  1.8697326]
