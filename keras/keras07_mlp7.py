# 예측: x = 9 -> 예상 y = 10, 1.9, 0

import numpy as np                                                   # numpy: 행/열 계산 편하게 하기 위해서 사용
from tensorflow.keras.models import Sequential                       # keras: tensorflow2 api
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])

x = x.transpose()
print(x.shape)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])     

y = y.transpose()
print(y.shape)


#2. 모델
model = Sequential()
model.add(Dense(5, input_dim = 1))                  # input_dim = x 열에 갯수와 동일 (input)
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))                                 # Dense = y 열에 갯수와 동일 (output)


#3단계. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 300, batch_size = 1)       # batch_size = 열에 갯수와 동일


#4단계. 평가, 예측
loss = model.evaluate(x, y)                         # 일반적으로 훈련된 데이터보다 결과가 좋지 않다
print('loss: ', loss)

result = model.predict([[9]])
print('[9] 예측값: ', result)                       # [10.062435    1.9050872   0.22473612]