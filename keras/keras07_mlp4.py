# 예측: x = 9, 30, 210 -> 예상 y = 10

# 전처리기
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])   # 0~9, 21~30, 201~210

x = x.transpose()   # 행/열 반전
print(x.shape)
print(x)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

y = y.transpose()   # 행/열 반전
print(y.shape)
print(y)


#2. 모델
model = Sequential()
model.add(Dense(5, input_dim = 3))                # input_dim = 열에 갯수와 동일
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


#3단계. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 100, batch_size = 3)   # batch_size = 열에 갯수와 동일


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict([[9, 30, 210]])
print('[9, 30, 210] 예측값: ', result)          # [10.721572]