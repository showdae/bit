import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
 
#1. 데이터
x = np.array(                                               # input
    [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1,1, 1,1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])      # output

# x = x.transpose()
x = x.T

print(x.shape)
print(y.shape)

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=3))            # input_dim = 열에 갯수와 동일
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3단계. 컴파일, 훈련
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x, y, epochs = 100, batch_size = 5)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict([[10, 1.4, 0]])
print('[10, 1.4, 0] 예측값: ', result)

