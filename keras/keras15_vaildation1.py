from tensorflow.python.keras.models import Sequential       # 기존이랑 똑같다!
from tensorflow.python.keras.layers import Dense            # 기존이랑 똑같다!
import numpy as np

#1. 데이터
###학습 데이타###
x_train = np.array(range(1,11))     # [ 1  2  3  4  5  6  7  8  9 10] - 스칼라 10 (벡터1)
y_train = np.array(range(1,11))     # [ 1  2  3  4  5  6  7  8  9 10] - 스칼라 10 (벡터1)
print("x_train", x_train)
print("y_train", y_train)

x_val = np.array([14,15,16])        # [14 15 16]
y_val = np.array([14,15,16])        # [14 15 16]
print("x_val", x_val)
print("y_val", y_val)

###평가 데이타###
x_test = np.array([11,12,13])       # [11 12 13]
y_test = np.array([11,12,13])       # [11 12 13]
print("x_test", x_test)
print("y_test", y_test)

#2. 모델
model = Sequential()
model.add(Dense(5,activation='linear', input_dim= 1))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_data=(x_val, y_val))               # validation_data: 검증 데이타 파라메타

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                   # 평가
print("loss: ", loss)

result = model.predict([17])                            # 예측
print("result: ", result)


