# R2 결정 계수, 평가 지표 (절대)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이타
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
                                x,y,
                                train_size = 0.7,
                                # test_size = 0.3,
                                shuffle = True,
                                random_state = 1234
)

print("x_train: ", x_train)
print("x_test: ", x_test)
print("y_train: ", y_train)
print("y_test: ", y_test)

#2. 마델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일
model.compile(loss = "mse", optimizer = "adam")
model.fit(x_train, y_train, epochs = 200, batch_size = 1)   # x_train, y_train 학습

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_predict = model.predict(x_test)                # x_test 예측
# print("predict: ", y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)                 # 예측 값이, 실제 목적 변수의 값과 어느정도 일지하는가를 표시하는 지표 (절대), "1"에 가까운 수록 좋다
print("r2_score : ", r2)

