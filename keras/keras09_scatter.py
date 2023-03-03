from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이타
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,6,5,7,8,9,11,10,12,13,15,14,16,17,18,19,20])

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
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일
model.compile(loss = "mse", optimizer = "adam")
model.fit(x_train, y_train, epochs = 10, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_predict = model.predict(x)                # x는 input 값 (input: x, output: y)

# 시각화
import matplotlib.pyplot as plt             # 그림 그리는 api

plt.scatter(x, y)                           # scatter: 점 찍는 함수
# plt.scatter(x, y_predict)
plt.plot(x, y_predict, color = "pink")      # plot: 라인 그리는 함수
plt.show()                                  # show: 디스플레이 함수

