#########################################
################[실습!!!]################
#########################################
# R2 0.62 이상
# train 0.7 ~ 0.9

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print("shape: ", x.shape, y.shape)  # (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size = 0.65,
                                shuffle = True,
                                random_state = 7777
)

x_test, x_val, y_test, y_val = train_test_split(
                                x_test, y_test,
                                train_size = 0.5,
                                shuffle = True,
                                random_state = 7777
)

#2. 모델
model = Sequential()
model.add(Dense(30, input_dim=10))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일
model.compile(loss = "mse", optimizer = "adam")
model.fit(x_train, y_train, epochs = 150, batch_size = 5,
            validation_split= 0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_predict = model.predict(x_test)       # x_test: 예측 input 값

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)        # 예측 값이, 실제 목적 변수의 값과 어느정도 일지하는가를 표시하는 지표 (절대), "1"에 가까운 수록 좋다
                                        # 보조 지표
print("r2_score : ", r2)                # 음수가 나올경우 값이 아주 안좋다 [r2_score :  0.5671414282825895]