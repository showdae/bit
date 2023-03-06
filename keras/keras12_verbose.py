#########################################
# verbose: 학습 속도 향상됨 (딜레이 제거)
# 0 아무것도 안나온다
# 1 다 보여줘
# 2 프러그래스바만 없어져
# 3, 4, 5 ... 에포만 나온다

# 파라메타 설명
# keras.io -> API docs

# 케글
# https://www.kaggle.com/

# 데이콘
# https://dacon.io/
#########################################

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이타
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size = 0.7,
                                # test_size = 0.3,
                                shuffle = True,
                                random_state = 1333
)

#2. 모델
model = Sequential()
model.add(Dense(30, input_dim=13))
model.add(Dense(40))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일
model.compile(loss = "mae", optimizer = "adam")
model.fit(x_train, y_train, epochs = 10, batch_size = 2, verbose = 3)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose = 'auto')
print("loss: ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2_score : ", r2)
