from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


#1. 정규화
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

# print('type(x)',type(x))
# print('x', x)

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size = 0.81,
                                shuffle = True,
                                random_state = 1333
)

scaler = MinMaxScaler()
# scaler = StandardScaler()                 # StandardScaler 사용법 
scaler.fit(x_train)                         # 준비
x_train = scaler.transform(x_train)         # 변환
x_test = scaler.transform(x_test)
# print('min/max: ',np.min(x_test), np.max(x_test))

#2.모델 구성                                      # 함수형 모델
intput1 = Input(shape=(8,))
dense1  = Dense(20, activation='relu')(intput1)
dense2  = Dense(50, activation='relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3  = Dense(100, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4  = Dense(50, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense4)
dense5  = Dense(30, activation='relu')(drop3)
output1  = Dense(1, activation='linear')(dense5)

model = Model(inputs=intput1, outputs=output1)   # 함수 정의

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=24, validation_split=0.2, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print("loss: ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)