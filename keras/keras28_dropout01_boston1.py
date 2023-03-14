from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

#1. 정규화
datasets = load_boston()
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

scaler = MinMaxScaler()                     # MinMaxScaler 사용법
# scaler = StandardScaler()                 # StandardScaler 사용법
# scaler = MaxAbsScaler()                   # MaxAbsScaler 사용법
# scaler = RobustScaler()                   # RobustScaler 사용법
scaler.fit(x_train)                         # 준비
x_train = scaler.transform(x_train)         # 변환
x_test = scaler.transform(x_test)
print('min/max: ',np.min(x_test), np.max(x_test))

#2. 모델 구성                                 # 함수형 모델
intput1 = Input(shape=(13,))
dense1  = Dense(20, activation='sigmoid')(intput1)
dense2  = Dense(50, activation='sigmoid')(dense1)
dense3  = Dense(100, activation='sigmoid')(dense2)
dense4  = Dense(150, activation='relu')(dense3)
drop1 = Dropout(0.2)(dense4)
dense5  = Dense(200, activation='relu')(drop1)
dense6  = Dense(150, activation='relu')(dense5)
dense7  = Dense(50, activation='relu')(dense6)
dense8  = Dense(30, activation='relu')(dense7)
output1  = Dense(1, activation='linear')(dense8)

model = Model(inputs=intput1, outputs=output1)  # 함수 정의

# model.summary()                                  # 레이어 보기

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

import datetime

date = datetime.datetime.now()
print('now',date)                                  # 2023-03-14 14:37:12.907742

date = date.strftime("%m%d_%H%M")                  # m = 달, d = 날짜 _ H = 시간, M = 분
print('strftime',date)     

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# EarlyStopping: 최소의 로스 지점을 찿을 수 있다
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,
                   restore_best_weights=True
                   )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                                            save_best_only=True,
                                            filepath=''.join([filepath, 'k27_', date, '_', filename])
                                            )

model.fit(x_train, y_train, epochs=100, batch_size=13, validation_split=0.2, verbose=1,
                 callbacks=[es])


# model.fit의 반환값
# print("======================================")
# print(hist)
# print("======================================")
# print(hist.history)
# print("======================================")
# print(hist.history['loss'])
# print("======================================")
# print(hist.history['val_loss'])
# print("======================================")
# hist -> history, validation

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)