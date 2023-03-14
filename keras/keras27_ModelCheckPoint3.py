# 

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.metrics import r2_score

# 스케일러의 종류
# 4종류의 함수 사용법은 똑같다
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import RobustScaler

#1. 데이터
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

scaler = MinMaxScaler()                       # MinMaxScaler 사용법

xtrains = scaler.fit_transform(x_train)       # 준비 / 변환을 한줄로 축약
x_test = scaler.transform(x_test)
print('min/max: ',np.min(x_test), np.max(x_test))     # 0.0 1.0


#2. 모델 구성                                 # 함수형 모델
intput1 = Input(shape=(13,))                 # 스칼렛 13개, 벡터 1개 (열의 형식을 적용)
dense1  = Dense(20, activation='sigmoid')(intput1)
dense2  = Dense(50, activation='sigmoid')(dense1)
dense3  = Dense(100, activation='sigmoid')(dense2)
dense4  = Dense(150, activation='relu')(dense3)
dense5  = Dense(200, activation='relu')(dense4)
dense6  = Dense(150, activation='relu')(dense5)
dense7  = Dense(50, activation='relu')(dense6)
dense8  = Dense(30, activation='relu')(dense7)
output1  = Dense(1, activation='linear')(dense8)

model = Model(inputs=intput1, outputs=output1)  # 함수 정의


#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='vall_loss', patience=10, mode='min',
                                            verbose=1,
                                            restore_best_weights=False
                                            )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                                            save_best_only=True,
                                            filepath='./_save/MCP/keras27_3_MCP.hdf5'    # csv 덮어쓰기
                                            )
                                            
model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=1,
                                            callbacks=[es, mcp]
                                            )

model.save('./_save/MCP/keras27_3_save_model.h5')

#4. 평가, 예측
print('======================= 1. 기본출력')
loss = model.evaluate(x_test, y_test, verbose=0)
print("loss: ", loss)                   # loss:  396.444091796875

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

print('======================= 2. load_model 출력')
model2 = load_model('./_save/MCP/keras27_3_save_model.h5')

loss = model2.evaluate(x_test, y_test, verbose=0)
print("loss: ", loss)                   # loss:  396.444091796875

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

print('======================= 3. MCP 출력')
model3 = load_model('./_save/MCP/keras27_3_MCP.hdf5')

loss = model3.evaluate(x_test, y_test, verbose=0)
print("loss: ", loss)                   # loss:  396.444091796875

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

'''
restore_best_weights=True
======================= 1. 기본출력
loss:  140.97093200683594
r2:  -0.8073412386378105
======================= 2. load_model 출력
loss:  140.97093200683594
r2:  -0.8073412386378105
======================= 3. MCP 출력
loss:  168.37847900390625
r2:  -1.1587241100161827


restore_best_weights=False
======================= 1. 기본출력
loss:  70.54857635498047
r2:  0.09552024337852727
======================= 2. load_model 출력
loss:  70.54857635498047
r2:  0.09552024337852727
======================= 3. MCP 출력
loss:  70.54857635498047
r2:  0.09552024337852727
'''
