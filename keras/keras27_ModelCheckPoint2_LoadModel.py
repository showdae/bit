# 모델 파일 저장

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model                # save = model.save !!!
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

# print('type(x)',type(x))                  # type: 자료형 확인 <class 'numpy.ndarray'>
# print('x', x)

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size = 0.81,
                                shuffle = True,
                                random_state = 1333
)

scaler = MinMaxScaler()                       # MinMaxScaler 사용법

# scaler.fit(x_train)                         # 준비
# x_train = scaler.transform(x_train)         # 변환

xtrains = scaler.fit_transform(x_train)       # 준비 / 변환을 한줄로 축약
x_test = scaler.transform(x_test)
print('min/max: ',np.min(x_test), np.max(x_test))     # 0.0 1.0

'''
#2. 모델 구성                                 # 함수형 모델
intput1 = Input(shape=(13,))                # 스칼렛 13개, 벡터 1개 (열의 형식을 적용)
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

# save
# model.save('./_save/keras26_1_save_model.h5')   # h5: 확장자, 모델의 구조만 저장되어 있음 (가중치는 저장되지 않음)


#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='vall_loss', patience=10, mode='min',
                                            verbose=1,
                                            restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                                            save_best_only=True,
                                            filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5'    # csv 덮어쓰기
                                            )
                                            
model.fit(x_train, y_train, epochs=100, batch_size=13, validation_split=0.2, verbose=1,
                                            callbacks=[es, mcp]
                                            )

'''

model = load_model('./_save/MCP/keras27_ModelCheckPoint1.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)