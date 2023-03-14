# 함수형 모델

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용
from sklearn.metrics import r2_score

# 스케일러의 종류
# 4종류의 함수 사용법은 똑같다
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import RobustScaler

#1. 정규화
datasets = load_boston()
x = datasets.data
y = datasets['target']

print('type(x)',type(x))                  # type: 자료형 확인 <class 'numpy.ndarray'>
print('x', x)

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
print('min/max: ',np.min(x_test), np.max(x_test))     # 0.0 1.0

'''
#2.모델 구성
model = Sequential()
model.add(Dense(20, input_dim=13, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
'''

#2.모델 구성                                                       # 함수형 모델
intput1 = Input(shape=(13,), name='s1')                           # 스칼렛 13개, 벡터 1개 (열의 형식을 적용)
dense1  = Dense(20, activation='sigmoid', name='s2')(intput1)     # name: 레이버 이름 변경
dense2  = Dense(50, activation='sigmoid')(dense1)
dense3  = Dense(100, activation='sigmoid')(dense2)
dense4  = Dense(150, activation='relu')(dense3)
dense5  = Dense(200, activation='relu')(dense4)
dense6  = Dense(150, activation='relu')(dense5)
dense7  = Dense(50, activation='relu')(dense6)
dense8  = Dense(30, activation='relu')(dense7)
output1  = Dense(1, activation='linear')(dense8)

model = Model(inputs=intput1, outputs=output1)                     # 함수 정의

model.summary()                                         # 레이어 보기

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

# EarlyStopping: 최소의 로스 지점을 찿을 수 있다
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,  # EarlyStopping: patience 만큼 반복하여, min 값과 비교 후 중단
                                                                            # mode: auto or min
                   restore_best_weights=True                                # restore_best_weights: 브레이크 잡은 시점에서 최적의 W 값 저장, 디폴트: 0
                   )

hist = model.fit(x_train, y_train, epochs=100, batch_size=13, validation_split=0.2, verbose=1,  # validation_split: 훈련시 모의모사 검증
                 callbacks=[es])                                            # EarlyStopping 함수 호출


# model.fit의 반환값
# print("======================================")
# print(hist)
# print("======================================")
# print(hist.history)
# print("======================================")
# print(hist.history['loss'])
print("======================================")
print(hist.history['val_loss'])
print("======================================")
# hist -> history, validation

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)


'''
#5. 그래프 출력
# 과적합 그래프 모양 체크
# 한글 타이틀은 깨짐
# 오버핏이란: 그래프 확인시 중간중간 튀는 값

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'                                                   
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker = '.', color = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker='.', color = 'blue', label = 'val_loss')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('로스, 발로스')
plt.legend()
plt.grid()
plt.show()
'''
