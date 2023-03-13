# 사이킷런 재설치
# pip uninstall scikit-learn
# pip install scikit-learn==1.1.0 (해당 버젼 설치)
# pip list
# scikit-learn                  1.1.0 (현재버젼)

# scaler

# 정규화
# 최대값을 최대값으로 나누어 0~1 사이로 만든다
# x - min
# -------
# max - min

# 장점
# 오버플로우 언더플로우가 발생하지 않는다
# 속도가 빨라진다
# 성능이 좋아질수도 있다
# input data를 0 ~ 1

# 단점
# 성능이 안 좋아질수도 있다

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

# scaler = MinMaxScaler()
# scaler.fit(x)                   # 준비
# x = scaler.transform(x)         # 변환

# print(np.min(x), np.max(x))     # 0.0 1.0

# 정규화 방법
#1 train / test 분리 후에 정규화 한다
#2 train 데이터만 먼저 정규화 해준다
#3 train 데이터 비율 test 데이터를 정규화 해준다
#4 test 데이터는 1을 넘어서도 상관 없다

# 주의사항: 모든 데이터를 정규화할 경우 과적합이 발생할 수 있다

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

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

# EarlyStopping: 최소의 로스 지점을 찿을 수 있다
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,  # EarlyStopping: patience 만큼 반복하여, min 값과 비교 후 중단
                                                                            # mode: auto or min
                   restore_best_weights=True                                # restore_best_weights: 브레이크 잡은 시점에서 최적의 W 값 저장, 디폴트: 0
                   )

hist = model.fit(x_train, y_train, epochs=10, batch_size=13, validation_split=0.2, verbose=1,  # validation_split: 훈련시 모의모사 검증
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

r2 = r2_score(y_predict, y_test)
print("r2: ", r2)


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
