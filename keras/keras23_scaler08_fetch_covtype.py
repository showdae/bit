from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

# 스케일러의 종류
# 4종류의 함수 사용법은 똑같다
from sklearn.preprocessing import MinMaxScaler 
# from sklearn.preprocessing import StandardScaler # StandardScaler: 평균점을 중심으로 데이터를 정규화한다
# from sklearn.preprocessing import MaxAbsScaler 최대 절대값
# from sklearn.preprocessing import RobustScaler 

#1. 정규화
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print('type(x)',type(x))
print('x', x)
print('y의 라벨값: ', np.unique(y))                      # [1 2 3 4 5 6 7]
                                                        # np.unique(y): 값의 종류 리턴

##################### 요지점에서 원핫 인코딩 ###########################
# y (150,) -> (150,3) 변경 (케라스=to_categorical, 판다스=겟더미, 사이킷런=원핫인코더)
# 케라스
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)                                   # 0번째 컬럼이 없을경우 자동으로 만들어줌!!!
                                                        # 
y = np.delete(y, 0, axis=1)
print("y.shape: ", y.shape)                                   # (581012, 7)
print(y)
print('==============================')

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size = 0.81,
                                shuffle = True,
                                random_state = 1333
)

# 정규화 방법
#1 train / test 분리 후에 정규화 한다
#2 train 데이터만 먼저 정규화 해준다
#3 train 데이터 비율 test 데이터를 정규화 해준다
#4 test 데이터는 1을 넘어서도 상관 없다

# 주의사항: 모든 데이터를 정규화할 경우 과적합이 발생할 수 있다

scaler = MinMaxScaler()
# scaler = StandardScaler()                 # StandardScaler 사용법  
scaler.fit(x_train)                         # 준비
x_train = scaler.transform(x_train)         # 변환
x_test = scaler.transform(x_test)
print('min/max: ',np.min(x_test), np.max(x_test))

print('y_train', y_train)
print(np.unique(y_train, return_counts=True))

#2. 모델
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=54))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))

# model.summary()                                             # 1,057


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

import time                                                  # 시간 측정하는 방법
start_time = time.time()
model.fit(x_train, y_train, epochs=3, batch_size=162,        # batch_size가 너무 크면 메모리가 터진다
          validation_split=0.2,
          verbose=1,
          )
edn_time = time.time()

print("훈련 시간: ", round(edn_time - start_time, 2), "초")    # round: 파이썬 자체의 round 함수

#####################accuracy_score를 사용해서 스코어를 빼세요###########################
# 넘파이에서 0 or 1로 변환

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print(results)
print('loss: ', results[0])
print('acc: ',  results[1])             # metrics=['acc']

y_pred = model.predict(x_test)          # y_pred 값은 실수로 출력됨

print(y_test.shape)                     # (30, 3) 원핫이 되어 있음
print(y_test[:5])
print(y_pred.shape)                     # (30, 3) 원핫이 되어 있음
print(y_pred[:5])

y_test_acc = np.argmax(y_test, axis=1)  # axis=1: 각 행에 있는 열끼리 비교
y_pred = np.argmax(y_pred, axis=1)      # axis=1: 각 행에 있는 열끼리 비교

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score: ', acc)
