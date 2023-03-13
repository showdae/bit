from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용
from sklearn.metrics import r2_score

# 스케일러의 종류
# 4종류의 함수 사용법은 똑같다
from sklearn.preprocessing import MinMaxScaler 
# from sklearn.preprocessing import StandardScaler # StandardScaler: 평균점을 중심으로 데이터를 정규화한다
# from sklearn.preprocessing import MaxAbsScaler 최대 절대값
# from sklearn.preprocessing import RobustScaler 

#1. 정규화
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

print('type(x)',type(x))
print('x', x)

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

#2.모델 구성
model=Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=24, validation_split=0.2, verbose=1)   # hist
print(hist.history)                                                                             # hist 출력

#4. 그래프 출력
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.show()