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
# 오버플러우 언더플로우가 발생하지 않는다
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

# 스케일러의 종류
from sklearn.preprocessing import MinMaxScaler 
# from sklearn.preprocessing import StandardScaler # StandardScaler: 평균점을 중심으로 데이터를 정규화한다
# from sklearn.preprocessing import MaxAbsScaler
# from sklearn.preprocessing import RobustScaler

#1. 정규화
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))                  # type: 자료형 확인 <class 'numpy.ndarray'>
print(x)

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
                                    train_size = 0.8,
                                    stratify = y,
                                    random_state = 333
)

scaler = MinMaxScaler()
# scaler = StandardScaler()                 # StandardScaler 사용법  
scaler.fit(x_train)                         # 준비
x_train = scaler.transform(x_train)         # 변환
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))     # 0.0 1.0

'''
#2.모델 구성
model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=13,
                 validation_split=0.2, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
'''