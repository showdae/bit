from sklearn.datasets import load_breast_cancer
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
datasets = load_breast_cancer()
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

#2. 모델
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=30))
model.add(Dense(20, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(40, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(1, activation='sigmoid'))                       # 이진 분류: 마지막 노드에 sigmoid !!! (0과 0로 한정 시킨다)

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',     # binary_crossentropy: 이진값 !!!
              metrics=['accuracy', 'mse'])                      # metrics=['accuracy']: 훈련시 accuracy 적용, mse 출력 추가 (훈련에 영향을 미치지 않음)

model.fit(x_train, y_train, epochs=30, batch_size=8,            # mse: 실제값과 예측값 비교
          validation_split=0.2,
          verbose=1,
          )

#4. 평가
result = model.evaluate(x_test, y_test)
print('result', result)                                         # result [binary_crossentropy: 0.24, accuracy: 0.89, mse: 0.072]

y_predict = np.round(model.predict(x_test))

print('===============================================')
print(y_test[:5])                       # 앞에 5개만 출력
print(y_predict[:5])                    # 앞에 5개만 출력
print(np.round(y_predict[:5]))                                   # np.round: 반올림 !!!
print('===============================================')

#5. 예측

acc = accuracy_score(y_test, y_predict)
print('acc:', acc)                                                # accuracy: 0.89