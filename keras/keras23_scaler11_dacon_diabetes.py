from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# 스케일러의 종류
# 4종류의 함수 사용법은 똑같다
from sklearn.preprocessing import MinMaxScaler 
# from sklearn.preprocessing import StandardScaler # StandardScaler: 평균점을 중심으로 데이터를 정규화한다
# from sklearn.preprocessing import MaxAbsScaler 최대 절대값
# from sklearn.preprocessing import RobustScaler 

#1. 정규화
##################### csv 로드 ###########################
path = './_data/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

print(train_csv)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col = 0)

print(test_csv)

##################### 결측치 처리 ###########################
print("train_csv.shape: ", train_csv.shape)
print("train_csv.isnull(): ", train_csv.isnull())
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
# print("train_csv.info(): ", train_csv.info())
print("train_csv.shape: ", train_csv.shape)         # (652, 9)

##################### train.csv 데이터에서 x와 y를 분리 ###########################

x = train_csv.drop(['Outcome'], axis= 1 )
print("x.shape: ", x.shape)                         # (652, 8)

y = train_csv['Outcome']
print("y.shape: ", y.shape)                         # (652,)

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

print("x_train.shape, x_test.shape", x_train.shape, x_test.shape)
print("y_train.shape, y_test.shape", y_train.shape, y_test.shape)

#2. 모델
model = Sequential()
model.add(Dense(10, activation='linear', input_dim=8))
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse'])

model.fit(x_train, y_train, epochs=100, batch_size=24,
          validation_split=0.2,
          verbose=1,
          )

#4. 평가, 예측
result = model.evaluate(x_test, y_test)                     # 마지막 훈련 예측값 출력
print('result', result)

y_predict = np.round(model.predict(x_test))                 # np.round(): 반올림
                                                            # predict 함수를 사용하여 y 예측값 할당

# print('===============================================')
# print(y_test[:5])                       # 앞에 5개만 출력
# print(y_predict[:5])                    # 앞에 5개만 출력
# print(np.round(y_predict[:5]))
# print('===============================================')

acc = accuracy_score(y_test, y_predict)                     # 실제 y
print('acc:', acc)                                          # acc: 0.7251908396946565 (높을수록 좋다)

#5. csv 출력
# print(test_csv.insull().sum())
y_submit = np.round(model.predict(test_csv))
# print("y_submit: ", y_submit)

submission = pd.read_csv(path + "sample_submission.csv", index_col=0)
# print("submission: ", submission)

submission['Outcome'] = y_submit
# print("submission: ", submission)

path_save = './_save/dacon_diabetes/'
submission.to_csv(path_save + "dacon_diabetes_submit.csv")
