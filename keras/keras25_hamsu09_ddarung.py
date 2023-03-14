# 함수형 모델

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# 스케일러의 종류
# 4종류의 함수 사용법은 똑같다
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler    # StandardScaler: 평균점을 중심으로 데이터를 정규화한다
from sklearn.preprocessing import MaxAbsScaler      #최대 절대값
from sklearn.preprocessing import RobustScaler 

#1. 정규화
##################### csv 로드 ###########################
path = './_data/ddarung/'                                   # . = study(현재폴더), / 하단
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv',                 # 판다스 csv 파일 리딩
                        index_col = 0)                      # id 제거

test_csv = pd.read_csv(path + 'test.csv',                   # 판다스 csv 파일 리딩
                        index_col = 0)                      # id 제거

##################### 결측치 처리 ###########################
# 결측치 처리 1. 제거
# print(train_csv.isnull())
# print(train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
# print(train_csv.isnull().sum())
# print(train_csv.info())
# print(train_csv.shape)

##################### train.csv 데이터에서 x와 y를 분리 ###########################
x = train_csv.drop(['count'], axis= 1 )
# print(x)

y = train_csv['count']
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size=0.65,
                                    random_state=777
)

x_test, x_val, y_test, y_val = train_test_split(
                                    x_test, y_test,
                                    shuffle = True,
                                    train_size=0.5,
                                    random_state=777
)

##################### 정규화 ###########################
#1 train / test 분리 후에 정규화 한다
#2 train 데이터만 먼저 정규화 해준다
#3 train 데이터 비율 test 데이터를 정규화 해준다
#4 test 데이터는 1을 넘어서도 상관 없다

# 주의사항: 모든 데이터를 정규화할 경우 과적합이 발생할 수 있다

scaler = MinMaxScaler()                   # MinMaxScaler 사용법
# scaler = StandardScaler()                 # StandardScaler 사용법
# scaler = MaxAbsScaler()                   # MaxAbsScaler 사용법
# scaler = RobustScaler()                   # RobustScaler 사용법

scaler.fit(x_train)                         # 준비
x_train = scaler.transform(x_train)         # 변환 (train_csv)
x_test = scaler.transform(x_test)           # 변환 (train_csv)
test_csv = scaler.transform(test_csv)       # 변환 (test_csv)
print('min/max: ',np.min(x_test), np.max(x_test))

'''
#2.모델 구성
model=Sequential()
model.add(Dense(20, input_dim=9, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
'''

#2.모델 구성                                            # 함수형 모델!!! (다차원 사용시 많이 사용[이미지])
intput1 = Input(shape=(9,))                            # 스칼렛 13개, 벡터 1개 (열의 형식을 적용)
dense1  = Dense(20, activation='sigmoid')(intput1)
dense2  = Dense(50, activation='sigmoid')(dense1)
dense3  = Dense(100, activation='sigmoid')(dense2)
dense4  = Dense(150, activation='relu')(dense3)
dense5  = Dense(200, activation='relu')(dense4)
dense6  = Dense(150, activation='relu')(dense5)
dense7  = Dense(50, activation='relu')(dense6)
dense8  = Dense(30, activation='relu')(dense7)
output1  = Dense(1, activation='linear')(dense8)

model = Model(inputs=intput1, outputs=output1)          # 함수 정의!!!

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용

# EarlyStopping: 최소의 로스 지점을 찿을 수 있다
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,  # EarlyStopping: patience 만큼 반복하여, min 값과 비교 후 중단
                                                                            # mode: auto or min
                   restore_best_weights=True                                # restore_best_weights: 브레이크 잡은 시점에서 최적의 W 값 저장, 디폴트: 0
                   )

hist = model.fit(x_train, y_train, epochs=100, batch_size=13, validation_split=0.2, verbose=1,
                 callbacks=[es])                                            # EarlyStopping 호출

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
print("loss", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_predict, y_test)
print("r2", r2)

def RMSE(y_test, y_predict):                                                            # 사용자 정의 함수 만드는 법!!!!
    return np.sqrt(mean_squared_error(y_test, y_predict))                               # return: 반환, np.sqrt(): 루트 적용, mean_squared_error: mse
rmse = RMSE(y_test, y_predict)                                                          # RMSE 함수 호출
print("rmse score: ", rmse)

##################### submission.csv 만들기 ###########################
# print(test_csv.insull().sum())                            # 요기도 결측치가 있네!!!
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)

submission['count'] = y_submit
print(submission)

submission.to_csv(path_save + "ddarung_submit.csv")
