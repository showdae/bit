from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

#1.데이터
##################### csv 로드 ###########################
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'  

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col = 0)

##################### 결측치 처리 ###########################
# print("train_csv.shape: ", train_csv.shape)
# print("train_csv.isnull(): ", train_csv.isnull())
# print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
# print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
# print("train_csv.info(): ", train_csv.info())
# print("train_csv.shape: ", train_csv.shape)

##################### train.csv 데이터에서 x와 y를 분리 ###########################
x = train_csv.drop(['casual', 'registered', 'count'], axis= 1 )
# print("x.shape: ", x.shape)

y = train_csv['count']
# print("y.shape: ", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size=0.7,
                                    random_state=111
)
print("x_train.shape, x_test.shape", x_train.shape, x_test.shape)
print("y_train.shape, y_test.shape", y_train.shape, y_test.shape)

scaler = MinMaxScaler()
# scaler = StandardScaler()                 # StandardScaler 사용법
# scaler = MaxAbsScaler()                   # MaxAbsScaler 사용법
# scaler = RobustScaler()                   # RobustScaler 사용법  
scaler.fit(x_train)                         # 준비
x_train = scaler.transform(x_train)         # 변환 (train_csv)
x_test = scaler.transform(x_test)           # 변환 (train_csv)
test_csv = scaler.transform(test_csv)       # 변환 (test_csv)
print('min/max: ',np.min(x_test), np.max(x_test))


#2.모델 구성                                        # 함수형 모델!!! (다차원 사용시 많이 사용[이미지])
intput1 = Input(shape=(8,))
dense1  = Dense(20, activation='sigmoid')(intput1)
dense2  = Dense(50, activation='sigmoid')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3  = Dense(100, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4  = Dense(150, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense4)
dense5  = Dense(200, activation='relu')(drop3)
drop4 = Dropout(0.3)(dense5)
dense6  = Dense(150, activation='relu')(drop4)
drop5 = Dropout(0.3)(dense6)
dense7  = Dense(50, activation='relu')(drop5)
drop6 = Dropout(0.3)(dense7)
dense8  = Dense(30, activation='relu')(drop6)
output1  = Dense(1, activation='linear')(dense8)

model = Model(inputs=intput1, outputs=output1)      # 함수 정의!!!

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping

# EarlyStopping: 최소의 로스 지점을 찿을 수 있다
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                   restore_best_weights=True
                   )

model.fit(x_train, y_train, epochs=100, batch_size=24, validation_split=0.2, verbose=1,
                 callbacks=[es])

# model.fit의 반환값
# print("======================================")
# print(hist)
# print("======================================")
# print(hist.history)
# print("======================================")
# print(hist.history['loss'])
# print("======================================")
# print(hist.history['val_loss'])
# print("======================================")
# hist -> history, validation

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_predict, y_test)
print("r2", r2)

##################### submission.csv 만들기 ###########################
# print(test_csv.insull().sum())                                    # 요기도 결측치가 있네!!!
y_submit = model.predict(test_csv)                                  # 예측값
# print("y_submit: ", y_submit)

submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)    # sampleSubmission.csv 리드
# print("submission: ", submission)

submission['count'] = y_submit          # test.csv 예측값을 submission 변수에 넣어준다
# print("submission: ", submission)

submission.to_csv(path_save + "kaggle_bike_submit.csv")         # csv 출력 끝!!!
