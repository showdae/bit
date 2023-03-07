##################### 전처리 ###########################
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd                                                                    # csv 사용시 편리한 자료형

##################### csv 로드 ###########################
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

# print("train_csv: ", train_csv)
print("train_csv.shape: ", train_csv.shape)             # (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col = 0)

# print("test_csv: ", test_csv)
print("test_csv.shape11111: ", test_csv.shape)          # (6493, 8)

# print("train_csv.columns: ", train_csv.columns)
# print("train_csv.info(): ", train_csv.info())
# print("train_csv.describe(): ", train_csv.describe())
# print("type(train_csv): ", type(train_csv))


##################### 결측치 처리 ###########################
print("train_csv.shape: ", train_csv.shape)                 # (10886, 11)
print("train_csv.isnull(): ", train_csv.isnull())
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
# print("train_csv.info(): ", train_csv.info())
print("train_csv.shape: ", train_csv.shape)                 # (10886, 11)

##################### train.csv 데이터에서 x와 y를 분리 ###########################
x = train_csv.drop(['casual', 'registered', 'count'], axis= 1 )
print("x.shape: ", x.shape)                                 # (10886, 8)

y = train_csv['count']
print("y.shape: ", y.shape)                                 # (10886,) - count만!!!

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size=0.7,
                                    random_state=777
)
print("x_train.shape, x_test.shape", x_train.shape, x_test.shape)           # (7620, 8) (3266, 8)
print("y_train.shape, y_test.shape", y_train.shape, y_test.shape)           # (7620,) (3266,) - count만!!!


##################### 모델 ###########################
model = Sequential()
model.add(Dense(30, input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))                                                                   # 활성화 함수란: 입력 신호의 총합을 출력 신호로 변환하는 함수를 일반적으로 활성화 함수라고 함
model.add(Dense(50, activation='linear'))                                               # linear: 디폴트 파라메타 (선형)
model.add(Dense(30, activation='relu'))                                                 # relu: 0 이하의 값은 다음 레이어에 전달하지 않습니다. 0이상의 값은 그대로 출력함
model.add(Dense(1))

##################### 컴파일/훈련 ###########################
model.compile(loss = "mse", optimizer = "adam")
model.fit(x_train, y_train, epochs = 50, batch_size =24, verbose=1)                    # fit: x_train, y_train 데이터
                                                                                        # epochs = train data / batch_size

##################### 평가/예측 ###########################
loss = model.evaluate(x_test, y_test)                                                   # evaluate: x_test, y_test 데이터
print("loss: ", loss)                                                                   # 24028.841796875

y_predict = model.predict(x_test)                                                       # predict: x_test 데이터

r2 = r2_score(y_test, y_predict)                                                        # r2_score: y_test, y_predict 데이터
print("r2_score: ", r2)                                                                 # 0.28019461661899747

def RMSE(y_test, y_predict):                                                            # 사용자 정의 함수 만드는 법!!!!
    return np.sqrt(mean_squared_error(y_test, y_predict))                               # return: 반환, np.sqrt(): 루트 적용, mean_squared_error: mse
rmse = RMSE(y_test, y_predict)                                                          # RMSE 함수 호출
print("rmse score: ", rmse)                                                             # 155.0123873512827

##################### submission.csv 만들기 ###########################
# print(test_csv.insull().sum())                                    # 요기도 결측치가 있네!!!
y_submit = model.predict(test_csv)                                  # 예측값
print("y_submit: ", y_submit)
'''
y_submit:  
[[64.96038 ]
 [61.56218 ]
 [61.56218 ]
 ...
 [61.406372]
 [63.52808 ]
 [60.952972]]
'''
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)    # sampleSubmission.csv 리드
print("submission: ", submission)
'''
submission456:                       count
datetime
2011-01-20 00:00:00      0
2011-01-20 01:00:00      0
2011-01-20 02:00:00      0
2011-01-20 03:00:00      0
2011-01-20 04:00:00      0
...                    ...
2012-12-31 19:00:00      0
2012-12-31 20:00:00      0
2012-12-31 21:00:00      0
2012-12-31 22:00:00      0
2012-12-31 23:00:00      0

[6493 rows x 1 columns]
'''
submission['count'] = y_submit          # test.csv 예측값을 submission 변수에 넣어준다
print("submission: ", submission)
'''
submission789:                            count
datetime
2011-01-20 00:00:00  64.960381
2011-01-20 01:00:00  61.562180
2011-01-20 02:00:00  61.562180
2011-01-20 03:00:00  63.277809
2011-01-20 04:00:00  63.277809
...                        ...
2012-12-31 19:00:00  63.553265
2012-12-31 20:00:00  63.553265
2012-12-31 21:00:00  61.406372
2012-12-31 22:00:00  63.528080
2012-12-31 23:00:00  60.952972

[6493 rows x 1 columns]
'''
submission.to_csv(path_save + "kaggle_bike_submit.csv")         # csv 출력 끝!!!
