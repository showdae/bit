import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd                                                                    # csv 사용시 편리한 자료형

##################### csv 로드 ###########################
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

print("train_csv: ", train_csv)
print("train_csv.shape: ", train_csv.shape)  # (1459, 10)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col = 0)

print("test_csv: ", test_csv)
print("test_csv.shape: ", test_csv.shape)   # (715, 9)

print("train_csv.columns: ", train_csv.columns)
print("train_csv.info(): ", train_csv.info())
print("train_csv.describe(): ", train_csv.describe())
print("type(train_csv): ", type(train_csv))


##################### 결측치 처리 ###########################
print("train_csv.shape: ", train_csv.shape)  # (1459, 10)
print("train_csv.isnull(): ", train_csv.isnull())
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
print("train_csv.info(): ", train_csv.info())
print("train_csv.shape: ", train_csv.shape)  # (1328, 10)

##################### train.csv 데이터에서 x와 y를 분리 ###########################
x = train_csv.drop(['count'], axis= 1 )
print("x.shape: ", x.shape)         # x.shape:  (1328, 9) - error만 제거 (x!!!)

y = train_csv['count']
print("y.shape: ", y.shape)         # y.shape:  (1328,) - error만 추출 (y!!!)

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size=0.7,
                                    random_state=777
)
print("x_train.shape, x_test.shape", x_train.shape, x_test.shape)                      # (929, 9) (399, 9) - error만 제거 (x!!!)
print("y_train.shape, y_test.shape", y_train.shape, y_test.shape)                      # (929,) (399,) - error만 추출 (y!!!)

##################### 모델 ###########################
model = Sequential()
model.add(Dense(30, input_dim=9))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

##################### 컴파일/훈련 ###########################
model.compile(loss = "mse", optimizer = "adam")
model.fit(x_train, y_train, epochs = 100, batch_size =31, verbose=1)                    # fit: x_train, y_train 데이터

##################### 평가/예측 ###########################
loss = model.evaluate(x_test, y_test)                                                   # evaluate: x_test, y_test 데이터
print("loss: ", loss)

y_predict = model.predict(x_test)                                                       # predict: x_test 데이터

r2 = r2_score(y_test, y_predict)                                                        # r2_score: y_test, y_predict 데이터
print("r2_score: ", r2)

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