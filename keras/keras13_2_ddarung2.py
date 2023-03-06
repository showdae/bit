#########################################
# 판다스 csv 파일로 출력, RMSE 함수 만들기
#########################################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd

#1. 데이터
path = './_data/ddarung/'

# train_csv = pd.read_csv('./_data/ddarung/train.csv')

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

# print(train_csv)
# print(train_csv.shape)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col = 0)

# print(test_csv)
# print(test_csv.shape)


print(train_csv.columns)

print(train_csv.info())

print(train_csv.describe())

print(type(train_csv))



##################### 결측치 처리 ###########################
# 결측치 처리 1. 제거!!!
# print(train_csv.isnull())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)

##################### train.csv 데이터에서 x와 y를 분리 ###########################
x = train_csv.drop(['count'], axis= 1 )
print(x)

y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size=0.7,
                                    random_state=777
)
                                                                                    # 결츨치 제거 후
print(x_train.shape, x_test.shape)                      # (1021, 9) (438, 9)    ->  (929, 9) (399, 9)
print(y_train.shape, y_test.shape)                      # (1021,) (438,)        ->  (929,) (399,)

#2. 모델
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

#3. 컴파일
model.compile(loss = "mse", optimizer = "adam")
model.fit(x_train, y_train, epochs = 100, batch_size =31, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2_score: ", r2)

def RMSE(y_test, y_predict):                                # 사용자 정의 함수 만드는 법!!!!
    return np.sqrt(mean_squared_error(y_test, y_predict))   # np.sqrt(): 루트 적용
rmse = RMSE(y_test, y_predict)                              # RMSE 함수 사용
print("rmse:", rmse)


##################### submission.csv 만들기 ###########################
# print(test_csv.insull().sum())                            # 요기도 결측치가 있네!!!
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)

submission['count'] = y_submit
print(submission)

submission.to_csv(path + "submit_0306.csv")