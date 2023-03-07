#########################################
# 판다스 csv 파일 read
#########################################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

print("train_csv: ", train_csv)
print("train_csv.shape: ", train_csv.shape)     # (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col = 0)

print("test_csv: ", test_csv)
print("test_csv.shape: ", test_csv.shape)       # (6493, 8)


print("train_csv.columns: ", train_csv.columns)
'''
train_csv.columns:  Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
'''
print("test_csv.columns: ", test_csv.columns)
'''
test_csv.columns:  Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'],
      dtype='object')
'''
# print("train_csv.info(): ", train_csv.info())

# print("train_csv.describe(): ", train_csv.describe())

# print("type(train_csv): ", type(train_csv))

##################### 결측치 처리 ###########################
# 결측치 처리 1. 제거
# print(train_csv.isnull())
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
print("train_csv.info(): ", train_csv.info())
print("train_csv.shape: ", train_csv.shape)     # (10886, 11)


##################### train.csv 데이터에서 x와 y를 분리 ###########################
x = train_csv.drop(['casual', 'registered', 'count'], axis= 1)
print("x.shape: ", x.shape)

y = train_csv['count']
print('y.shape: ', y.shape)
x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size=0.7,
                                    random_state=777
)

print("x_train.shape, x_test.shape: ", x_train.shape, x_test.shape)
print("y_train.shape, y_test.shape: ", y_train.shape, y_test.shape)



#2. 모델
model = Sequential()
model.add(Dense(30, input_dim=8))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일
model.compile(loss = "mse", optimizer = "adam")
model.fit(x_train, y_train, epochs = 30, batch_size =24, verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
