from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler 

#1. 정규화
##################### csv 로드 ###########################
path = './_data/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

# print(train_csv)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col = 0)

# print(test_csv)

##################### 결측치 처리 ###########################
# print("train_csv.shape: ", train_csv.shape)
# print("train_csv.isnull(): ", train_csv.isnull())
# print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
# print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
# print("train_csv.info(): ", train_csv.info())
# print("train_csv.shape: ", train_csv.shape)

##################### train.csv 데이터에서 x와 y를 분리 ###########################

x = train_csv.drop(['Outcome'], axis= 1 )
# print("x.shape: ", x.shape)

y = train_csv['Outcome']
# print("y.shape: ", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size = 0.81,
                                shuffle = True,
                                random_state = 1333
)

scaler = MinMaxScaler()
# scaler = StandardScaler()                 # StandardScaler 사용법  
scaler.fit(x_train)                         # 준비
x_train = scaler.transform(x_train)         # 변환
x_test = scaler.transform(x_test)
print('min/max: ',np.min(x_test), np.max(x_test))

print("x_train.shape, x_test.shape", x_train.shape, x_test.shape)
print("y_train.shape, y_test.shape", y_train.shape, y_test.shape)


#2.모델 구성                                        # 함수형 모델
intput1 = Input(shape=(8,))
dense1  = Dense(10, activation='linear')(intput1)
dense2  = Dense(20, activation='relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3  = Dense(50, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4  = Dense(100, activation='relu')(drop2)
drop3 = Dropout(0.3)(dense4)
dense5  = Dense(150, activation='relu')(drop3)
drop4 = Dropout(0.3)(dense5)
dense6  = Dense(200, activation='relu')(drop4)
drop5 = Dropout(0.3)(dense6)
dense7  = Dense(300, activation='relu')(drop5)
drop6 = Dropout(0.3)(dense7)
dense8  = Dense(200, activation='relu')(drop6)
drop7 = Dropout(0.3)(dense8)
dense9  = Dense(150, activation='relu')(drop7)
dense10  = Dense(100, activation='relu')(dense9)
dense11  = Dense(50, activation='relu')(dense10)
dense12  = Dense(20, activation='relu')(dense11)
output1  = Dense(1, activation='sigmoid')(dense12)

model = Model(inputs=intput1, outputs=output1)      # 함수 정의

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse'])

model.fit(x_train, y_train, epochs=100, batch_size=24,
          validation_split=0.2,
          verbose=1,
          )

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result', result)

y_predict = np.round(model.predict(x_test))

# print('===============================================')
# print(y_test[:5])                       # 앞에 5개만 출력
# print(y_predict[:5])                    # 앞에 5개만 출력
# print(np.round(y_predict[:5]))
# print('===============================================')

acc = accuracy_score(y_test, y_predict)
print('acc:', acc)

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
