from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#1.데이터
##################### csv 로드 ###########################
path = './_data/dacon_wine/'
path_save = './_save/dacon_wine/'  

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
x = train_csv.drop(['quality', 'type'], axis= 1 )
print("x.shape: ", x.shape)         # (5497, 12)

y = train_csv['quality']
print("y.shape: ", y.shape)         # (5497,)

##################### 원핫 인코딩 ###########################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print("!!!", y.shape)               # (5497, 10)
print('==============================')

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size=0.7,
                                    random_state=111
)
print("x_train.shape, x_test.shape", x_train.shape, x_test.shape)       # (3847, 12) (1650, 12)
print("y_train.shape, y_test.shape", y_train.shape, y_test.shape)       # (3847,) (1650,)


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
intput1 = Input(shape=(11,))
dense1  = Dense(20, activation='relu')(intput1)
dense2  = Dense(50, activation='relu')(dense1)
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
output1  = Dense(10, activation='softmax')(dense8)

model = Model(inputs=intput1, outputs=output1)      # 함수 정의!!!

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # adam: 평타 이상의 성능
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=4,
          validation_split=0.2,
          verbose=1,
          )

#####################accuracy_score를 사용해서 스코어를 빼세요###########################
#4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print(results)
print('loss: ', results[0])
print('acc: ',  results[1])             # metrics=['acc']

y_pred = model.predict(x_test)

print(y_test.shape)                 # (30, 3) 원핫이 되어 있음
print(y_test[:5])
print(y_pred.shape)                 # (30, 3) 원핫이 되어 있음
print(y_pred[:5])

y_test_acc = np.argmax(y_test, axis=1)  # axis=1: 각 행에 있는 열끼리 비교
y_pred = np.argmax(y_pred, axis=1)      # axis=1: 각 행에 있는 열끼리 비교

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score: ', acc)

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping

# EarlyStopping: 최소의 로스 지점을 찿을 수 있다
# es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
#                    restore_best_weights=True
#                    )

model.fit(x_train, y_train, epochs=10, batch_size=12, 
                                                    # validation_split=0.2, 
                                                    # verbose=1,
                                                    # callbacks=[es]
                                                    )

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
