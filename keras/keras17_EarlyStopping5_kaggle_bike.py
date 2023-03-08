from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

#1.데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)

test_csv = pd.read_csv(path + 'test.csv',
                        index_col = 0)

##################### 결측치 처리 ###########################
print("train_csv.shape: ", train_csv.shape)
print("train_csv.isnull(): ", train_csv.isnull())
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
train_csv = train_csv.dropna()          # 결측치 제거
print("train_csv.isnull().sum(): ", train_csv.isnull().sum())
# print("train_csv.info(): ", train_csv.info())
print("train_csv.shape: ", train_csv.shape)

##################### train.csv 데이터에서 x와 y를 분리 ###########################
x = train_csv.drop(['casual', 'registered', 'count'], axis= 1 )
print("x.shape: ", x.shape)

y = train_csv['count']
print("y.shape: ", y.shape)

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size=0.7,
                                    random_state=111
)
print("x_train.shape, x_test.shape", x_train.shape, x_test.shape)
print("y_train.shape, y_test.shape", y_train.shape, y_test.shape)

#2.모델 구성
model=Sequential()
model.add(Dense(20, input_dim=8, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용

# EarlyStopping: 최소의 로스 지점을 찿을 수 있다
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,  # EarlyStopping: patience 만큼 반복하여, min 값과 비교 후 중단
                                                                            # mode: auto or min
                   restore_best_weights=True                                # restore_best_weights: 브레이크 잡은 시점에서 최적의 W 값 저장, 디폴트: 0
                   )

hist = model.fit(x_train, y_train, epochs=100, batch_size=24, validation_split=0.2, verbose=1,
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

#5. 그래프 출력
# 과적합 그래프 모양 체크
# 한글 타이틀은 깨짐
# 오버핏이란: 그래프 확인시 중간중간 튀는 값

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'                                                   
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker = '.', color = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker='.', color = 'blue', label = 'val_loss')
plt.title('케글 바이크')
plt.xlabel('epochs')
plt.ylabel('로스, 발로스')
plt.legend()
plt.grid()
plt.show()
