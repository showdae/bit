from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용
from sklearn.metrics import r2_score
import pandas as pd

# 스케일러의 종류
# 4종류의 함수 사용법은 똑같다
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler    # StandardScaler: 평균점을 중심으로 데이터를 정규화한다
from sklearn.preprocessing import MaxAbsScaler      #최대 절대값
from sklearn.preprocessing import RobustScaler 

#1. 정규화
##################### csv 로드 ###########################
path = './_data/ddarung/'                                   # . = study(현재폴더), / 하단

train_csv = pd.read_csv(path + 'train.csv',                 # 판다스 csv 파일 리딩
                        index_col = 0)                      # id 제거

test_csv = pd.read_csv(path + 'test.csv',                   # 판다스 csv 파일 리딩
                        index_col = 0)                      # id 제거

##################### 결측치 처리 ###########################
# 결측치 처리 1. 제거
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

##################### train.csv 데이터에서 x와 y를 분리 ###########################
x = train_csv.drop(['count'], axis= 1 )
print(x)

y = train_csv['count']
print(y)

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

# 정규화 방법
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

#2.모델 구성                                            # 함수형 모델
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

model = Model(inputs=intput1, outputs=output1)          # 함수 정의

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

#5. 그래프 출력
# 과적합 그래프 모양 체크
# 한글 타이틀은 깨짐
# 오버핏이란: 그래프 확인시 중간중간 튀는 값

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'                                                   
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker = '.', color = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker='.', color = 'blue', label = 'val_loss')
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('로스, 발로스')
plt.legend()
plt.grid()
plt.show()


'''
정규화 테스트 결과
노말
    loss 3532.9267578125
    r2 -0.07619412139537007

    loss 3390.755126953125
    r2 0.03499841418967753

    loss 3720.37841796875
    r2 -0.028389494341979127


스케일링
MinMaxScaler
    loss 2706.042724609375
    r2 0.313438303299426

    loss 2771.231201171875
    r2 0.266694543043391

    loss 2773.159912109375
    r2 0.2540985052813247

StandardScaler
    loss 2873.26025390625
    r2 0.25986038296227554

    loss 2866.111083984375
    r2 0.22595538400081372

    loss 2873.531982421875
    r2 0.2249937217424417

MaxAbsScaler
    loss 2745.104736328125
    r2 0.29193753015885016

    loss 2793.201171875
    r2 0.2629141716368739

    loss 2758.197509765625
    r2 0.27966464084474263

RobustScaler
    loss 2833.240478515625
    r2 0.28576739140556184

    loss 2802.8203125
    r2 0.28247633779015724

    loss 2885.658203125
    r2 0.23711433112752234
    
함수형 모델
    loss 2720.650390625
    r2 0.25058543580849335

    loss 2718.094482421875
    r2 0.2533365254471278

    loss 2774.20458984375
    r2 0.23986856017729252
'''