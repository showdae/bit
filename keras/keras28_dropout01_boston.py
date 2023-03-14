# 과적합 배제(오버핏)
#1 데이터량을 늘린다
#2 일부 노드를 배제하고 훈련을 시킨다
#3 좋아질지 않좋아질지는 알 수 없다

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout            # Dropout: 노드 배제하는 함수
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

# print('type(x)',type(x))
# print('x', x)

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size = 0.81,
                                shuffle = True,
                                random_state = 1333
)

scaler = MinMaxScaler()

xtrains = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# print('min/max: ',np.min(x_test), np.max(x_test))

#2. 모델 구성                                 # 함수형 모델
intput1 = Input(shape=(13,))
dense1  = Dense(20, activation='sigmoid')(intput1)
dense2  = Dense(50, activation='sigmoid')(dense1)
dense3  = Dense(100, activation='sigmoid')(dense2)
dense4  = Dense(150, activation='relu')(dense3)
drop1 = Dropout(0.2)(dense4)
dense5  = Dense(200, activation='relu')(drop1)
dense6  = Dense(150, activation='relu')(dense5)
dense7  = Dense(50, activation='relu')(dense6)
dense8  = Dense(30, activation='relu')(dense7)
output1  = Dense(1, activation='linear')(dense8)

model = Model(inputs=intput1, outputs=output1)  # 함수 정의

'''
#2. 모델 구성                                   # 시퀀셜 모델
model = Sequential()
model.add(Dense(20, input_dim=13, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.3))                         # Dropout: 파라메타 값 만큼 노드 배제, evaluate에는 적용 안됨, 20% 정도의 과적합 방지를 해준다
                                                # evaluate에서는 모든 가중치가 적용됨
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))
'''

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

import datetime

date = datetime.datetime.now()
print('now',date)                                  # 2023-03-14 14:37:12.907742

date = date.strftime("%m%d_%H%M")                  # m = 달, d = 날짜 _ H = 시간, M = 분
print('strftime',date)                             # 0314_1437

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='vall_loss', patience=20, mode='min',
                                            verbose=1,
                                            restore_best_weights=True
                                            )

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                                            save_best_only=True,
                                            filepath=''.join([filepath, 'k27_', date, '_', filename])    # 빈 메모리가 하나 생긴다
                                            )
                                            
model.fit(x_train, y_train, epochs=100, batch_size=13, validation_split=0.2, verbose=1,
                                            # callbacks=[es, mcp]
                                            callbacks=[es] #, mcp]                                       # mcp csv 파일 출력 주석
                                            )


#4. 평가, 예측
print('======================= 1. 기본출력')
loss = model.evaluate(x_test, y_test, verbose=0)    
print("loss: ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2: ", r2)
