from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler 

#1. 정규화
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

print('type(x)',type(x))
print('x', x)

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


#2.모델 구성                                 # 함수형 모델
intput1 = Input(shape=(30,))
dense1  = Dense(10, activation='relu')(intput1)
dense2  = Dense(20, activation='linear')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3  = Dense(30, activation='linear')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4  = Dense(40, activation='linear')(drop2)
drop3 = Dropout(0.3)(dense4)
dense5  = Dense(30, activation='linear')(drop3)
drop4 = Dropout(0.3)(dense5)
dense6  = Dense(20, activation='linear')(drop4)
output1  = Dense(1, activation='sigmoid')(dense6)

model = Model(inputs=intput1, outputs=output1)  # 함수 정의

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse'])

model.fit(x_train, y_train, epochs=30, batch_size=8,
          validation_split=0.2,
          verbose=1,
          )

#4. 평가
result = model.evaluate(x_test, y_test)
print('result', result)

y_predict = np.round(model.predict(x_test))

'''
print('===============================================')
print(y_test[:5])                       # 앞에 5개만 출력
print(y_predict[:5])                    # 앞에 5개만 출력
print(np.round(y_predict[:5]))
print('===============================================')
'''

#5. 예측

acc = accuracy_score(y_test, y_predict)
print('acc:', acc)