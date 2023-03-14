from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler 

#1. 정규화
datasets = load_digits()
x = datasets.data
y = datasets['target']

# print('type(x)',type(x))
# print('x', x)
# print('y의 라벨값: ', np.unique(y))
                                                                                
##################### 요지점에서 원핫 인코딩 ###########################
# y (150,) -> (150,3) 변경 (케라스=to_categorical, 판다스=겟더미, 사이킷런=원핫인코더)
from tensorflow.keras.utils import to_categorical
# print('==============================')
y = to_categorical(y)
# print("!!!", y.shape)
# print('==============================')    

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

# print('y_train', y_train)
# print(np.unique(y_train, return_counts=True))


#2.모델 구성                                 # 함수형 모델
intput1 = Input(shape=(64,))
dense1  = Dense(10, activation='relu')(intput1)
drop1 = Dropout(0.3)(dense1)
dense2  = Dense(20, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3  = Dense(10, activation='relu')(drop2)
output1  = Dense(10, activation='softmax')(dense3)

model = Model(inputs=intput1, outputs=output1)  # 함수 정의

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=64,
          validation_split=0.2,
          verbose=1,
          )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print(results)
print('loss: ', results[0])
print('acc: ',  results[1])

y_pred = model.predict(x_test)

# print(y_test.shape)
# print(y_test[:5])
# print(y_pred.shape)
# print(y_pred[:5])

y_test_acc = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score: ', acc)