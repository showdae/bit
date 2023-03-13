from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping                 # EarlyStopping 클래스 사용
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

# 스케일러의 종류
# 4종류의 함수 사용법은 똑같다
from sklearn.preprocessing import MinMaxScaler 
# from sklearn.preprocessing import StandardScaler # StandardScaler: 평균점을 중심으로 데이터를 정규화한다
# from sklearn.preprocessing import MaxAbsScaler 최대 절대값
# from sklearn.preprocessing import RobustScaler 

#1. 정규화
datasets = load_digits()
x = datasets.data
y = datasets['target']

print('type(x)',type(x))
print('x', x)
print('y의 라벨값: ', np.unique(y))                      #  [0 1 2 3 4 5 6 7 8 9]
                                                        # np.unique(y): 값의 종류 리턴
                                                                                
##################### 요지점에서 원핫 인코딩 ###########################
# y (150,) -> (150,3) 변경 (케라스=to_categorical, 판다스=겟더미, 사이킷런=원핫인코더)
from tensorflow.keras.utils import to_categorical
print('==============================')
y = to_categorical(y)                                   # (1797, 10)
print("!!!", y.shape)
print('==============================')    

x_train, x_test, y_train, y_test = train_test_split(
                                x, y,
                                train_size = 0.81,
                                shuffle = True,
                                random_state = 1333
)

# 정규화 방법
#1 train / test 분리 후에 정규화 한다
#2 train 데이터만 먼저 정규화 해준다
#3 train 데이터 비율 test 데이터를 정규화 해준다
#4 test 데이터는 1을 넘어서도 상관 없다

# 주의사항: 모든 데이터를 정규화할 경우 과적합이 발생할 수 있다

scaler = MinMaxScaler()
# scaler = StandardScaler()                 # StandardScaler 사용법  
scaler.fit(x_train)                         # 준비
x_train = scaler.transform(x_train)         # 변환
x_test = scaler.transform(x_test)
print('min/max: ',np.min(x_test), np.max(x_test))

print('y_train', y_train)
print(np.unique(y_train, return_counts=True))           # array([5, 5, 5]: 값의 갯수 리턴

'''
#2. 모델
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=64))    # relu: 양수만 추출
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))               # softmax: 라벨들의 합이 1, 가장 놓은 라벨값을 추출 !!!
                                                        # output layer: 라벨의 갯수만큼 노드를 설정한다 !!!
                                                        # 원핫 인코딩: 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고,
                                                        #             다른 인덱스에는 0을 부여하는 벡터 표현 방식
                                                        #             가치에 대해서 평가를 못하게 하기 위해서 사용
'''

#2.모델 구성                                 # 함수형 모델
intput1 = Input(shape=(64,))                # 스칼렛 13개, 벡터 1개 (열의 형식을 적용)
dense1  = Dense(10, activation='relu')(intput1)
dense2  = Dense(20, activation='relu')(dense1)
dense3  = Dense(10, activation='relu')(dense2)
output1  = Dense(10, activation='softmax')(dense3)

model = Model(inputs=intput1, outputs=output1)  # 함수 정의

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # adam: 평타 이상의 성능
              metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=64,
          validation_split=0.2,
          verbose=1,
          )

#####################accuracy_score를 사용해서 스코어를 빼세요###########################
# 넘파이에서 0 or 1로 변환

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