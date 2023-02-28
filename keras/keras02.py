#1. 데이터
import numpy as np                               # import: 모듈을 가져올 수 있다 (numpy 데이터 형식)
                                                 # numpy: 과학 계산을 위한 라이브러리, 행렬/배열 처리 및 연산, 난수생성
x = np.array([1,2,3])                            # 1차원 배열
y = np.array([1,2,3])


#2. 모델 구성
import tensorflow as tf                         # tensorflow: 라이브러리
from tensorflow.keras.models import Sequential  # Sequential: 가져온다
from tensorflow.keras.layers import Dense       # Dense: 가져온다

model = Sequential()
model.add(Dense(1, input_dim = 1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam') # loss가 0이 나와야 가장 좋은값
model.fit(x, y, epochs = 100)                    # fit: 훈련을 시키다, epochs: 훈련 횟수

#4. test
# loss: 17.5176
# loss: 1.2861
# loss: 22.9264
# loss: 0.9032
# loss: 5.5597e-04