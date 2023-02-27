#1. 데이터
import numpy as np      # import: 모듈을 가져올 수 있다 (numpy 데이터 형식)
                        # numpy: 과학 계산을 위한 라이브러리, 행렬/배열 처리 및 연산, 난수생성
x = np.array([1,2,3])
y = np.array([1,2,3])


#2. 모델 구성
import tensorflow as tf # tensorflow: 라이브러리
from tensorflow.keras.models import Sequential  # Sequential 가져온다
from tensorflow.keras.layers import Dense       # Dense 가져온다

model = Sequential()
model.add(Dense(1, input_dim = 1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 30)    # loss가 0이 나와야 가장 좋은값

#4. git test