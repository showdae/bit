# [검색] 트레인과 테스트를 섞어서 7:3으로 찿을 수 있는 방법!!!
# 힌트 사이킷런
# train, test set 분할

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split    # 훈련 / 테스트 데이타 섞는 클래스


#1. 데이터 분리 (셔플 적용)
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    #train_size = 0.7,            # 훈련 데이타의 비율
    test_size = 0.3,              # 테스트 데이타의 비율
    shuffle = True,               # 데이타를 섞는다 (디폴트: True)
    random_state = 123            # 랜덤 Seed 고정 (파라메타 튜닝을 위함)
    )

print('X_train:', x_train)

print('X_test:', x_test)

print('y_train:', y_train)

print('y_test:', y_test)

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일/훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size = 1)     # 훈련 데이타를 매개변수로

#4. 평가/예측
loss = model.evaluate(x_test, y_test)                       # 평가 데이타를 매개변수로
print('평가: ', loss)

result = model.predict([11])                                # y 11번째 요소 예측
print('예측[11]: ', result)