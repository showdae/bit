import numpy as np      # numpy: 자료형
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 행무시, 열우선 (열: 특성, 컬럼, 피쳐)
# 모델링을 할때 열의 갯수를 판단한다
 
#1. 데이터
x = np.array(           # 2차원 배열(빠른 연산)
    [[1,2,3,4,5,6,7,8,9,10],
     [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]
)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# x = x.transpose()       # 행/열 전환 함수
x = x.T                   # 행/열 전환 함수



print(x.shape)          # (10, 2) -> 2개의 특성을 가진 10개의 데이터
print(y.shape)          # (10,) -> 



#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=2))   # 열의 갯수와 동일하다
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))


#3단계. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=30, batch_size=3)        # 웨이트 생성됨


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

result = model.predict([[10, 1.4]])
print('[10, 1.4] 예측값: ', result)

