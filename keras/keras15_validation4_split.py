# 실습: 잘라봐!!
# 실습: train_test_split 로만 잘라라
# 10:3:3

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential       # 기존이랑 똑같다!
from tensorflow.python.keras.layers import Dense            # 기존이랑 똑같다!
import numpy as np

#1. 데이터
###학습 데이타###
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(
                                        x, y,
                                        train_size=0.65,
                                        shuffle = False,
                                        random_state=777
                                        
)

x_test, x_val, y_test, y_val = train_test_split(
                                    x_test, y_test,
                                    shuffle = False,
                                    train_size=0.5,
                                    random_state=777
)

print("x_train",x_train)    # [ 1  2  3  4  5  6  7  8  9 10 ]
print("x_test", x_test)     # [ 11 12 13 ]
print("x_val", x_val)       # [ 14 15 16 ]
print("y_train", y_train)   # [ 1  2  3  4  5  6  7  8  9 10 ]
print("y_test", y_test)     # [ 11 12 13 ]
print("y_val", y_val)       # [ 14 15 16 ]

# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])


#2. 모델
model = Sequential()
model.add(Dense(5,activation='linear', input_dim= 1))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=1,
            validation_split=0.2)              # validation_data: 검증 데이타 파라메타

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

result = model.predict([17])
print("result: ", result)
