# 이진 분류
# 분류의 종류: 이진 or 다중
# 이전까지는 회귀

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# [1,2,3,4] 리스트
# 키 벨류

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)           # 판다스: describe
# print(datasets.feature_names)   # 판다스: columns

x = datasets['data']
y = datasets.target             # y data: 0 or 1

# print(x.shape)                  # (569, 30)
# print(y.shape)                  # (569,)                569개의 스칼라가 모여있는 벡터 output_dim = 1
# print(y)

x_train, x_test, y_train, y_test = train_test_split(
                                                    x,y,
                                                    shuffle=True,
                                                    random_state=333,
                                                    test_size = 0.2)

#2. 모델
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=30))
model.add(Dense(20, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(40, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(1, activation='sigmoid'))                       # 이진 분류: 마지막 노드에 sigmoid !!! (0과 0로 한정 시킨다)

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',     # binary_crossentropy: 이진값 !!!
              metrics=['accuracy', 'mse'])                      # metrics=['accuracy']: 훈련시 accuracy 적용, mse 출력 추가 (훈련에 영향을 미치지 않음)

model.fit(x_train, y_train, epochs=30, batch_size=8,            # mse: 실제값과 예측값 비교
          validation_split=0.2,
          verbose=1,
          )

#4. 평가
result = model.evaluate(x_test, y_test)
print('result', result)                                         # result [binary_crossentropy: 0.24, accuracy: 0.89, mse: 0.072]

y_predict = np.round(model.predict(x_test))

print('===============================================')
print(y_test[:5])                       # 앞에 5개만 출력
print(y_predict[:5])                    # 앞에 5개만 출력
print(np.round(y_predict[:5]))                                   # np.round: 반올림 !!!
print('===============================================')

#5. 예측

acc = accuracy_score(y_test, y_predict)
print('acc:', acc)                                                # accuracy: 0.89