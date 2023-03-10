from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = fetch_covtype()
# print(datasets.DESCR)

# print('feature_names', datasets.feature_names)         # 판다스: columns
# feature_names ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']                                  # target 컬럼에 값을 넣어라
print(x.shape, y.shape)                                 # (581012, 54) (581012,)
print('y의 라벨값: ', np.unique(y))                      # [1 2 3 4 5 6 7]
                                                        # np.unique(y): 값의 종류 리턴


##################### 요지점에서 원핫 인코딩 ###########################
# y (150,) -> (150,3) 변경 (케라스=to_categorical, 판다스=겟더미, 사이킷런=원핫인코더)
# 케라스
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)                                   # 0번째 컬럼이 없을경우 자동으로 만들어줌
                                                        # np.unique(y): [1 2 3 4 5 6 7] 일 경우
y = np.delete(y, 0, axis=1)
print("y.shape: ", y.shape)                                   # (581012, 7)
print(y)
print('==============================')

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size = 0.8,
                                    stratify = y,
                                    random_state = 333
)

print('y_train', y_train)
print(np.unique(y_train, return_counts=True))

#2. 모델
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=54))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=162,
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

y_pred = model.predict(x_test)          # y_pred 값은 실수로 출력됨

print(y_test.shape)                     # (30, 3) 원핫이 되어 있음
print(y_test[:5])
print(y_pred.shape)                     # (30, 3) 원핫이 되어 있음
print(y_pred[:5])

y_test_acc = np.argmax(y_test, axis=1)  # axis=1: 각 행에 있는 열끼리 비교
y_pred = np.argmax(y_pred, axis=1)      # axis=1: 각 행에 있는 열끼리 비교

acc = accuracy_score(y_test_acc, y_pred)
print('accuracy_score: ', acc)
