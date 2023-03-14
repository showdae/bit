# 다중 분류
'''
소프트맥스 함수
input값을 [0,1] 사이의 값으로 모두 정규화하여 출력하며,
출력값들의 총합은 항상 1이 되는 특성을 가진 함수이다.
다중분류(multi-class classification) 문제에서 사용한다.
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_iris()
print('DESCR', datasets.DESCR)                           # 판다스: describe
'''
    :Number of Instances: 150 (50 in each of three classes)                 # 행
    :Number of Attributes: 4 numeric, predictive attributes and the class   # 열
    :Attribute Information:         # 열
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa       # 0
                - Iris-Versicolour  # 1
                - Iris-Virginica    # 2
'''
print('feature_names', datasets.feature_names)          # 판다스: columns
# feature_names ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                                 # (150, 4) (150,)   
print('x', x)
print('y', y)
print('y의 라벨값: ', np.unique(y))                      # [0 1 2]
                                                        # np.unique(y): 값의 종류 리턴

                
##################### 요지점에서 원핫 인코딩 ###########################
# y (150,) -> (150,3) 변경 (케라스=to_categorical, 판다스=겟더미, 사이킷런=원핫인코더)
# 케라스
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)                                   # (150, 3)

# 판다스
# import pandas as pd
# pd.get_dummies(y, )

#사이킷런


print("!!!", y.shape)
print('==============================')

x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    shuffle = True,
                                    train_size = 0.8,
                                    stratify = y,
                                    # random_state = 333
)

print('y_train', y_train)                                           # [1 0 0 2 0 2 0 2 0 2 1 2 1 1 1]
print('np.unique',np.unique(y_train, return_counts=True))           # array([5, 5, 5]: 값의 갯수 리턴

#2. 모델
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=4))    # relu: 양수만 추출
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))               # softmax: 라벨들의 합이 1, 가장 놓은 라벨값을 추출 !!!
                                                        # output layer: 라벨의 갯수만큼 노드를 설정한다 !!!
                                                        # 원핫 인코딩: 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고,
                                                        #             다른 인덱스에는 0을 부여하는 벡터 표현 방식
                                                        #             가치에 대해서 평가를 못하게 하기 위해서 사용

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # adam: 평타 이상의 성능
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=4,
          validation_split=0.2,
          verbose=1,
          )

#####################accuracy_score를 사용해서 스코어를 빼세요###########################
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
