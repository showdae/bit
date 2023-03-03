# [검색] 트레인과 테스트를 섞어서 7:3으로 찿을 수 있는 방법!!!
# 힌트 사이킷런
# train, test set 분할

# 전처리기
import numpy as np                                      # 넘파이 자료형 (배열간의 연산) - 자료형
from tensorflow.keras.models import Sequential          # 모델 Sequential 클래스 사용 - 클래스
from tensorflow.keras.layers import Dense               # 층 Dense 클래스 사용 - 클래스
from sklearn.model_selection import train_test_split    # train_test_split 클래스 사용 - 함수

#1. 데이타
x = np.array([1,2,3,4,5,6,7,8,9,10])                    # x 배열 선언
y = np.array([1,2,3,4,5,6,7,8,9,10])                    # y 배열 선언

x_train, x_test, y_train, y_test = train_test_split(    # 4개 변수 선언, x,y 데이타 스플릿
    x, y,
    # train_size= 0.7,                                  # 훈련 데이타 비율
    test_size= 0.3,                                     # 평가 데이타 비율
    shuffle = True,                                     # 셔플 플래그
    random_state = 123                                  # 랜덤 시드 set (랜덤 난수표가 있음)
)

print("x_train: ", x_train)
print("x_test: ", x_test)
print("y_train: ", y_train)
print("y_test: ", y_test)

#2. 모델 (행무시, 열우선)
model = Sequential()                                    # 순차적 모델
model.add(Dense(5, input_dim=1))                        # in layer: input = 1, output = 5
model.add(Dense(7))                                     # hidden layer
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))                                     # out layer: output = 1

#3. 컴파일/훈련
model.compile(loss="mse", optimizer = "adam")           # 컴파일, mse: 절대값
model.fit(x_train, y_train, epochs=100, batch_size=5)   # 훈련, batch_size: 한번 학습할때 데이타의 수, 가중치가 결정됨

#4. 평가/예측
loss = model.evaluate(x_test, y_test)                   # 훈련되지 않은 데이타의 평가
print("평가[test 데이타: ]", loss)

result = model.predict([11])                            # 임의의 값 [11] 예측
print("예측[임의의값 11: ]", result)