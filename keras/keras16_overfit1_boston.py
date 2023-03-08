                                                        # 네이밍 룰
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential   # 대문자로 시작: Sequential(클래스)
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split    # 소문자로 시작: train_test_split(함수) - c언어 방식

#1.데이터
datasets = load_boston()
print("datasets.feature_names: ", datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                                                                         # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, test_size=0.3)

#2.모델 구성
model=Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=8, validation_split=0.2, verbose=1)    # hist

# model.fit의 반환값
print("======================================")
print(hist)
print("======================================")
print(hist.history)
print("======================================")
print(hist.history['loss'])
print("======================================")
print(hist.history['val_loss'])
print("======================================")
# hist -> history, validation

#4. 그래프 출력
# 과적합 그래프 모양 체크
# 한글 타이틀은 깨짐
# 오버핏이란: 그래프 확인시 중간중간 튀는 값


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'                                                   # 한글 깨질 경우 사용하는 함수 (Malgun Gothic 윈도우에서 제공하는 글씨체)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker = '.', color = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker='.', color = 'blue', label = 'val_loss')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('로스, 발로스')
plt.legend()
plt.grid()
plt.show()
