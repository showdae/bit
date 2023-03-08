#0.전처리
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1.데이터
datasets = load_diabetes()
print("datasets.feature_names: ", datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, test_size=0.3)

#2.모델 구성
model=Sequential()
model.add(Dense(20, input_dim=10, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=30, validation_split=0.2, verbose=1)   # hist
print(hist.history)                                                                              # hist 출력

#4. 그래프 출력
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.show()
