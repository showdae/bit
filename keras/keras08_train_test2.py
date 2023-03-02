# 넘파이 리스트의 슬라이싱!!!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 분리
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1])

x_train = x[:7]                     # [1 2 3 4 5 6 7] get
x_test  = x[7:]                     # [8 9 10] get
# x_train = x[0]                    # 첫번째 요소 get
# x_train = x[3]                    # 네번째 요소 get
# x_train = x[-1]                   # 마지막 요소 get

y_train = y[:7]                     # [10 9 8 7 6 5 4]
y_test  = y[7:]                     # [3 2 1]

print(x_train)
print(x_test)

print(y_train)
print(y_test)

print(x_train.shape, x_test.shape)  # (7,) (3,)
print(y_train.shape, y_test.shape)  # (7,) (3,)