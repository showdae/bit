import numpy as np

a = np.array([[1,2,3], 
              [6,4,5], 
              [7,9,2], 
              [3,2,1], 
              [2,3,1]])

print('a: ', a)
print('a.shape: ', a.shape)     # (5, 3)
# argmax: 벡터중 가장 높은 값을 리턴 한다

print(np.argmax(a))             # 7 : 7번째 위치값이 9 가장 높다
# axis = 0 = 행(row)
# axis = 1 = 열(col)
# axis = -1 = 가장 마지막 축, 이건 2차원이니까 가장 마지막축은 1
# 그래서 -1을 쓰면 이 데이터는 1과 동일 
print(np.argmax(a, axis=0))     # [2 2 1]: 행에서 가장 높은값 리턴
print(np.argmax(a, axis=1))     # [2 0 1 0 1]: 열에서 가장 높은값 리턴
print(np.argmax(a, axis=-1))    # [2 0 1 0 1]: 가장 마지막 축, 이건 2차원이니까 가장 마지막축은 1
