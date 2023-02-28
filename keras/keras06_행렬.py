import numpy as np

x1 = np.array([[1,2], [3,4]])
x2 = np.array([[[1,2,3]]])
x3 = np.array([[[1,2,3],[4,5,6]]])
x4 = np.array([[1],[2],[3]])
x5 = np.array([[[1]], [[2]], [[3]]])
x6 = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
x7 = np.array([[[1,2]], [[3,4]], [[5,6]], [[7,8]]])

# shape: 배열 출력
print('x1: ',x1.shape)     # (2, 2)
print('x2: ',x2.shape)     # (1, 1, 3)
print('x3: ',x3.shape)     # (1, 2, 3)
print('x4: ',x4.shape)     # (3, 1)
print('x5: ',x5.shape)     # (3, 1, 1)
print('x6: ',x6.shape)     # (2, 2, 2)
print('x7: ',x7.shape)     # (4, 1, 2) 


# 스칼라:                   개체 하나 (0차원)
# 벡터:                     스칼라의 모임 (1차원)
# 메트릭스:                 벡터의 모임 (2차원)
# tensor:                   3차원을 가리킴
# 4차원 tensor:             4차원을 가리킴