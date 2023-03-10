# [과제]

# 3가지 웟핫 인코딩 방식을 비교할 것

#1. pandas의 get_dummies
# pd.get_dummies()함수는 명목변수만 원핫 인코딩을 해준다.
import pandas as pd

df  = pd.DataFrame()
df['명목변수']=['사과','바나나','배']
df['순위변수']=[2,1,0]

print(df)
print(df.shape)

df = pd.get_dummies(df)

print(df)
print(df.shape)

#2. keras의 to_categorical



#3. sklearn의 OneHotEncoder
# OneHotEncoder()는 명목변수든 순위변수든 모두 원핫 인코딩을 해준다.
from sklearn.preprocessing import OneHotEncoder

onehot=OneHotEncoder()
onehot_df=pd.DataFrame(onehot.fit_transform(df).toarray())
onehot_df

from sklearn.preprocessing import OneHotEncoder

onehot=OneHotEncoder()
nominal_df=pd.DataFrame(onehot.fit_transform(df[['명목변수']]).toarray())
nominal_df

# 미세한 차이를 정리하시오!!!