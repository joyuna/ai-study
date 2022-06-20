import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
              [9,8,7,6,5,4,3,2,1,0]]
             )
y = np.array([11,12,13,14,15,16,17,18,19,20])       
# ValueError: Data cardinality is ambiguous: x sizes: 2,  y sizes: 10 
print(x.shape) #(2, 10)
print(y.shape) #(10,) 

x = x.T 
#=> 행과 열을 바꾸는건 .T 와 transpose 다
# x = x.transpose()
# x = x.reshape(10,2)
print(x)
print(x.shape) # (10, 2)

#[숙제] 모델을 완성하시오.
#예측 : [[10, 1.4, 0]]

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3)) # input_dim=2 의 2는 컬럼,열,특성 의 개수를 넣는다.
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #옵티마이저는 로스에 힘을 싣는 역할
model.fit(x, y, epochs=100, batch_size=1) #batch로 할 경우 3,3,3,1=> 4번 돌아감

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model. predict([[10, 1.4, 0]])
print('[10, 1.4, 0]의 예측값: ', result)

#loss :  0.00023812460131011903
#[10, 1.4, 0]의 예측값:  [[19.978045]]