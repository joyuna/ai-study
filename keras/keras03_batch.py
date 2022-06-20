import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

# [실습] 맹그러봐!!! [6]을 예측한다.

#2 모델구성
model = Sequential() 
model.add(Dense(10, input_dim=1)) 
model.add(Dense(100)) 
model.add(Dense(100)) 
model.add(Dense(20)) 
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=400, batch_size=1) #batch 값을 하나씩 작업하는 것(장점1.메모리를 적게 차지함, 장점2.훈련을 더 많이해서 loss,weight가 갱신됨, 단점1.작업시간이 오래 걸림)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss :  ', loss)

result=model.predict([6]) 
print('6의 예측값: ', result) 
 
# loss :   0.4349851608276367
# 6의 예측값:  [[6.025126]]