#1. 데이터
import numpy as np # import numpy  numpy를 불러오다.
x = np.array([1,2,3]) # np.array  np의 정렬을 1,2,3으로
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential #텐서플로 케라스의 모델 폴더안에 있는 걸 순차적으로 불러오다.
from tensorflow.keras.layers import Dense #import Dense  밀도있게? 불러오다.

# 줄복사(여러줄 선택가능):  ctrl+c, ctrl+v
# 줄삭제(여러줄 선택가능):  shift+delete
# 실행:  ctrl+f5
# 주석처리:  ctrl+/

model = Sequential() #모델이 시퀀셜 모델로 정의하다.
model.add(Dense(4, input_dim=1)) #밀집점을 하나 추가(add)할거다. input_dim(dim=dimension 차원) 1차원을 넣다. 인풋레이어에 들어가는 데이터의 형태. 숫자4는 아웃풋
model.add(Dense(50)) #인풋은 4, 아웃풋은 50
model.add(Dense(30)) #인풋은 50, 아웃풋은 30
model.add(Dense(20)) #인풋은 30, 아웃풋은 20
model.add(Dense(1)) #인풋은 20, 아웃풋은 1

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')#컴퓨터한테 오차계산을 시킬건데 오차를 줄이는 식은 mse를 쓰겠다. mean squad 평균 제곱 오차(오차값이 음수일경우 상쇄되면 0이 되기 때문에 양수화해야한다. 방법 1.제곱, 2.절대값) 그리고 그것에 대한 최적화는 아담을 쓸거다.
model.fit(x, y, epochs=400) #훈련시킬거다. 데이터  x,y 500번(epochs)을 훈련시킬거야./ 앞에서 구한 최소의 오차값이 모델에 이미 들어가서 최적의 웨이트가 구해졌다.

#4. 평가, 예측
loss = model.evaluate(x, y) #평가 x와 y값을 넣은 평가값을 출력할거야
print('loss :  ', loss)

result=model.predict([4]) #model.predict([4])를  result라는 변수에 저장한다.
print('4의 예측값: ', result) #훈련량 조절해서 4를 만들어보자, 히든레이어를 조절한다. 레이어의 층을 늘린다.(1.훈련량조절 2.(히든)레이어조절 3.레이어층 4.노드조절 5.loss,optimizer 변경 등)=> 취미는 하이퍼파라미터 튜닝, **특기는 데이터전처리**
 
# loss :   1.3057506009772624e-07
# 4의 예측값:  [[3.999301]]
